import numpy as np
import random
import copy
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 2D Flocking Env with Fixed Map & Crash Logic
# ------------------------------
class FlockingEnv:
    def __init__(self, n_agents=3, n_obstacles=3, world_size=10.0, dt=0.1, max_steps=200,
                 neighbor_obs=2, obstacle_obs=2, max_speed=1.0, seed=None, save_map_path=None):
        if seed is not None:
            np.random.seed(seed)
        self.n = n_agents
        self.n_obstacles = n_obstacles
        self.world_size = world_size
        self.dt = dt
        self.max_steps = max_steps
        self.neighbor_obs = neighbor_obs
        self.obstacle_obs = obstacle_obs
        self.goal_threshold = 1.0
        self.obstacle_radius = 0.8
        self.max_speed = max_speed

        # Save map if requested (for reproducible visualization/testing)
        self.save_map_path = save_map_path

        # --- Generate Fixed Map (Goal & Obstacles) ONCE ---
        self.goal = np.array([self.world_size * 0.85, self.world_size * 0.85], dtype=np.float32)

        self.obstacles = []
        attempts = 0
        start_zone_center = np.array([self.world_size * 0.15, self.world_size * 0.15])

        while len(self.obstacles) < self.n_obstacles and attempts < 5000:
            attempts += 1
            obs_pos = np.random.rand(2) * (self.world_size * 0.8) + (self.world_size * 0.1)
            dist_to_goal = np.linalg.norm(obs_pos - self.goal)
            dist_to_start = np.linalg.norm(obs_pos - start_zone_center)
            if dist_to_goal > 2.5 and dist_to_start > 3.0:
                if all(np.linalg.norm(obs_pos - o) > (1.8 + 0.1) for o in self.obstacles):
                    self.obstacles.append(obs_pos)
        self.obstacles = np.array(self.obstacles, dtype=np.float32)

        if self.save_map_path is not None:
            np.savez(self.save_map_path, goal=self.goal, obstacles=self.obstacles)

        # dynamic variables
        self.reset()

    def load_map(self, mapfile):
        d = np.load(mapfile)
        self.goal = d['goal']
        self.obstacles = d['obstacles']

    def reset(self):
        # Agents start in small region near bottom-left (scaled by world_size).
        # Use reproducible small randomization centered at 0.15*world_size
        center = np.array([self.world_size * 0.15, self.world_size * 0.15])
        spread = self.world_size * 0.08
        self.pos = center + (np.random.rand(self.n, 2) - 0.5) * spread
        self.vel = (np.random.rand(self.n, 2) - 0.5) * (self.max_speed * 0.2)

        self.finished = np.zeros(self.n, dtype=bool)
        self.crashed = np.zeros(self.n, dtype=bool)

        # track previous distances for delta shaping
        self.prev_dist_to_goal = np.linalg.norm(self.pos - self.goal, axis=1)

        self.t = 0
        return self._get_obs()

    def step(self, actions):
        # Always define clipped actions (fix bug when all inactive)
        a = np.clip(actions, -1.0, 1.0)

        # Active mask: only agents NOT finished and NOT crashed act
        active_mask = ~(self.finished | self.crashed)
        a = a.copy()
        a[~active_mask] = 0.0

        # store previous state to compute delta rewards
        prev_pos = self.pos.copy()
        prev_dist_to_goal = np.linalg.norm(prev_pos - self.goal, axis=1)

        # Keep old crashed to detect newly_crashed in this step
        old_crashed = self.crashed.copy()
        old_finished = self.finished.copy()

        # Dynamics integration for active agents
        if np.any(active_mask):
            # update velocities (simple acceleration model)
            self.vel += a * self.dt

            # limit speed
            speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
            speed_clip = np.clip(speed, 0, self.max_speed)
            self.vel = self.vel * (speed_clip / (1e-9 + speed))

            # hard stop inactive
            self.vel[~active_mask] = 0.0

            # update positions
            new_pos = self.pos + self.vel * self.dt
        else:
            new_pos = self.pos.copy()

        # --- Crash check against obstacles (and optionally walls) ---
        if len(self.obstacles) > 0:
            for i in range(self.n):
                if not active_mask[i]:
                    continue
                dists = np.linalg.norm(self.obstacles - new_pos[i], axis=1)
                # collision if distance less than obstacle radius + small margin
                if np.any(dists < (self.obstacle_radius + 0.05)):
                    self.crashed[i] = True
                    self.vel[i] = 0.0
                    # do not update position into obstacle (simulate stopping short)
                    new_pos[i] = self.pos[i]

        # update positions (clipped to world)
        self.pos = np.clip(new_pos, 0.0, self.world_size)

        # Check for reaching goal
        dist_to_goal = np.linalg.norm(self.pos - self.goal, axis=1)
        newly_finished = (dist_to_goal < self.goal_threshold) & (~self.finished) & (~self.crashed)
        self.finished = self.finished | newly_finished

        # detect newly crashed
        newly_crashed = (~old_crashed) & self.crashed

        # update timestep
        self.t += 1
        done = (self.t >= self.max_steps) or np.all(self.finished | self.crashed)

        # compute rewards: pass in clipped `a`, prev distances, newly flags
        rewards = self._compute_rewards(a, prev_dist_to_goal, newly_finished, newly_crashed)

        obs = self._get_obs()

        # per-agent done flags for replay (useful for agent-specific bootstrapping)
        per_agent_done = (self.finished | self.crashed).astype(np.float32)

        return obs, rewards, done, {'per_agent_done': per_agent_done}

    def _get_obs(self):
        # Build per-agent observation and normalize positions/velocities
        obs = []
        for i in range(self.n):
            # own position (relative to world, normalized), own vel (normalized)
            own_pos = (self.pos[i] / self.world_size).astype(np.float32)
            own_vel = (self.vel[i] / self.max_speed).astype(np.float32)
            rel_goal = ((self.goal - self.pos[i]) / self.world_size).astype(np.float32)

            # neighbors: sorted by real-world dist
            dists = np.linalg.norm(self.pos - self.pos[i], axis=1)
            idx = np.argsort(dists)
            valid_neighbors = idx[1:1 + self.neighbor_obs]
            neighbors = []
            for k in valid_neighbors:
                relp = (self.pos[k] - self.pos[i]) / self.world_size
                relv = (self.vel[k] - self.vel[i]) / self.max_speed
                neighbors.append(np.concatenate([relp, relv]).astype(np.float32))
            while len(neighbors) < self.neighbor_obs:
                neighbors.append(np.zeros(4, dtype=np.float32))

            # obstacles: nearest N, positions relative to agent and normalized
            if len(self.obstacles) > 0:
                obs_dists = np.linalg.norm(self.obstacles - self.pos[i], axis=1)
                obs_idx = np.argsort(obs_dists)
                obs_feats = []
                for k in obs_idx[:self.obstacle_obs]:
                    rel_obs = (self.obstacles[k] - self.pos[i]) / self.world_size
                    obs_feats.append(rel_obs.astype(np.float32))
                while len(obs_feats) < self.obstacle_obs:
                    obs_feats.append(np.zeros(2, dtype=np.float32))
                flat_obs = np.concatenate(obs_feats)
            else:
                flat_obs = np.zeros(self.obstacle_obs * 2, dtype=np.float32)

            neighbors_flat = np.concatenate(neighbors)
            obs_i = np.concatenate([own_pos, own_vel, rel_goal, neighbors_flat, flat_obs])
            obs.append(obs_i.astype(np.float32))

        return np.stack(obs, axis=0)

    def _compute_rewards(self, a, prev_dist_to_goal, newly_finished, newly_crashed):
        """
        rewards:
          - delta distance shaping: +C * (prev_dist - new_dist)  (dense)
          - one-time big positive for newly_finished
          - one-time big negative for newly_crashed
          - small penalty per step proportional to distance (encourages move)
          - inter-agent collision penalty
          - control cost
        """
        rewards = np.zeros(self.n, dtype=np.float32)

        # Distance-based shaping (delta)
        dist_to_goal = np.linalg.norm(self.pos - self.goal, axis=1)
        delta = prev_dist_to_goal - dist_to_goal  # positive if agent moved closer
        rewards += (delta * 20.0)  # scale; tune as needed

        # Small per-step penalty proportional to distance (encourages being closer)
        rewards -= dist_to_goal * 0.02

        # One-time success / crash rewards (strong)
        rewards[newly_finished] += 50.0
        rewards[newly_crashed] += -80.0

        # If currently crashed, give a fixed negative (discourage being dead)
        rewards[self.crashed] += -1.0  # small continuing penalty if desired

        # Agent-agent collisions (one-time - scaled)
        min_dist = 0.25 * (self.world_size / 10.0)  # scaled by world size (here world_size=10)
        for i in range(self.n):
            for j in range(i+1, self.n):
                if np.linalg.norm(self.pos[i] - self.pos[j]) < min_dist:
                    rewards[i] -= 2.0
                    rewards[j] -= 2.0

        # Control cost (small)
        ctrl_cost = -0.01 * np.sum(a**2, axis=1)
        rewards += ctrl_cost

        return rewards.astype(np.float32)

# ---------------------------------
# Neural Network Code (Same as before)
# ---------------------------------
Transition = namedtuple('Transition', ('obs', 'actions', 'rewards', 'next_obs', 'dones'))
class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args): self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return Transition(*zip(*samples))
    def __len__(self): return len(self.buffer)

def mlp(input_dim, hidden_dims=[64,64], output_dim=None, activation=nn.ReLU):
    layers = []
    d = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h)); layers.append(activation()); d = h
    if output_dim is not None: layers.append(nn.Linear(d, output_dim))
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=[64,64]):
        super().__init__()
        self.net = mlp(obs_dim, hidden, output_dim=action_dim)
    def forward(self, x): return torch.tanh(self.net(x))

class AttentionCritic(nn.Module):
    def __init__(self, full_obs_dim, full_action_dim, hidden=64, K=4, head_dim=32):
        super().__init__()
        self.K = K
        self.encoder = mlp(full_obs_dim + full_action_dim, [hidden], output_dim=hidden)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden, head_dim), nn.ReLU()) for _ in range(K)])
        self.h_net = nn.Sequential(nn.Linear(full_action_dim, hidden), nn.ReLU(), nn.Linear(hidden, head_dim))
        self.final = nn.Linear(head_dim, 1)
    def forward(self, full_obs, full_actions):
        x = torch.cat([full_obs, full_actions], dim=-1)
        enc = self.encoder(x)
        Qks = torch.stack([h(enc) for h in self.heads], dim=1)
        hi = self.h_net(full_actions).unsqueeze(1)
        scores = torch.sum(hi * Qks, dim=-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        contextual = torch.sum(weights * Qks, dim=1)
        return self.final(contextual), Qks, weights

class Agent:
    def __init__(self, obs_dim, action_dim, lr_actor=1e-3):
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)

class AttMADDPG:
    def __init__(self, n_agents, obs_dim, action_dim, neighbor_obs, K=4, critic_lr=1e-2, gamma=0.95, tau=0.01):
        self.n = n_agents
        self.gamma, self.tau = gamma, tau
        self.agents = [Agent(obs_dim, action_dim) for _ in range(n_agents)]
        full_obs_dim = obs_dim * n_agents
        full_action_dim = action_dim * n_agents
        self.critics = [AttentionCritic(full_obs_dim, full_action_dim, K=K).to(device) for _ in range(n_agents)]
        self.target_critics = [copy.deepcopy(c).to(device) for c in self.critics]
        self.critic_opts = [optim.Adam(c.parameters(), lr=critic_lr) for c in self.critics]

    def select_actions(self, obs, noise_scale=0.1):
        actions = []
        for i, agent in enumerate(self.agents):
            o = torch.tensor(obs[i:i+1], dtype=torch.float32, device=device)
            with torch.no_grad(): a = agent.actor(o).cpu().numpy()[0]
            a = a + noise_scale * np.random.randn(*a.shape)
            actions.append(np.clip(a, -1.0, 1.0))
        return np.stack(actions, axis=0)

    def update(self, buffer: ReplayBuffer, batch_size=128):
        if len(buffer) < batch_size: return
        batch = buffer.sample(batch_size)
        obs = np.stack(batch.obs)
        actions = np.stack(batch.actions)
        rewards = np.stack(batch.rewards)
        next_obs = np.stack(batch.next_obs)
        dones = np.stack(batch.dones)
        B = obs.shape[0]
        full_obs = torch.tensor(obs.reshape(B, -1), dtype=torch.float32, device=device)
        full_actions = torch.tensor(actions.reshape(B, -1), dtype=torch.float32, device=device)
        full_next_obs = torch.tensor(next_obs.reshape(B, -1), dtype=torch.float32, device=device)
        
        next_actions_list = []
        for j in range(self.n):
            o_next = torch.tensor(next_obs[:, j, :], dtype=torch.float32, device=device)
            with torch.no_grad(): a_next = self.agents[j].target_actor(o_next)
            next_actions_list.append(a_next)
        next_actions_tensor = torch.stack(next_actions_list, dim=1).view(B, -1)

        for i in range(self.n):
            critic, target_critic, critic_opt = self.critics[i], self.target_critics[i], self.critic_opts[i]
            with torch.no_grad():
                tq, _, _ = target_critic(full_next_obs, next_actions_tensor)
                r = torch.tensor(rewards[:, i:i+1], dtype=torch.float32, device=device)
                done = torch.tensor(dones[:, i:i+1], dtype=torch.float32, device=device)
                y = r + self.gamma * (1.0 - done) * tq
            q, _, _ = critic(full_obs, full_actions)
            loss_q = ((q - y)**2).mean()
            critic_opt.zero_grad(); loss_q.backward(); critic_opt.step()

            curr_actions_list = []
            for j in range(self.n):
                o = torch.tensor(obs[:, j, :], dtype=torch.float32, device=device)
                if j == i: a_j = self.agents[j].actor(o)
                else: 
                    with torch.no_grad(): a_j = self.agents[j].actor(o)
                curr_actions_list.append(a_j)
            curr_actions_tensor = torch.stack(curr_actions_list, dim=1).view(B, -1)
            q_pi, _, _ = critic(full_obs, curr_actions_tensor)
            actor_loss = -q_pi.mean()
            self.agents[i].actor_opt.zero_grad(); actor_loss.backward(); self.agents[i].actor_opt.step()
            for p, tp in zip(critic.parameters(), target_critic.parameters()): tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.agents[i].actor.parameters(), self.agents[i].target_actor.parameters()): tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            
    def save_checkpoint(self, filename):
        state = {}
        for i, agent in enumerate(self.agents): state[f'agent_{i}_actor'] = agent.actor.state_dict()
        torch.save(state, filename)
    def load_checkpoint(self, filename):
        map_loc = 'cpu' if not torch.cuda.is_available() else None
        state = torch.load(filename, map_location=map_loc)
        for i, agent in enumerate(self.agents):
            if f'agent_{i}_actor' in state: agent.actor.load_state_dict(state[f'agent_{i}_actor'])

# ------------------------------
# Snapshot & Visualization
# ------------------------------
def save_environment_snapshot(env, filename="initial_layout.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, env.world_size)
    ax.set_ylim(0, env.world_size)
    ax.set_title("Training Environment Layout")
    
    goal_circle = plt.Circle(env.goal, env.goal_threshold, color='green', alpha=0.2, label='Goal Zone')
    ax.add_patch(goal_circle)
    ax.scatter(env.goal[0], env.goal[1], marker='x', color='green', s=100)

    for ob_pos in env.obstacles:
        ob_patch = plt.Circle(ob_pos, env.obstacle_radius, color='red', alpha=0.5)
        ax.add_patch(ob_patch)

    # Plot initial agent positions (random start)
    ax.scatter(env.pos[:, 0], env.pos[:, 1], c='blue', s=80, edgecolors='k', label='Start Area')

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()
    print(f"--> Saved fixed environment layout to '{filename}'")

def visualize_model(filename, n_agents=3, neighbor_obs=2, n_obstacles=3):
    # Important: In Test mode, we normally regenerate obstacles.
    # To match training, we would need to load the map config, but here we just
    # let it generate a NEW map to test GENERALIZATION.
    # If you want to test on the EXACT same map, you'd need to save/load obstacle coords.
    
    env = FlockingEnv(n_agents=n_agents, n_obstacles=n_obstacles, neighbor_obs=neighbor_obs)
    obs = env.reset()
    
    obs_dim = obs.shape[1]
    action_dim = 2
    trainer = AttMADDPG(n_agents, obs_dim, action_dim, neighbor_obs)
    try:
        trainer.load_checkpoint(filename)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return

    pos_history = []
    status_history = [] # 0=Active, 1=Finished, 2=Crashed
    
    def get_status():
        s = np.zeros(n_agents)
        s[env.finished] = 1
        s[env.crashed] = 2
        return s

    pos_history.append(env.pos.copy())
    status_history.append(get_status())
    
    print("Running simulation...")
    for t in range(env.max_steps):
        actions = trainer.select_actions(obs, noise_scale=0.0)
        next_obs, rewards, done, _ = env.step(actions)
        pos_history.append(env.pos.copy())
        status_history.append(get_status())
        obs = next_obs
        if done: break
            
    print(f"Generating animation ({len(pos_history)} steps)...")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, env.world_size)
    ax.set_ylim(0, env.world_size)
    ax.set_title(f"Flocking with Crash Logic")
    
    goal_circle = plt.Circle(env.goal, env.goal_threshold, color='green', alpha=0.2, label='Goal Zone')
    ax.add_patch(goal_circle)
    ax.scatter(env.goal[0], env.goal[1], marker='x', color='green', s=100)

    for ob_pos in env.obstacles:
        ob_patch = plt.Circle(ob_pos, env.obstacle_radius, color='red', alpha=0.5)
        ax.add_patch(ob_patch)
    
    colors = plt.cm.jet(np.linspace(0, 1, n_agents))
    initial_pos = pos_history[0]
    scat = ax.scatter(initial_pos[:, 0], initial_pos[:, 1], c=colors, s=80, edgecolors='k')
    
    def update(frame):
        current_pos = pos_history[frame]
        status = status_history[frame]
        
        current_colors = []
        for i in range(n_agents):
            if status[i] == 1: # Finished
                current_colors.append((0, 1, 0, 1)) # Green
            elif status[i] == 2: # Crashed
                current_colors.append((0, 0, 0, 1)) # Black
            else:
                current_colors.append(colors[i])
        
        scat.set_offsets(current_pos)
        scat.set_color(current_colors)
        return [scat]

    ani = FuncAnimation(fig, update, frames=len(pos_history), interval=50, blit=True)
    output_file = "flocking_crash_test.gif"
    print(f"Saving to {output_file}...")
    ani.save(output_file, writer='pillow', fps=20)
    print("Done.")

# ------------------------------
# Training Main
# ------------------------------
def train_example(save_path="att_maddpg_obstacles.pth", mapfile="fixed_map.npz"):
    n_agents = 3
    n_obstacles = 3
    neighbor_obs = 2

    # Init Env (Generates fixed obstacles)
    env = FlockingEnv(n_agents=n_agents, n_obstacles=n_obstacles, neighbor_obs=neighbor_obs,
                      max_speed=1.0, seed=None, save_map_path=mapfile)

    # Save the FIXED map image
    save_environment_snapshot(env, "initial_layout.png")

    obs0 = env.reset()
    obs_dim = obs0.shape[1]
    action_dim = 2
    print(f"State Dim: {obs_dim} (includes normalized obstacle info)")

    trainer = AttMADDPG(n_agents, obs_dim, action_dim, neighbor_obs, K=4, critic_lr=1e-3)
    buffer = ReplayBuffer(100000)

    episodes = 4000
    batch_size = 128
    warmup_steps = 5000  # don't start learning until some experience collected
    total_steps = 0

    # Training stats
    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0.0
        ep_success = 0
        ep_crash = 0

        for t in range(env.max_steps):
            noise = max(0.05, 0.5 * (1 - ep / episodes))
            actions = trainer.select_actions(obs, noise_scale=noise)
            next_obs, rewards, done, info = env.step(actions)

            # Use per-agent done flags (more correct than repeating episode-done)
            per_agent_done = info.get('per_agent_done', (env.finished | env.crashed).astype(np.float32))

            # store transition: obs shape (n_agents, obs_dim), actions (n_agents, action_dim), rewards (n_agents,)
            buffer.push(obs.copy(), actions.copy(), rewards.copy(), next_obs.copy(), per_agent_done.copy())

            total_steps += 1
            if len(buffer) > batch_size and total_steps > warmup_steps:
                trainer.update(buffer, batch_size=batch_size)

            obs = next_obs
            ep_reward += np.sum(rewards)
            ep_success += np.sum(env.finished & ~old_false_mask()) if False else 0  # placeholder; we log after episode
            if done:
                break

        # After episode: compute final statistics
        ep_finished = np.sum(env.finished)
        ep_crashed = np.sum(env.crashed)
        if (ep + 1) % 10 == 0:
            print(f"Ep {ep+1}/{episodes} | Reward {ep_reward:.1f} | Finished {ep_finished}/{n_agents} | Crashed {ep_crashed}/{n_agents} | Buffer {len(buffer)}")

    trainer.save_checkpoint(save_path)

if __name__ == "__main__":
    MODE = "train" 
    MODEL_FILE = "att_maddpg_obstacles.pth"
    
    if MODE == "train":
        train_example(save_path=MODEL_FILE)
        visualize_model(MODEL_FILE, n_agents=3, n_obstacles=3)
        
    elif MODE == "test":
        visualize_model(MODEL_FILE, n_agents=3, n_obstacles=3)
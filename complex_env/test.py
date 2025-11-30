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
                 neighbor_obs=2, obstacle_obs=2, max_speed=1.0, seed=None, save_map_path=None, agent_radius=0.15):
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
        self.save_map_path = save_map_path
        self.agent_radius = agent_radius
        self.flocking_weight = 1.0   # starts with full flocking emphasis


        # Generate fixed map once
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

        self.reset()

    def load_map(self, mapfile):
        d = np.load(mapfile)
        self.goal = d['goal']
        self.obstacles = d['obstacles']

    def reset(self):
        """
        Initialize agents in a formation around a start center such that:
        - inter-agent spacing >= desired_spacing (if possible)
        - uses circular formation for natural flocking spacing
        - falls back to a centered grid if the circular radius would exceed start area
        The agents get a small random velocity perturbation (not colliding).
        """
        # Start zone center and available radius for placing agents
        start_center = np.array([self.world_size * 0.15, self.world_size * 0.15], dtype=np.float32)
        max_start_radius = self.world_size * 0.08  # how far from center agents can be placed

        # Desired safe spacing between agents (world units). Tune to your desired formation.
        # Using same 'desired' from flocking rewards is reasonable.
        desired_spacing = 0.8

        n = self.n
        positions = []

        if n == 1:
            positions = [start_center.copy()]
        else:
            # try a single-ring circular placement with arc spacing >= desired_spacing
            # arc chord spacing s between neighbors on a circle of radius r: s = 2 * r * sin(pi/n)
            # solve for r: r = s / (2 * sin(pi/n))
            denom = 2.0 * np.sin(np.pi / max(1, n))
            # numerically safe
            if denom <= 1e-6:
                circ_r = max_start_radius
            else:
                circ_r = desired_spacing / denom

            if circ_r <= max_start_radius:
                # place on a circle of radius circ_r
                angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
                for theta in angles:
                    p = start_center + np.array([circ_r * np.cos(theta), circ_r * np.sin(theta)], dtype=np.float32)
                    positions.append(p)
            else:
                # fallback: centered grid with spacing = desired_spacing
                grid_spacing = desired_spacing
                grid_side = int(np.ceil(np.sqrt(n)))
                # Build grid centered at start_center
                half_span = (grid_side - 1) * grid_spacing / 2.0
                xs = np.linspace(start_center[0] - half_span, start_center[0] + half_span, grid_side)
                ys = np.linspace(start_center[1] - half_span, start_center[1] + half_span, grid_side)
                for xi in xs:
                    for yi in ys:
                        if len(positions) < n:
                            positions.append(np.array([xi, yi], dtype=np.float32))
                # If any position out of bounds due to edges, clip them into world but try keep spacing
                for i in range(len(positions)):
                    positions[i] = np.clip(positions[i], 0.0 + self.agent_radius, self.world_size - self.agent_radius)

        # convert to numpy array and assign
        self.pos = np.vstack(positions).astype(np.float32)

        # Safety: if any accidental overlap remains (numerical), apply tiny jitter outward
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.pos[i] - self.pos[j])
                min_allowed = 2.0 * getattr(self, 'agent_radius', 0.15) + 1e-3
                if d < min_allowed:
                    # push j slightly along outward radial direction relative to start center
                    dir_vec = self.pos[j] - start_center
                    if np.linalg.norm(dir_vec) < 1e-6:
                        dir_vec = np.random.randn(2)
                    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-9)
                    self.pos[j] += dir_vec * (min_allowed - d + 1e-3)

        # small random initial velocities (low magnitude) for variety but not causing immediate collisions
        self.vel = (np.random.rand(self.n, 2) - 0.5) * (self.max_speed * 0.05)

        # reset flags and bookkeeping
        self.finished = np.zeros(self.n, dtype=bool)
        self.crashed = np.zeros(self.n, dtype=bool)
        self.prev_dist_to_goal = np.linalg.norm(self.pos - self.goal, axis=1)
        self.t = 0

        return self._get_obs()


    def step(self, actions):
        # ALWAYS clip actions
        a = np.clip(actions, -1.0, 1.0)

        active_mask = ~(self.finished | self.crashed)
        a = a.copy()
        a[~active_mask] = 0.0

        prev_pos = self.pos.copy()
        prev_dist_to_goal = np.linalg.norm(prev_pos - self.goal, axis=1)
        old_crashed = self.crashed.copy()
        old_finished = self.finished.copy()

        # Dynamics
        if np.any(active_mask):
            self.vel += a * self.dt
            speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
            speed_clip = np.clip(speed, 0, self.max_speed)
            self.vel = self.vel * (speed_clip / (1e-9 + speed))
            self.vel[~active_mask] = 0.0
            new_pos = self.pos + self.vel * self.dt
        else:
            new_pos = self.pos.copy()

        # Crash check with fixed obstacles
        if len(self.obstacles) > 0:
            for i in range(self.n):
                if not active_mask[i]:
                    continue
                dists = np.linalg.norm(self.obstacles - new_pos[i], axis=1)
                if np.any(dists < (self.obstacle_radius + 0.05)):
                    self.crashed[i] = True
                    self.vel[i] = 0.0
                    new_pos[i] = self.pos[i]  # stop before entering obstacle

        # --- NEW: Agent-Agent collision detection (treat other agents like obstacles) ---
        # Compute pairwise distances on the proposed new positions
        agent_collision_dist = 2.0 * self.agent_radius  # or tune separately
        # We will detect collisions and mark both agents as crashed and revert them
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Skip already crashed or already finished agents (they shouldn't move)
                if (self.crashed[i] or self.crashed[j] or self.finished[i] or self.finished[j]):
                    continue
                d = np.linalg.norm(new_pos[i] - new_pos[j])
                if d < agent_collision_dist:
                    # Mark both as crashed
                    self.crashed[i] = True
                    self.crashed[j] = True
                    # Stop velocities and revert positions so they appear to stop before collision
                    self.vel[i] = 0.0
                    self.vel[j] = 0.0
                    new_pos[i] = self.pos[i]
                    new_pos[j] = self.pos[j]

        # update positions clipped to world
        self.pos = np.clip(new_pos, 0.0, self.world_size)

        # Check for reaching goal
        dist_to_goal = np.linalg.norm(self.pos - self.goal, axis=1)
        newly_finished = (dist_to_goal < self.goal_threshold) & (~self.finished) & (~self.crashed)
        self.finished = self.finished | newly_finished

        # detect newly crashed
        newly_crashed = (~old_crashed) & self.crashed

        self.t += 1
        done = (self.t >= self.max_steps) or np.all(self.finished | self.crashed)

        # Compute rewards (pass prev distances and newly flags)
        rewards = self._compute_rewards(a, prev_dist_to_goal, newly_finished, newly_crashed)

        obs = self._get_obs()
        per_agent_done = (self.finished | self.crashed).astype(np.float32)

        info = {
            'per_agent_done': per_agent_done,
            'newly_finished': newly_finished.astype(np.float32),
            'newly_crashed': newly_crashed.astype(np.float32)
        }
        return obs, rewards, done, info


    def _get_obs(self):
        obs = []
        for i in range(self.n):
            own_pos = (self.pos[i] / self.world_size).astype(np.float32)
            own_vel = (self.vel[i] / self.max_speed).astype(np.float32)
            rel_goal = ((self.goal - self.pos[i]) / self.world_size).astype(np.float32)

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
        Goal-weighted reward: stronger delta-distance + success.
        Flocking reward reduced; proximity penalties kept to avoid collisions.
        """
        rewards = np.zeros(self.n, dtype=np.float32)

        # -------- parameters (tune these) --------
        delta_scale = 30.0                # increase goal-directed shaping
        per_step_dist_pen = 0.02
        success_reward = 80.0             # larger finishing bonus
        newly_crashed_penalty = -100.0
        crashed_ongoing_pen = -0.5

        # flocking/cohesion parameters (reduced)
        flocking_good_reward = 0.4        # smaller positive reward for staying in band
        flocking_close_penalty = -2.0     # penalty if too close
        flocking_far_penalty_scale = 0.2  # small penalty for being too far

        # proximity (pre-crash) penalty scale
        prox_multiplier = 2.5             # scale for normalized proximity penalty

        # -------- 1) Dense progress shaping (stronger) --------
        dist_to_goal = np.linalg.norm(self.pos - self.goal, axis=1)
        delta = prev_dist_to_goal - dist_to_goal
        rewards += (delta * delta_scale)
        rewards -= dist_to_goal * per_step_dist_pen

        # -------- 2) Success & crash one-time --------
        rewards[np.asarray(newly_finished, dtype=bool)] += success_reward
        rewards[np.asarray(newly_crashed, dtype=bool)] += newly_crashed_penalty
        rewards[self.crashed] += crashed_ongoing_pen

        # -------- 3) Pairwise proximity (pre-crash) --------
        n = self.n
        pos = self.pos
        pdist = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                pdist[i, j] = d
                pdist[j, i] = d

        agent_collision_dist = 2.0 * self.agent_radius
        safe_margin = agent_collision_dist * 1.6
        soft_margin = agent_collision_dist * 3.0

        for i in range(n):
            if self.crashed[i] or self.finished[i]:
                continue
            prox_pen = 0.0
            for j in range(n):
                if j == i: continue
                d = pdist[i, j]
                if d < safe_margin:
                    # normalized closeness: 1 when d==agent_collision_dist, 0 at safe_margin
                    denom = max(1e-6, safe_margin - agent_collision_dist)
                    prox_pen += max(0.0, (safe_margin - d) / denom)
                elif d > soft_margin:
                    # slight negative to encourage cohesion but small (kept gentle)
                    prox_pen -= 0.01 * (d - soft_margin)
            rewards[i] += -prox_multiplier * prox_pen

        # -------- 4) Reduced flocking reward (safety-aware) --------
        desired = 0.8
        tol = 0.45
        min_band = max(0.1, desired - tol)
        max_band = desired + tol

        for i in range(n):
            if self.crashed[i] or self.finished[i]:
                continue
            others = [j for j in range(n) if j != i]
            if not others:
                continue
            dists = np.array([pdist[i, j] for j in others])
            mean_dist = np.mean(dists)
            if np.any(dists < (agent_collision_dist * 1.05)):
                # strongly discourage being dangerously close
                rewards[i] += flocking_close_penalty * self.flocking_weight
            else:
                if (mean_dist >= min_band) and (mean_dist <= max_band):
                    rewards[i] += flocking_good_reward * self.flocking_weight
                else:
                    if mean_dist < min_band:
                        rewards[i] -= 1.0 * (min_band - mean_dist)
                    else:
                        rewards[i] -= flocking_far_penalty_scale * (mean_dist - max_band)

        # -------- 5) Hard collision penalty as redundancy --------
        min_dist_collision = agent_collision_dist * 0.95
        for i in range(n):
            for j in range(i+1, n):
                if pdist[i, j] < min_dist_collision:
                    rewards[i] -= 8.0
                    rewards[j] -= 8.0

        # -------- 6) Control cost --------
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

    # uncomment below line to test the algo on fixed obstacles on which training done
    env.load_map("fixed_map.npz")
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
    n_agents = 4
    n_obstacles = 5
    neighbor_obs = 2

    env = FlockingEnv(n_agents=n_agents, n_obstacles=n_obstacles, neighbor_obs=neighbor_obs,
                      max_speed=1.0, seed=None, save_map_path=mapfile)

    save_environment_snapshot(env, "initial_layout.png")

    obs0 = env.reset()
    obs_dim = obs0.shape[1]
    action_dim = 2
    print(f"State Dim: {obs_dim} (includes normalized obstacle info)")

    trainer = AttMADDPG(n_agents, obs_dim, action_dim, neighbor_obs, K=4, critic_lr=1e-3)
    buffer = ReplayBuffer(200000)

    episodes = 4000
    batch_size = 128
    warmup_steps = 2000
    total_steps = 0

    # Logging lists for plotting
    episode_rewards = []
    episode_mean_agent_rewards = []
    episode_finished_counts = []
    episode_crashed_counts = []
    episode_avg_dist_to_goal = []
    episode_lengths = []

    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0.0
        ep_agent_rewards = np.zeros(n_agents, dtype=np.float32)
        ep_newly_finished = 0
        ep_newly_crashed = 0

        for t in range(env.max_steps):
            noise = max(0.05, 0.5 * (1 - ep / episodes))
            actions = trainer.select_actions(obs, noise_scale=noise)

            next_obs, rewards, done, info = env.step(actions)
            per_agent_done = info.get('per_agent_done', (env.finished | env.crashed).astype(np.float32))
            newly_finished = info.get('newly_finished', np.zeros(n_agents, dtype=np.float32))
            newly_crashed = info.get('newly_crashed', np.zeros(n_agents, dtype=np.float32))

            # push transition using per-agent dones
            buffer.push(obs.copy(), actions.copy(), rewards.copy(), next_obs.copy(), per_agent_done.copy())

            total_steps += 1
            # update after warmup
            if len(buffer) > batch_size and total_steps > warmup_steps:
                trainer.update(buffer, batch_size=batch_size)

            obs = next_obs
            ep_reward += np.sum(rewards)
            ep_agent_rewards += rewards
            ep_newly_finished += int(np.sum(newly_finished))
            ep_newly_crashed += int(np.sum(newly_crashed))

            if done:
                episode_lengths.append(t + 1)
                break
        else:
            # if loop not broken by done, append full length
            episode_lengths.append(env.max_steps)

        ep_finished = int(np.sum(env.finished))
        ep_crashed = int(np.sum(env.crashed))
        mean_agent_reward = float(np.mean(ep_agent_rewards))
        avg_dist_goal = float(np.mean(np.linalg.norm(env.pos - env.goal, axis=1)))

        episode_rewards.append(ep_reward)
        episode_mean_agent_rewards.append(mean_agent_reward)
        episode_finished_counts.append(ep_finished)
        episode_crashed_counts.append(ep_crashed)
        episode_avg_dist_to_goal.append(avg_dist_goal)
        # Decay flocking emphasis as episodes progress
        env.flocking_weight = max(0.1, env.flocking_weight * 0.995)


        # Print more informative logging every episode or every N episodes
        if (ep + 1) % 10 == 0:  # print every ep (change to 10 if you want less verbose)
            print(f"Ep {ep+1}/{episodes} | TotalR {ep_reward:8.2f} | MeanAgentR {mean_agent_reward:6.3f} | "
                  f"Finished {ep_finished}/{n_agents} | Crashed {ep_crashed}/{n_agents} | "
                  f"NewFin {ep_newly_finished} | NewCrash {ep_newly_crashed} | AvgDist {avg_dist_goal:5.2f} | Len {episode_lengths[-1]} | Buf {len(buffer)}")

        # periodic checkpointing + quick eval
        if (ep + 1) % 50 == 0:
            trainer.save_checkpoint(save_path)
            print(f"Saved checkpoint at episode {ep+1}")

    # final save
    trainer.save_checkpoint(save_path)
    print("Training finished. Saved final model.")

    # Plot and save reward curve and some stats
    plt.figure(figsize=(8, 4))
    plt.plot(episode_rewards, label='episode total reward')
    plt.plot(np.convolve(episode_rewards, np.ones(50)/50, mode='same'), label='50-ep MA')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_rewards.png')
    plt.close()
    print("Saved training reward curve to training_rewards.png")

    # Save a small CSV-like text summary for quick inspection
    import csv
    with open('training_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward', 'mean_agent_reward', 'finished', 'crashed', 'avg_dist', 'length'])
        for i in range(len(episode_rewards)):
            writer.writerow([i+1, episode_rewards[i], episode_mean_agent_rewards[i],
                             episode_finished_counts[i], episode_crashed_counts[i],
                             episode_avg_dist_to_goal[i], episode_lengths[i]])
    print("Saved training_summary.csv")

if __name__ == "__main__":
    MODE = "train" 
    MODEL_FILE = "att_maddpg_obstacles.pth"
    
    if MODE == "train":
        train_example(save_path=MODEL_FILE)
        visualize_model(MODEL_FILE, n_agents=4, n_obstacles=5)
        
    elif MODE == "test":
        visualize_model(MODEL_FILE, n_agents=4, n_obstacles=5, neighbor_obs=2)
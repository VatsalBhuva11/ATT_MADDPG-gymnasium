import numpy as np
import random
import copy
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Simple 2D flocking environment
# ------------------------------
class FlockingEnv:
    def __init__(self, n_agents=3, world_size=10.0, dt=0.1, max_steps=200, neighbor_obs=2):
        self.n = n_agents
        self.world_size = world_size
        self.dt = dt
        self.max_steps = max_steps
        self.neighbor_obs = neighbor_obs  # k nearest neighbors observed
        self.reset()

    def reset(self):
        # positions uniformly in a corner, velocities small random
        self.pos = np.random.rand(self.n, 2) * (self.world_size * 0.2) + 0.0
        self.vel = (np.random.rand(self.n, 2) - 0.5) * 0.1
        # fixed goal (center-right)
        self.goal = np.array([self.world_size * 0.8, self.world_size * 0.5], dtype=np.float32)
        self.t = 0
        obs = self._get_obs()
        return obs

    def step(self, actions):
        # actions: (n,2) acceleration
        a = np.clip(actions, -1.0, 1.0)
        self.vel += a * self.dt
        
        # limit speed
        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        max_speed = 1.0
        # Fix: Add epsilon to avoid divide by zero
        speed_clip = np.clip(speed, 0, max_speed)
        self.vel = self.vel * (speed_clip / (1e-6 + speed))
        
        self.pos += self.vel * self.dt
        
        # bound positions
        self.pos = np.clip(self.pos, 0.0, self.world_size)
        self.t += 1
        
        obs = self._get_obs()
        rewards = self._compute_rewards(a)
        done = (self.t >= self.max_steps)
        info = {}
        return obs, rewards, done, info

    def _get_obs(self):
        # For each agent, return: [px,py,vx,vy, gx,gy ] + relative pos/vel of k nearest neighbors
        obs = []
        for i in range(self.n):
            own = np.concatenate([self.pos[i], self.vel[i], (self.goal - self.pos[i])])
            
            # find k nearest neighbors
            dists = np.linalg.norm(self.pos - self.pos[i], axis=1)
            idx = np.argsort(dists)
            
            neighbors = []
            # idx[0] is self (dist 0), so start from 1
            # Handle case where n_agents < neighbor_obs
            valid_neighbors = idx[1:1 + self.neighbor_obs]
            
            for k in valid_neighbors:
                relp = self.pos[k] - self.pos[i]
                relv = self.vel[k] - self.vel[i]
                neighbors.append(np.concatenate([relp, relv]))
            
            # pad if fewer neighbors (e.g. if n_agents < neighbor_obs + 1)
            while len(neighbors) < self.neighbor_obs:
                neighbors.append(np.zeros(4, dtype=np.float32))
                
            obs_i = np.concatenate([own] + neighbors)
            obs.append(obs_i.astype(np.float32))
            
        return np.stack(obs, axis=0)

    def _compute_rewards(self, a):
        # Shared team reward
        dist_to_goal = np.linalg.norm(self.pos - self.goal, axis=1)
        reward_goal = -np.mean(dist_to_goal)
        
        # cohesion: negative mean distance between neighbors
        pairwise = []
        for i in range(self.n):
            d = np.linalg.norm(self.pos - self.pos[i], axis=1)
            # exclude self (distance 0)
            valid_d = d[d > 1e-5] 
            if len(valid_d) > 0:
                pairwise.append(np.mean(valid_d))
            else:
                pairwise.append(0.0)
                
        cohesion = -np.mean(pairwise) if len(pairwise) > 0 else 0.0
        
        # separation penalty (hard collisions)
        min_dist = 0.2
        collisions = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                if np.linalg.norm(self.pos[i]-self.pos[j]) < min_dist:
                    collisions += 1
                    
        collision_pen = - collisions * 1.0
        
        # control cost
        ctrl_cost = -0.01 * np.mean(np.sum(a**2, axis=1))
        
        team_reward = reward_goal + 0.5*cohesion + collision_pen + ctrl_cost
        
        return np.array([team_reward]*self.n, dtype=np.float32)

# ---------------------------------
# Replay buffer
# ---------------------------------
Transition = namedtuple('Transition', ('obs', 'actions', 'rewards', 'next_obs', 'dones'))

class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*samples))
        return batch
    
    def __len__(self):
        return len(self.buffer)

# ------------------------------
# Networks
# ------------------------------
def mlp(input_dim, hidden_dims=[64,64], output_dim=None, activation=nn.ReLU):
    layers = []
    d = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        layers.append(activation())
        d = h
    if output_dim is not None:
        layers.append(nn.Linear(d, output_dim))
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=[64,64]):
        super().__init__()
        self.net = mlp(obs_dim, hidden, output_dim=action_dim)
    
    def forward(self, x):
        return torch.tanh(self.net(x))

class AttentionCritic(nn.Module):
    def __init__(self, full_obs_dim, full_action_dim, hidden=64, K=4, head_dim=32):
        super().__init__()
        self.K = K
        self.head_dim = head_dim
        
        # Encoder: takes joint state + joint action
        self.encoder = mlp(full_obs_dim + full_action_dim, [hidden], output_dim=hidden)
        
        # K-heads that output head_dim vectors each
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, head_dim), nn.ReLU()) 
            for _ in range(K)
        ])
        
        # hi generator (from all teammates' actions only)
        self.h_net = nn.Sequential(
            nn.Linear(full_action_dim, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden, head_dim)
        )
        
        # final linear to scalar from contextual vector
        self.final = nn.Linear(head_dim, 1)

    def forward(self, full_obs, full_actions):
        """
        full_obs: (batch, full_obs_dim)
        full_actions: (batch, full_action_dim)
        """
        # Concatenate full obs and full actions
        x = torch.cat([full_obs, full_actions], dim=-1)
        enc = self.encoder(x)  # (b, hidden)
        
        # compute K heads
        Qks = torch.stack([h(enc) for h in self.heads], dim=1)  # (b, K, head_dim)
        
        # compute hi from actions
        hi = self.h_net(full_actions).unsqueeze(1)  # (b, 1, head_dim)
        
        # attention scores dot(hi, Qk)
        # (b, 1, head_dim) * (b, K, head_dim) -> (b, K, head_dim) -> sum last dim -> (b, K)
        scores = torch.sum(hi * Qks, dim=-1)  
        
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (b, K, 1)
        
        # Weighted sum of Qks
        contextual = torch.sum(weights * Qks, dim=1)  # (b, head_dim)
        
        q = self.final(contextual)  # (b, 1)
        return q, Qks, weights

# ------------------------------
# Agent wrapper
# ------------------------------
class Agent:
    def __init__(self, obs_dim, action_dim, lr_actor=1e-3):
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)

# ------------------------------
# ATT-MADDPG trainer
# ------------------------------
class AttMADDPG:
    def __init__(self, n_agents, obs_dim, action_dim, neighbor_obs, K=4, critic_lr=1e-2, gamma=0.95, tau=0.01):
        self.n = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.agents = [Agent(obs_dim, action_dim) for _ in range(n_agents)]

        # Centralized critics inputs
        full_obs_dim = obs_dim * n_agents
        full_action_dim = action_dim * n_agents
        
        self.critics = [AttentionCritic(full_obs_dim, full_action_dim, K=K).to(device) for _ in range(n_agents)]
        self.target_critics = [copy.deepcopy(c).to(device) for c in self.critics]
        self.critic_opts = [optim.Adam(c.parameters(), lr=critic_lr) for c in self.critics]

    def select_actions(self, obs, noise_scale=0.1):
        actions = []
        for i, agent in enumerate(self.agents):
            # Fix: Ensure input is float32 and on device
            o = torch.tensor(obs[i:i+1], dtype=torch.float32, device=device)
            with torch.no_grad():
                a = agent.actor(o).cpu().numpy()[0] # returns (action_dim,)
            
            # Add noise
            a = a + noise_scale * np.random.randn(*a.shape)
            actions.append(np.clip(a, -1.0, 1.0))
        return np.stack(actions, axis=0) # (n, action_dim)

    def update(self, buffer: ReplayBuffer, batch_size=128):
        if len(buffer) < batch_size:
            return

        batch = buffer.sample(batch_size)
        
        # Fix: Convert list of arrays to stacked numpy arrays
        obs = np.stack(batch.obs)               # (B, n, obs_dim)
        actions = np.stack(batch.actions)       # (B, n, action_dim)
        rewards = np.stack(batch.rewards)       # (B, n)
        next_obs = np.stack(batch.next_obs)     # (B, n, obs_dim)
        dones = np.stack(batch.dones)           # (B, n)

        B = obs.shape[0]
        
        # Flatten for critic
        full_obs_np = obs.reshape(B, -1)
        full_actions_np = actions.reshape(B, -1)
        full_next_obs_np = next_obs.reshape(B, -1)

        # Fix: Convert to tensors with correct dtype and device
        full_obs = torch.tensor(full_obs_np, dtype=torch.float32, device=device)
        full_actions = torch.tensor(full_actions_np, dtype=torch.float32, device=device)
        full_next_obs = torch.tensor(full_next_obs_np, dtype=torch.float32, device=device)
        
        # 1. Compute target actions (No Grad)
        next_actions_list = []
        for j in range(self.n):
            o_next = torch.tensor(next_obs[:, j, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                a_next = self.agents[j].target_actor(o_next)
            next_actions_list.append(a_next)
        
        # Stack and view as (B, full_action_dim)
        next_actions_tensor = torch.stack(next_actions_list, dim=1).view(B, -1)

        # 2. Update Critics and Actors per agent
        for i in range(self.n):
            critic = self.critics[i]
            target_critic = self.target_critics[i]
            critic_opt = self.critic_opts[i]
            
            # --- Critic Update ---
            with torch.no_grad():
                tq, _, _ = target_critic(full_next_obs, next_actions_tensor)
                
                # Reshape rewards/dones for broadcasting: (B, 1)
                r = torch.tensor(rewards[:, i:i+1], dtype=torch.float32, device=device)
                done = torch.tensor(dones[:, i:i+1], dtype=torch.float32, device=device)
                
                y = r + self.gamma * (1.0 - done) * tq
            
            # Current Q
            q, _, _ = critic(full_obs, full_actions)
            loss_q = ((q - y)**2).mean()
            
            critic_opt.zero_grad()
            loss_q.backward()
            critic_opt.step()

            # --- Actor Update ---
            # To update Actor i, we need the critic to differentiate w.r.t Agent i's action.
            # Other agents' actions should be treated as fixed (detach them).
            
            curr_actions_list = []
            for j in range(self.n):
                o = torch.tensor(obs[:, j, :], dtype=torch.float32, device=device)
                if j == i:
                    # Agent i: keep gradient
                    a_j = self.agents[j].actor(o)
                else:
                    # Others: detach
                    with torch.no_grad():
                        a_j = self.agents[j].actor(o)
                curr_actions_list.append(a_j)
            
            curr_actions_tensor = torch.stack(curr_actions_list, dim=1).view(B, -1)
            
            # Actor loss = -Q(obs, actions_with_new_actor_i)
            q_pi, _, _ = critic(full_obs, curr_actions_tensor)
            actor_loss = -q_pi.mean()
            
            self.agents[i].actor_opt.zero_grad()
            actor_loss.backward()
            self.agents[i].actor_opt.step()

            # --- Soft Updates ---
            for p, tp in zip(critic.parameters(), target_critic.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
                
            for p, tp in zip(self.agents[i].actor.parameters(), self.agents[i].target_actor.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)


    def save_checkpoint(self, filename):
        """Saves the actor state dictionaries of all agents."""
        state = {}
        for i, agent in enumerate(self.agents):
            state[f'agent_{i}_actor'] = agent.actor.state_dict()
        torch.save(state, filename)
        print(f"Models saved to {filename}")

    def load_checkpoint(self, filename):
        """Loads actor state dictionaries."""
        if not torch.cuda.is_available():
            state = torch.load(filename, map_location='cpu')
        else:
            state = torch.load(filename)
            
        for i, agent in enumerate(self.agents):
            if f'agent_{i}_actor' in state:
                agent.actor.load_state_dict(state[f'agent_{i}_actor'])
        print(f"Models loaded from {filename}")


def visualize_model(filename, n_agents=3, neighbor_obs=2):
    # 1. Setup Environment
    env = FlockingEnv(n_agents=n_agents, neighbor_obs=neighbor_obs)
    obs = env.reset()
    
    # 2. Setup Trainer & Load Weights
    obs_dim = obs.shape[1]
    action_dim = 2
    trainer = AttMADDPG(n_agents, obs_dim, action_dim, neighbor_obs)
    try:
        trainer.load_checkpoint(filename)
    except FileNotFoundError:
        print(f"Error: Could not find file {filename}.")
        return

    # 3. Run Simulation
    pos_history = []
    actions_history = []
    
    # Store initial position
    pos_history.append(env.pos.copy())
    
    for t in range(env.max_steps):
        actions = trainer.select_actions(obs, noise_scale=0.0)
        next_obs, rewards, done, _ = env.step(actions)
        pos_history.append(env.pos.copy())
        obs = next_obs
        if done: break
            
    # 4. Animation
    print(f"Generating animation for {len(pos_history)} steps...")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, env.world_size)
    ax.set_ylim(0, env.world_size)
    ax.set_title(f"ATT-MADDPG Flocking ({n_agents} Agents)")
    
    goal_circle = plt.Circle(env.goal, 0.5, color='green', alpha=0.3, label='Goal')
    ax.add_patch(goal_circle)
    
    colors = plt.cm.jet(np.linspace(0, 1, n_agents))
    
    # Initialize with starting positions
    initial_pos = pos_history[0]
    scat = ax.scatter(initial_pos[:, 0], initial_pos[:, 1], c=colors, s=100)
    
    trails = [ax.plot([], [], '-', alpha=0.3, color=c)[0] for c in colors]

    def update(frame):
        current_pos = pos_history[frame]
        scat.set_offsets(current_pos)
        
        tail_len = 20
        start = max(0, frame - tail_len)
        for i in range(n_agents):
            path = np.array([p[i] for p in pos_history[start:frame+1]])
            if len(path) > 0:
                trails[i].set_data(path[:, 0], path[:, 1])
        return [scat] + trails

    ani = FuncAnimation(fig, update, frames=len(pos_history), interval=50, blit=True)
    
    # --- CHANGED CODE BELOW ---
    output_file = "flocking_simulation.gif"
    print(f"Saving animation to {output_file}...")
    
    # 'pillow' writer creates a GIF and usually comes installed with matplotlib
    ani.save(output_file, writer='pillow', fps=20)
    print("Done! Check your folder for the GIF file.")
    
    # plt.show()  <-- Removed this
# ------------------------------
# Training script (main)
# ------------------------------
def train_example(save_path="att_maddpg_model_3_agents.pth"):
    n_agents = 6    
    neighbor_obs = 2
    env = FlockingEnv(n_agents=n_agents, neighbor_obs=neighbor_obs)
    
    obs0 = env.reset()
    obs_dim = obs0.shape[1]
    action_dim = 2

    print(f"Device: {device}")
    trainer = AttMADDPG(n_agents, obs_dim, action_dim, neighbor_obs, K=4, critic_lr=1e-3)
    buffer = ReplayBuffer(100000)
    
    episodes = 10000 # Increased slightly for better convergence before saving
    max_steps = env.max_steps
    batch_size = 64

    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0.0
        
        for t in range(max_steps):
            noise = max(0.05, 0.5 * (1 - ep / episodes))
            actions = trainer.select_actions(obs, noise_scale=noise)
            next_obs, rewards, done, _ = env.step(actions)
            
            buffer.push(
                obs.copy(), actions.copy(), rewards.copy(), 
                next_obs.copy(), np.array([float(done)]*n_agents, dtype=np.float32)
            )

            trainer.update(buffer, batch_size=batch_size)
            obs = next_obs
            ep_reward += rewards[0]
            if done: break
                
        if (ep+1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes}, Reward: {ep_reward:.3f}")

    print("Training finished.")
    trainer.save_checkpoint(save_path)

if __name__ == "__main__":
    # Choose mode: "train" or "test"
    MODE = "test" 
    MODEL_FILE = "./training_6_agents/att_maddpg_model.pth"

    if MODE == "train":
        train_example(save_path=MODEL_FILE)
        # Optional: auto-visualize after training
        visualize_model(MODEL_FILE, n_agents=3)
        
    elif MODE == "test":
        visualize_model(MODEL_FILE, n_agents=5)
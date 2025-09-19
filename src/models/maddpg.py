import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .attention import AttentionMADDPGEncoder

class Actor(nn.Module):
    """
    Actor network for MADDPG with attention mechanism.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=128, max_action=1.0):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # max_action is kept for compatibility but we will rescale to [0,1]
        self.max_action = max_action 
        
        self.attention_encoder = AttentionMADDPGEncoder(obs_dim, hidden_dim)
        
        self.actor_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, obs, mask=None):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        
        encoded_obs, attention_weights = self.attention_encoder(obs, mask)
        encoded_obs = encoded_obs.squeeze(1)
        
        # Tanh gives action in [-1, 1]
        action = self.actor_net(encoded_obs)
        
        # Rescale action to [0, 1] to match the environment's expected space
        action = (action + 1.0) / 2.0
        
        return action, attention_weights

class Critic(nn.Module):
    """
    Critic network for MADDPG with attention mechanism.
    """
    def __init__(self, obs_dim, total_action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.obs_dim = obs_dim
        self.total_action_dim = total_action_dim
        
        self.attention_encoder = AttentionMADDPGEncoder(obs_dim, hidden_dim)
        
        self.critic_net = nn.Sequential(
            nn.Linear(hidden_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs, action, mask=None):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            action = action.unsqueeze(0)
        
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            
        encoded_obs, attention_weights = self.attention_encoder(obs, mask)
        encoded_obs = encoded_obs.squeeze(1)
        
        x = torch.cat([encoded_obs, action], dim=1)
        q_value = self.critic_net(x)
        
        return q_value, attention_weights

class MADDPGAgent:
    """
    MADDPG agent with attention mechanism.
    """
    def __init__(self, obs_dim, action_dim, num_agents, total_action_dim, 
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.95, tau=0.01, 
                 hidden_dim=128, max_action=1.0):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.total_action_dim = total_action_dim
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(obs_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.critic = Critic(obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.target_actor = Actor(obs_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.target_critic = Critic(obs_dim, total_action_dim, hidden_dim).to(self.device)
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.noise_scale = 0.1 # Noise scale should be relative to action space size (1.0)
        
    def select_action(self, obs, add_noise=True, mask=None):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            action, attention_weights = self.actor(obs_tensor, mask)
            
            if add_noise:
                noise = torch.randn_like(action) * self.noise_scale
                # Clamp action to the [0, 1] range
                action = torch.clamp(action + noise, 0.0, 1.0)
            
            return action.squeeze(0).cpu().numpy(), attention_weights
    
    # ... (the rest of the MADDPGAgent class is the same as the previous fix)
    def update(self, batch, other_agents, agent_idx):
        obs, actions, rewards, next_obs, dones = batch
        
        obs = torch.FloatTensor(np.array(obs)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_obs = torch.FloatTensor(np.array(next_obs)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)

        obs = obs.transpose(0, 1)
        actions = actions.transpose(0, 1)
        rewards = rewards.transpose(0, 1)
        next_obs = next_obs.transpose(0, 1)
        dones = dones.transpose(0, 1)

        batch_size = obs.shape[0]

        with torch.no_grad():
            next_actions = []
            for i, agent in enumerate(other_agents):
                next_action, _ = agent.target_actor(next_obs[:, i, :])
                next_actions.append(next_action)
            
            next_joint_actions = torch.cat(next_actions, dim=1)
            target_q, _ = self.target_critic(next_obs[:, agent_idx, :], next_joint_actions)
            target_q = rewards[:, agent_idx].unsqueeze(1) + self.gamma * target_q * (~dones[:, agent_idx]).unsqueeze(1)

        joint_actions = actions.reshape(batch_size, -1)
        current_q, _ = self.critic(obs[:, agent_idx, :], joint_actions)
        
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        actor_actions = []
        for i, agent in enumerate(other_agents):
            if i == agent_idx:
                action, _ = self.actor(obs[:, i, :])
            else:
                action, _ = agent.actor(obs[:, i, :])
            actor_actions.append(action)

        actor_joint_actions = torch.cat(actor_actions, dim=1)
        actor_loss = -self.critic(obs[:, agent_idx, :], actor_joint_actions)[0].mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)
        
        return { 'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item() }
    
    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save(self, filepath):
        torch.save({ 'actor_state_dict': self.actor.state_dict(), 'critic_state_dict': self.critic.state_dict() }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
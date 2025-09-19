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
        self.max_action = max_action
        
        # Attention encoder for multi-agent observations
        self.attention_encoder = AttentionMADDPGEncoder(obs_dim, hidden_dim)
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, obs, mask=None):
        """
        Forward pass of actor network.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            mask: Optional mask tensor
        
        Returns:
            Action tensor of shape (batch_size, action_dim)
        """
        # Add batch dimension if needed
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Add agent dimension for attention
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch_size, 1, obs_dim)
        
        # Apply attention encoding
        encoded_obs, attention_weights = self.attention_encoder(obs, mask)
        
        # Remove agent dimension for actor network
        encoded_obs = encoded_obs.squeeze(1)  # (batch_size, hidden_dim)
        
        # Generate action
        action = self.actor(encoded_obs)
        action = self.max_action * action
        
        return action, attention_weights

class Critic(nn.Module):
    """
    Critic network for MADDPG with attention mechanism.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Attention encoder for multi-agent observations
        self.attention_encoder = AttentionMADDPGEncoder(obs_dim, hidden_dim)
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs, action, mask=None):
        """
        Forward pass of critic network.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            action: Action tensor of shape (batch_size, action_dim)
            mask: Optional mask tensor
        
        Returns:
            Q-value tensor of shape (batch_size, 1)
        """
        # Add batch dimension if needed
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            action = action.unsqueeze(0)
        
        # Add agent dimension for attention
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch_size, 1, obs_dim)
        
        # Apply attention encoding
        encoded_obs, attention_weights = self.attention_encoder(obs, mask)
        
        # Remove agent dimension
        encoded_obs = encoded_obs.squeeze(1)  # (batch_size, hidden_dim)
        
        # Concatenate observation and action
        x = torch.cat([encoded_obs, action], dim=-1)
        
        # Compute Q-value
        q_value = self.critic(x)
        
        return q_value, attention_weights

class MADDPGAgent:
    """
    MADDPG agent with attention mechanism.
    """
    def __init__(self, obs_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3, 
                 gamma=0.95, tau=0.01, hidden_dim=128, max_action=1.0):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = Actor(obs_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.critic = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_actor = Actor(obs_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.target_critic = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        
        # Initialize target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Noise for exploration
        self.noise_scale = 0.1
        
    def select_action(self, obs, add_noise=True, mask=None):
        """
        Select action using actor network.
        
        Args:
            obs: Observation tensor
            add_noise: Whether to add noise for exploration
            mask: Optional mask tensor
        
        Returns:
            Action tensor and attention weights
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            action, attention_weights = self.actor(obs_tensor, mask)
            
            if add_noise:
                noise = torch.randn_like(action) * self.noise_scale
                action = torch.clamp(action + noise, -self.max_action, self.max_action)
            
            return action.cpu().numpy(), attention_weights
    
    def update(self, batch, other_agents, agent_idx):
        """
        Update actor and critic networks.
        
        Args:
            batch: Batch of experiences
            other_agents: List of other agents
            agent_idx: Index of current agent
        """
        obs, actions, rewards, next_obs, dones = batch
        
        # Convert to tensors
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current agent's observations and actions
        current_obs = obs[:, agent_idx]
        current_actions = actions[:, agent_idx]
        current_rewards = rewards[:, agent_idx]
        current_next_obs = next_obs[:, agent_idx]
        current_dones = dones[:, agent_idx]
        
        # Update critic
        with torch.no_grad():
            # Get next actions from target actors
            next_actions = []
            for i, other_agent in enumerate(other_agents):
                if i == agent_idx:
                    next_action, _ = other_agent.target_actor(current_next_obs)
                else:
                    next_action, _ = other_agent.target_actor(next_obs[:, i])
                next_actions.append(next_action)
            
            next_actions = torch.cat(next_actions, dim=1)
            target_q, _ = self.target_critic(current_next_obs, next_actions)
            target_q = current_rewards.unsqueeze(1) + self.gamma * target_q * (~current_dones).unsqueeze(1)
        
        # Current Q-value
        current_q, _ = self.critic(current_obs, current_actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        # Get current actions from all agents
        current_actions_all = []
        for i, other_agent in enumerate(other_agents):
            if i == agent_idx:
                action, _ = self.actor(current_obs)
            else:
                action, _ = other_agent.actor(obs[:, i])
            current_actions_all.append(action)
        
        current_actions_all = torch.cat(current_actions_all, dim=1)
        
        # Actor loss (negative Q-value)
        actor_loss = -self.critic(current_obs, current_actions_all)[0].mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }
    
    def soft_update(self, target, source):
        """
        Soft update target network.
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def save(self, filepath):
        """
        Save agent parameters.
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """
        Load agent parameters.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

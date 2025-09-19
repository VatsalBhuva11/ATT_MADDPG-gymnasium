import numpy as np
import torch
from collections import deque

class MultiAgentReplayBuffer:
    """
    Replay buffer for multi-agent reinforcement learning.
    Stores experiences for all agents and provides sampling functionality.
    """
    def __init__(self, capacity, obs_dims, action_dims, num_agents):
        self.capacity = capacity
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.num_agents = num_agents
        
        # Initialize buffers for each agent
        self.observations = {i: deque(maxlen=capacity) for i in range(num_agents)}
        self.actions = {i: deque(maxlen=capacity) for i in range(num_agents)}
        self.rewards = {i: deque(maxlen=capacity) for i in range(num_agents)}
        self.next_observations = {i: deque(maxlen=capacity) for i in range(num_agents)}
        self.dones = {i: deque(maxlen=capacity) for i in range(num_agents)}
        
        # Global buffer for joint experiences
        self.global_observations = deque(maxlen=capacity)
        self.global_actions = deque(maxlen=capacity)
        self.global_rewards = deque(maxlen=capacity)
        self.global_next_observations = deque(maxlen=capacity)
        self.global_dones = deque(maxlen=capacity)
        
        self.size = 0
        
    def add(self, observations, actions, rewards, next_observations, dones):
        """
        Add experience to replay buffer.
        
        Args:
            observations: Dict of observations for each agent
            actions: Dict of actions for each agent
            rewards: Dict of rewards for each agent
            next_observations: Dict of next observations for each agent
            dones: Dict of done flags for each agent
        """
        # Add individual agent experiences
        for i in range(self.num_agents):
            self.observations[i].append(observations[i])
            self.actions[i].append(actions[i])
            self.rewards[i].append(rewards[i])
            self.next_observations[i].append(next_observations[i])
            self.dones[i].append(dones[i])
        
        # Add global experience
        self.global_observations.append(observations)
        self.global_actions.append(actions)
        self.global_rewards.append(rewards)
        self.global_next_observations.append(next_observations)
        self.global_dones.append(dones)
        
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        Sample batch of experiences.
        
        Args:
            batch_size: Size of batch to sample
        
        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones)
        """
        if self.size < batch_size:
            batch_size = self.size
        
        # Sample random indices
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        # Sample experiences
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        done_batch = []
        
        for i in range(self.num_agents):
            obs_batch.append([self.observations[i][idx] for idx in indices])
            action_batch.append([self.actions[i][idx] for idx in indices])
            reward_batch.append([self.rewards[i][idx] for idx in indices])
            next_obs_batch.append([self.next_observations[i][idx] for idx in indices])
            done_batch.append([self.dones[i][idx] for idx in indices])
        
        # Convert to numpy arrays
        obs_batch = np.array(obs_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_obs_batch = np.array(next_obs_batch)
        done_batch = np.array(done_batch)
        
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
    
    def sample_global(self, batch_size):
        """
        Sample batch of global experiences.
        
        Args:
            batch_size: Size of batch to sample
        
        Returns:
            Tuple of global experiences
        """
        if self.size < batch_size:
            batch_size = self.size
        
        # Sample random indices
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        # Sample global experiences
        obs_batch = [self.global_observations[idx] for idx in indices]
        action_batch = [self.global_actions[idx] for idx in indices]
        reward_batch = [self.global_rewards[idx] for idx in indices]
        next_obs_batch = [self.global_next_observations[idx] for idx in indices]
        done_batch = [self.global_dones[idx] for idx in indices]
        
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
    
    def __len__(self):
        return self.size

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for multi-agent learning.
    """
    def __init__(self, capacity, obs_dims, action_dims, num_agents, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.num_agents = num_agents
        self.alpha = alpha
        self.beta = beta
        
        # Initialize buffers
        self.observations = {i: deque(maxlen=capacity) for i in range(num_agents)}
        self.actions = {i: deque(maxlen=capacity) for i in range(num_agents)}
        self.rewards = {i: deque(maxlen=capacity) for i in range(num_agents)}
        self.next_observations = {i: deque(maxlen=capacity) for i in range(num_agents)}
        self.dones = {i: deque(maxlen=capacity) for i in range(num_agents)}
        
        # Priority buffer
        self.priorities = deque(maxlen=capacity)
        
        self.size = 0
        self.max_priority = 1.0
    
    def add(self, observations, actions, rewards, next_observations, dones, priority=None):
        """
        Add experience with priority.
        """
        if priority is None:
            priority = self.max_priority
        
        # Add individual agent experiences
        for i in range(self.num_agents):
            self.observations[i].append(observations[i])
            self.actions[i].append(actions[i])
            self.rewards[i].append(rewards[i])
            self.next_observations[i].append(next_observations[i])
            self.dones[i].append(dones[i])
        
        # Add priority
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
        
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        Sample batch with prioritized sampling.
        """
        if self.size < batch_size:
            batch_size = self.size
        
        # Calculate sampling probabilities
        priorities = np.array(list(self.priorities))
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, replace=False, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Sample experiences
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        done_batch = []
        
        for i in range(self.num_agents):
            obs_batch.append([self.observations[i][idx] for idx in indices])
            action_batch.append([self.actions[i][idx] for idx in indices])
            reward_batch.append([self.rewards[i][idx] for idx in indices])
            next_obs_batch.append([self.next_observations[i][idx] for idx in indices])
            done_batch.append([self.dones[i][idx] for idx in indices])
        
        # Convert to numpy arrays
        obs_batch = np.array(obs_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_obs_batch = np.array(next_obs_batch)
        done_batch = np.array(done_batch)
        
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, weights, indices
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for given indices.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.size

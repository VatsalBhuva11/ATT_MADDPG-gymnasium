import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils import parallel_to_aec

from models.maddpg import MADDPGAgent
from utils.replay_buffer import MultiAgentReplayBuffer
from visualization.plotter import TrainingPlotter

class ATT_MADDPG_Trainer:
    """
    Trainer for ATT-MADDPG algorithm on Cooperative Navigation environment.
    """
    def __init__(self, env_name="simple_spread_v3", num_agents=3, num_landmarks=3,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.95, tau=0.01,
                 buffer_size=100000, batch_size=1024, hidden_dim=128,
                 max_episodes=10000, save_interval=1000, eval_interval=100):
        
        self.env_name = env_name
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.max_episodes = max_episodes
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        
        # Create environment
        self.env = simple_spread_v3.parallel_env(
            N=num_agents, 
            local_ratio=0.5, 
            max_cycles=25, 
            continuous_actions=True
        )
        
        # Get observation and action dimensions
        self.obs_dims = []
        self.action_dims = []
        
        for agent in self.env.possible_agents:
            obs_space = self.env.observation_space(agent)
            action_space = self.env.action_space(agent)
            self.obs_dims.append(obs_space.shape[0])
            self.action_dims.append(action_space.shape[0])
        
        # Create agents
        self.agents = []
        for i in range(num_agents):
            agent = MADDPGAgent(
                obs_dim=self.obs_dims[i],
                action_dim=self.action_dims[i],
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                tau=tau,
                hidden_dim=hidden_dim
            )
            self.agents.append(agent)
        
        # Create replay buffer
        self.replay_buffer = MultiAgentReplayBuffer(
            capacity=buffer_size,
            obs_dims=self.obs_dims,
            action_dims=self.action_dims,
            num_agents=num_agents
        )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = [[] for _ in range(num_agents)]
        self.critic_losses = [[] for _ in range(num_agents)]
        
        # Create directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        
        # Initialize plotter
        self.plotter = TrainingPlotter()
        
    def train(self, visualize=False):
        """
        Train the ATT-MADDPG agents.
        """
        print(f"Starting ATT-MADDPG training on {self.env_name}")
        print(f"Number of agents: {self.num_agents}")
        print(f"Observation dimensions: {self.obs_dims}")
        print(f"Action dimensions: {self.action_dims}")
        print(f"Max episodes: {self.max_episodes}")
        print("-" * 50)
        
        best_reward = float('-inf')
        
        for episode in tqdm(range(self.max_episodes), desc="Training"):
            # Reset environment
            observations, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Store observations and actions for all agents
            obs_dict = {agent: [] for agent in self.env.possible_agents}
            action_dict = {agent: [] for agent in self.env.possible_agents}
            reward_dict = {agent: [] for agent in self.env.possible_agents}
            next_obs_dict = {agent: [] for agent in self.env.possible_agents}
            done_dict = {agent: [] for agent in self.env.possible_agents}
            
            while True:
                # Select actions for all agents
                actions = {}
                for i, agent in enumerate(self.env.possible_agents):
                    action, _ = self.agents[i].select_action(
                        observations[agent], 
                        add_noise=True
                    )
                    actions[agent] = action
                
                # Store observations
                for agent in self.env.possible_agents:
                    obs_dict[agent].append(observations[agent])
                
                # Step environment (returns obs, rewards, terminations, truncations, infos)
                next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
                dones = {a: (terminations[a] or truncations[a]) for a in self.env.possible_agents}
                
                # Store actions, rewards, next observations, and dones
                for agent in self.env.possible_agents:
                    action_dict[agent].append(actions[agent])
                    reward_dict[agent].append(rewards[agent])
                    next_obs_dict[agent].append(next_observations[agent])
                    done_dict[agent].append(dones[agent])
                
                # Update episode reward
                episode_reward += sum(rewards.values())
                episode_length += 1
                
                # Check if episode is done
                if all(dones.values()):
                    break
                
                observations = next_observations
            
            # Store episode data
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Add experiences to replay buffer per timestep
            num_steps = len(next_obs_dict[self.env.possible_agents[0]])
            for t in range(num_steps):
                obs_t = {i: obs_dict[self.env.possible_agents[i]][t] for i in range(self.num_agents)}
                act_t = {i: action_dict[self.env.possible_agents[i]][t] for i in range(self.num_agents)}
                rew_t = {i: reward_dict[self.env.possible_agents[i]][t] for i in range(self.num_agents)}
                next_obs_t = {i: next_obs_dict[self.env.possible_agents[i]][t] for i in range(self.num_agents)}
                done_t = {i: done_dict[self.env.possible_agents[i]][t] for i in range(self.num_agents)}
                self.replay_buffer.add(obs_t, act_t, rew_t, next_obs_t, done_t)
            
            # Update agents if buffer has enough samples
            if len(self.replay_buffer) >= self.batch_size:
                for i in range(self.num_agents):
                    batch = self.replay_buffer.sample(self.batch_size)
                    losses = self.agents[i].update(batch, self.agents, i)
                    
                    self.actor_losses[i].append(losses['actor_loss'])
                    self.critic_losses[i].append(losses['critic_loss'])
            
            # Evaluation
            if episode % self.eval_interval == 0:
                eval_reward = self.evaluate(num_episodes=5, visualize=False)
                print(f"Episode {episode}, Eval Reward: {eval_reward:.2f}")
                
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    self.save_models(f"best_model_episode_{episode}")
            
            # Save models
            if episode % self.save_interval == 0:
                self.save_models(f"checkpoint_episode_{episode}")
            
            # Plot training progress
            if episode % 100 == 0:
                self.plotter.plot_training_progress(
                    self.episode_rewards,
                    self.episode_lengths,
                    self.actor_losses,
                    self.critic_losses,
                    save_path=f"plots/training_progress_episode_{episode}.png"
                )
        
        print(f"Training completed! Best reward: {best_reward:.2f}")
        self.save_models("final_model")
        
    def evaluate(self, num_episodes=10, visualize=False):
        """
        Evaluate the trained agents.
        """
        total_rewards = []
        
        for episode in range(num_episodes):
            observations, _ = self.env.reset()
            episode_reward = 0
            
            while True:
                actions = {}
                for i, agent in enumerate(self.env.possible_agents):
                    action, _ = self.agents[i].select_action(
                        observations[agent], 
                        add_noise=False
                    )
                    actions[agent] = action
                
                next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
                dones = {a: (terminations[a] or truncations[a]) for a in self.env.possible_agents}
                episode_reward += sum(rewards.values())
                
                if all(dones.values()):
                    break
                
                observations = next_observations
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)
    
    def save_models(self, model_name):
        """
        Save all agent models.
        """
        for i, agent in enumerate(self.agents):
            agent.save(f"models/{model_name}_agent_{i}.pth")
        
        # Save training metrics
        np.save(f"logs/{model_name}_rewards.npy", self.episode_rewards)
        np.save(f"logs/{model_name}_lengths.npy", self.episode_lengths)
        
        for i in range(self.num_agents):
            np.save(f"logs/{model_name}_actor_losses_{i}.npy", self.actor_losses[i])
            np.save(f"logs/{model_name}_critic_losses_{i}.npy", self.critic_losses[i])
    
    def load_models(self, model_name):
        """
        Load all agent models.
        """
        for i, agent in enumerate(self.agents):
            agent.load(f"models/{model_name}_agent_{i}.pth")

def main():
    parser = argparse.ArgumentParser(description="Train ATT-MADDPG on Cooperative Navigation")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of training episodes")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--landmarks", type=int, default=3, help="Number of landmarks")
    parser.add_argument("--lr_actor", type=float, default=1e-4, help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="Critic learning rate")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization during training")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ATT_MADDPG_Trainer(
        num_agents=args.agents,
        num_landmarks=args.landmarks,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        max_episodes=args.episodes
    )
    
    # Train
    trainer.train(visualize=args.visualize)

if __name__ == "__main__":
    main()

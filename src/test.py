import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pettingzoo.mpe import simple_spread_v3
import argparse

from models.maddpg import MADDPGAgent
from visualization.plotter import TrainingPlotter

class ATT_MADDPG_Tester:
    """
    Tester for ATT-MADDPG algorithm with visualization.
    """
    def __init__(self, env_name="simple_spread_v3", num_agents=3, num_landmarks=3,
                 hidden_dim=128, model_path="models"):
        
        self.env_name = env_name
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.hidden_dim = hidden_dim
        self.model_path = model_path
        
        # Create environment
        self.env = simple_spread_v3.parallel_env(
            N=num_agents, 
            local_ratio=0.5, 
            max_cycles=25, 
            continuous_actions=True,
            render_mode="human"
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
                hidden_dim=hidden_dim
            )
            self.agents.append(agent)
        
        # Initialize plotter
        self.plotter = TrainingPlotter()
        
    def load_models(self, model_name):
        """
        Load trained models.
        """
        print(f"Loading models from {model_name}")
        
        for i, agent in enumerate(self.agents):
            model_file = f"{self.model_path}/{model_name}_agent_{i}.pth"
            if os.path.exists(model_file):
                agent.load(model_file)
                print(f"Loaded model for agent {i}")
            else:
                print(f"Warning: Model file {model_file} not found!")
                return False
        
        return True
    
    def test(self, num_episodes=10, visualize=True, save_video=False):
        """
        Test the trained agents with visualization.
        """
        print(f"Testing ATT-MADDPG on {self.env_name}")
        print(f"Number of episodes: {num_episodes}")
        print(f"Visualization: {visualize}")
        print("-" * 50)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")
            
            observations, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Store data for visualization
            if visualize or save_video:
                positions_history = []
                landmarks_positions = []
            
            while True:
                # Select actions for all agents
                actions = {}
                attention_weights = []
                
                for i, agent in enumerate(self.env.possible_agents):
                    action, attn_weights = self.agents[i].select_action(
                        observations[agent], 
                        add_noise=False
                    )
                    actions[agent] = action
                    attention_weights.append(attn_weights)
                
                # Store positions for visualization
                if visualize or save_video:
                    # Extract agent positions from observations
                    agent_positions = []
                    for i, agent in enumerate(self.env.possible_agents):
                        # First two elements are x, y positions
                        pos = observations[agent][:2]
                        agent_positions.append(pos)
                    positions_history.append(agent_positions)
                    
                    # Extract landmark positions (assuming they're in the observation)
                    # This is environment-specific and may need adjustment
                    landmark_pos = []
                    for i in range(self.num_landmarks):
                        # Landmark positions are typically after agent positions
                        landmark_start = 2 + i * 2
                        landmark_pos.append(observations[self.env.possible_agents[0]][landmark_start:landmark_start+2])
                    landmarks_positions.append(landmark_pos)
                
                # Step environment (returns obs, rewards, terminations, truncations, infos)
                next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
                dones = {a: (terminations[a] or truncations[a]) for a in self.env.possible_agents}
                
                # Update episode metrics
                episode_reward += sum(rewards.values())
                episode_length += 1
                
                # Render if visualization is enabled
                if visualize:
                    self.env.render()
                
                # Check if episode is done
                if all(dones.values()):
                    break
                
                observations = next_observations
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"Episode {episode + 1} - Reward: {episode_reward:.2f}, Length: {episode_length}")
            
            # Create visualization for this episode
            if visualize or save_video:
                self.plotter.plot_episode_trajectory(
                    positions_history,
                    landmarks_positions,
                    episode_reward,
                    episode_length,
                    save_path=f"plots/episode_{episode + 1}_trajectory.png" if save_video else None
                )
        
        # Print summary
        print("\n" + "="*50)
        print("TESTING SUMMARY")
        print("="*50)
        print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        print(f"Best Reward: {np.max(episode_rewards):.2f}")
        print(f"Worst Reward: {np.min(episode_rewards):.2f}")
        
        # Plot testing results
        self.plotter.plot_testing_results(
            episode_rewards,
            episode_lengths,
            save_path="plots/testing_results.png"
        )
        
        return episode_rewards, episode_lengths
    
    def create_attention_visualization(self, observations, attention_weights, save_path=None):
        """
        Create visualization of attention weights.
        """
        fig, axes = plt.subplots(1, len(self.agents), figsize=(5*len(self.agents), 5))
        if len(self.agents) == 1:
            axes = [axes]
        
        for i, (agent, attn_weights) in enumerate(zip(self.agents, attention_weights)):
            if attn_weights is not None:
                # Plot attention weights
                attn_matrix = attn_weights[0].cpu().numpy()  # First batch
                im = axes[i].imshow(attn_matrix, cmap='Blues', aspect='auto')
                axes[i].set_title(f'Agent {i} Attention Weights')
                axes[i].set_xlabel('Key Position')
                axes[i].set_ylabel('Query Position')
                plt.colorbar(im, ax=axes[i])
            else:
                axes[i].text(0.5, 0.5, 'No attention data', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Agent {i} - No Attention Data')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Test ATT-MADDPG on Cooperative Navigation")
    parser.add_argument("--model", type=str, required=True, help="Model name to load")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--landmarks", type=int, default=3, help="Number of landmarks")
    parser.add_argument("--no_visualize", action="store_true", help="Disable visualization")
    parser.add_argument("--save_video", action="store_true", help="Save episode videos")
    parser.add_argument("--model_path", type=str, default="models", help="Path to model files")
    
    args = parser.parse_args()
    
    # Create tester
    tester = ATT_MADDPG_Tester(
        num_agents=args.agents,
        num_landmarks=args.landmarks,
        model_path=args.model_path
    )
    
    # Load models
    if not tester.load_models(args.model):
        print("Failed to load models. Exiting.")
        return
    
    # Test
    tester.test(
        num_episodes=args.episodes,
        visualize=not args.no_visualize,
        save_video=args.save_video
    )

if __name__ == "__main__":
    main()

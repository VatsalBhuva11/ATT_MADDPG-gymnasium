import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle
import matplotlib.animation as animation

class TrainingPlotter:
    """
    Visualization tools for ATT-MADDPG training and testing.
    """
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_training_progress(self, episode_rewards, episode_lengths, 
                             actor_losses, critic_losses, save_path=None):
        """
        Plot training progress including rewards, lengths, and losses.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(episode_rewards, alpha=0.7, color=self.colors[0])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Moving average of rewards
        if len(episode_rewards) > 100:
            window = min(100, len(episode_rewards) // 10)
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(episode_rewards)), moving_avg, 
                           color=self.colors[1], linewidth=2, label=f'Moving Avg ({window})')
            axes[0, 0].legend()
        
        # Episode lengths
        axes[0, 1].plot(episode_lengths, alpha=0.7, color=self.colors[2])
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Actor losses
        for i, losses in enumerate(actor_losses):
            if losses:
                axes[1, 0].plot(losses, alpha=0.7, color=self.colors[i % len(self.colors)], 
                               label=f'Agent {i}')
        axes[1, 0].set_title('Actor Losses')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Critic losses
        for i, losses in enumerate(critic_losses):
            if losses:
                axes[1, 1].plot(losses, alpha=0.7, color=self.colors[i % len(self.colors)], 
                               label=f'Agent {i}')
        axes[1, 1].set_title('Critic Losses')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_testing_results(self, episode_rewards, episode_lengths, save_path=None):
        """
        Plot testing results.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Episode rewards
        axes[0].bar(range(len(episode_rewards)), episode_rewards, 
                   color=self.colors[0], alpha=0.7)
        axes[0].axhline(y=np.mean(episode_rewards), color=self.colors[1], 
                       linestyle='--', linewidth=2, label=f'Mean: {np.mean(episode_rewards):.2f}')
        axes[0].set_title('Test Episode Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[1].bar(range(len(episode_lengths)), episode_lengths, 
                   color=self.colors[2], alpha=0.7)
        axes[1].axhline(y=np.mean(episode_lengths), color=self.colors[3], 
                       linestyle='--', linewidth=2, label=f'Mean: {np.mean(episode_lengths):.2f}')
        axes[1].set_title('Test Episode Lengths')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Steps')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_episode_trajectory(self, positions_history, landmarks_positions, 
                               episode_reward, episode_length, save_path=None):
        """
        Plot agent trajectories during an episode.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Convert to numpy arrays
        positions_history = np.array(positions_history)
        landmarks_positions = np.array(landmarks_positions)
        
        # Plot landmarks
        for i, landmark_pos in enumerate(landmarks_positions[0]):
            circle = Circle(landmark_pos, 0.1, color=self.colors[i % len(self.colors)], 
                          alpha=0.5, label=f'Landmark {i}')
            ax.add_patch(circle)
        
        # Plot agent trajectories
        for i in range(positions_history.shape[1]):
            trajectory = positions_history[:, i, :]
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   color=self.colors[i % len(self.colors)], 
                   linewidth=2, alpha=0.7, label=f'Agent {i}')
            
            # Mark start and end positions
            ax.scatter(trajectory[0, 0], trajectory[0, 1], 
                      color=self.colors[i % len(self.colors)], 
                      s=100, marker='o', edgecolors='black', linewidth=2, 
                      label=f'Agent {i} Start' if i == 0 else "")
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                      color=self.colors[i % len(self.colors)], 
                      s=100, marker='s', edgecolors='black', linewidth=2, 
                      label=f'Agent {i} End' if i == 0 else "")
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'Episode Trajectory (Reward: {episode_reward:.2f}, Length: {episode_length})')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_attention_heatmap(self, attention_weights, agent_idx, save_path=None):
        """
        Plot attention heatmap for a specific agent.
        """
        if attention_weights is None or len(attention_weights) == 0:
            print("No attention weights available for visualization")
            return
        
        fig, axes = plt.subplots(1, len(attention_weights), figsize=(5*len(attention_weights), 5))
        if len(attention_weights) == 1:
            axes = [axes]
        
        for i, attn_weights in enumerate(attention_weights):
            if attn_weights is not None:
                attn_matrix = attn_weights[0].cpu().numpy()  # First batch
                im = axes[i].imshow(attn_matrix, cmap='Blues', aspect='auto')
                axes[i].set_title(f'Agent {agent_idx} - Attention Layer {i+1}')
                axes[i].set_xlabel('Key Position')
                axes[i].set_ylabel('Query Position')
                plt.colorbar(im, ax=axes[i])
            else:
                axes[i].text(0.5, 0.5, 'No attention data', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Agent {agent_idx} - Layer {i+1} - No Data')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_reward_distribution(self, rewards, save_path=None):
        """
        Plot distribution of episode rewards.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Histogram
        ax.hist(rewards, bins=20, alpha=0.7, color=self.colors[0], edgecolor='black')
        
        # Statistics
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        median_reward = np.median(rewards)
        
        ax.axvline(mean_reward, color=self.colors[1], linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_reward:.2f}')
        ax.axvline(median_reward, color=self.colors[2], linestyle='--', linewidth=2, 
                  label=f'Median: {median_reward:.2f}')
        
        ax.set_title('Episode Reward Distribution')
        ax.set_xlabel('Total Reward')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        textstr = f'Mean: {mean_reward:.2f}\nStd: {std_reward:.2f}\nMin: {np.min(rewards):.2f}\nMax: {np.max(rewards):.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def create_animation(self, positions_history, landmarks_positions, save_path=None):
        """
        Create animation of agent movements.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Convert to numpy arrays
        positions_history = np.array(positions_history)
        landmarks_positions = np.array(landmarks_positions)
        
        # Set up the plot
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title('Agent Movement Animation')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        
        # Plot landmarks
        landmarks = []
        for i, landmark_pos in enumerate(landmarks_positions[0]):
            circle = Circle(landmark_pos, 0.1, color=self.colors[i % len(self.colors)], 
                          alpha=0.5)
            ax.add_patch(circle)
            landmarks.append(circle)
        
        # Initialize agent positions
        agents = []
        for i in range(positions_history.shape[1]):
            agent, = ax.plot([], [], 'o', color=self.colors[i % len(self.colors)], 
                           markersize=10, label=f'Agent {i}')
            agents.append(agent)
        
        # Trajectory lines
        trajectories = []
        for i in range(positions_history.shape[1]):
            traj, = ax.plot([], [], color=self.colors[i % len(self.colors)], 
                          alpha=0.5, linewidth=1)
            trajectories.append(traj)
        
        ax.legend()
        
        def animate(frame):
            # Update agent positions
            for i, agent in enumerate(agents):
                pos = positions_history[frame, i, :]
                agent.set_data([pos[0]], [pos[1]])
            
            # Update trajectories
            for i, traj in enumerate(trajectories):
                traj_data = positions_history[:frame+1, i, :]
                traj.set_data(traj_data[:, 0], traj_data[:, 1])
            
            return agents + trajectories
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(positions_history),
                                     interval=100, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
        else:
            plt.show()
        
        return anim

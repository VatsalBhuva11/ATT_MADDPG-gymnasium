#!/usr/bin/env python3
"""
ATT-MADDPG Implementation for Cooperative Navigation
==================================================

This script implements the Attention Multi-Agent Deep Deterministic Policy Gradient
(ATT-MADDPG) algorithm for the Cooperative Navigation environment using PettingZoo.

Usage:
    python main.py train --episodes 10000 --agents 3 --visualize
    python main.py test --model best_model --episodes 10 --visualize

Author: AI Assistant
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train import ATT_MADDPG_Trainer
from test import ATT_MADDPG_Tester

def train_command(args):
    """Handle training command."""
    print("="*60)
    print("ATT-MADDPG Training for Cooperative Navigation")
    print("="*60)
    
    # Create trainer
    trainer = ATT_MADDPG_Trainer(
        num_agents=args.agents,
        num_landmarks=args.landmarks,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        max_episodes=args.episodes,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval
    )
    
    # Train
    trainer.train(visualize=args.visualize)
    
    print("\nTraining completed successfully!")
    print(f"Models saved in: models/")
    print(f"Logs saved in: logs/")
    print(f"Plots saved in: plots/")

def test_command(args):
    """Handle testing command."""
    print("="*60)
    print("ATT-MADDPG Testing for Cooperative Navigation")
    print("="*60)
    
    # Create tester
    tester = ATT_MADDPG_Tester(
        num_agents=args.agents,
        num_landmarks=args.landmarks,
        model_path=args.model_path
    )
    
    # Load models
    if not tester.load_models(args.model):
        print("Failed to load models. Please check the model path and name.")
        return
    
    # Test
    rewards, lengths = tester.test(
        num_episodes=args.episodes,
        visualize=not args.no_visualize,
        save_video=args.save_video
    )
    
    print(f"\nTesting completed!")
    print(f"Average reward: {sum(rewards)/len(rewards):.2f}")
    print(f"Average length: {sum(lengths)/len(lengths):.2f}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ATT-MADDPG for Cooperative Navigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default parameters
  python main.py train
  
  # Train with custom parameters
  python main.py train --episodes 5000 --agents 3 --lr_actor 1e-4 --visualize
  
  # Test trained model
  python main.py test --model best_model --episodes 10 --visualize
  
  # Test without visualization
  python main.py test --model best_model --episodes 5 --no_visualize
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train ATT-MADDPG agents')
    train_parser.add_argument('--episodes', type=int, default=10000, 
                             help='Number of training episodes (default: 10000)')
    train_parser.add_argument('--agents', type=int, default=3, 
                             help='Number of agents (default: 3)')
    train_parser.add_argument('--landmarks', type=int, default=3, 
                             help='Number of landmarks (default: 3)')
    train_parser.add_argument('--lr_actor', type=float, default=1e-4, 
                             help='Actor learning rate (default: 1e-4)')
    train_parser.add_argument('--lr_critic', type=float, default=1e-3, 
                             help='Critic learning rate (default: 1e-3)')
    train_parser.add_argument('--batch_size', type=int, default=1024, 
                             help='Batch size (default: 1024)')
    train_parser.add_argument('--buffer_size', type=int, default=100000, 
                             help='Replay buffer size (default: 100000)')
    train_parser.add_argument('--save_interval', type=int, default=1000, 
                             help='Model save interval (default: 1000)')
    train_parser.add_argument('--eval_interval', type=int, default=100, 
                             help='Evaluation interval (default: 100)')
    train_parser.add_argument('--visualize', action='store_true', 
                             help='Enable visualization during training')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test trained ATT-MADDPG agents')
    test_parser.add_argument('--model', type=str, required=True, 
                            help='Model name to load (e.g., best_model, final_model)')
    test_parser.add_argument('--episodes', type=int, default=10, 
                            help='Number of test episodes (default: 10)')
    test_parser.add_argument('--agents', type=int, default=3, 
                            help='Number of agents (default: 3)')
    test_parser.add_argument('--landmarks', type=int, default=3, 
                            help='Number of landmarks (default: 3)')
    test_parser.add_argument('--model_path', type=str, default='models', 
                            help='Path to model files (default: models)')
    test_parser.add_argument('--no_visualize', action='store_true', 
                            help='Disable visualization during testing')
    test_parser.add_argument('--save_video', action='store_true', 
                            help='Save episode videos')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'test':
        test_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

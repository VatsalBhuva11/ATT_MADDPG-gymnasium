#!/usr/bin/env python3
"""
Quick demo of ATT-MADDPG implementation.
This script runs a short training session and then tests the trained model.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import ATT_MADDPG_Trainer
from src.test import ATT_MADDPG_Tester

def main():
    print("="*60)
    print("ATT-MADDPG Demo - Cooperative Navigation")
    print("="*60)
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    print("\n1. Training ATT-MADDPG agents...")
    print("   (This will run for 100 episodes for demo purposes)")
    
    # Create trainer with reduced episodes for demo
    trainer = ATT_MADDPG_Trainer(
        num_agents=3,
        num_landmarks=3,
        max_episodes=100,  # Reduced for demo
        save_interval=50,
        eval_interval=25
    )
    
    # Train
    trainer.train(visualize=False)
    
    print("\n2. Testing trained model...")
    
    # Create tester
    tester = ATT_MADDPG_Tester(
        num_agents=3,
        num_landmarks=3
    )
    
    # Load the final model
    if tester.load_models("final_model"):
        # Test with visualization
        rewards, lengths = tester.test(
            num_episodes=3,
            visualize=True,
            save_video=True
        )
        
        print(f"\nDemo completed!")
        print(f"Test results:")
        print(f"  Average reward: {sum(rewards)/len(rewards):.2f}")
        print(f"  Average length: {sum(lengths)/len(lengths):.2f}")
        print(f"  Best reward: {max(rewards):.2f}")
        print(f"  Worst reward: {min(rewards):.2f}")
    else:
        print("Failed to load trained model!")

if __name__ == "__main__":
    main()

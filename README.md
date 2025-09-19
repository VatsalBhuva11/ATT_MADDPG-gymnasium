# ATT-MADDPG: Attention Multi-Agent Deep Deterministic Policy Gradient

This project implements the ATT-MADDPG (Attention Multi-Agent Deep Deterministic Policy Gradient) algorithm for the Cooperative Navigation environment using PettingZoo and Gymnasium.

## Overview

ATT-MADDPG extends the MADDPG algorithm by incorporating attention mechanisms to improve multi-agent coordination. The algorithm is particularly effective for environments where agents need to coordinate their actions based on the states of other agents.

### Key Features

- **Attention Mechanism**: Multi-head attention for processing multi-agent observations
- **MADDPG Core**: Actor-critic networks with target networks and experience replay
- **Cooperative Navigation**: Implementation on PettingZoo's simple_spread_v3 environment
- **Comprehensive Visualization**: Training progress, attention weights, and episode trajectories
- **Model Persistence**: Save and load trained models
- **Flexible Testing**: Optional visualization during testing

## Environment

The Cooperative Navigation environment (simple_spread_v3) features:
- Multiple agents that must reach different landmarks
- Continuous action space
- Partial observability
- Cooperative objectives requiring coordination
- Collision avoidance

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mini-proj-7th-sem
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train ATT-MADDPG agents with default parameters:
```bash
python main.py train
```

Train with custom parameters:
```bash
python main.py train --episodes 5000 --agents 3 --lr_actor 1e-4 --visualize
```

### Testing

Test trained models with visualization:
```bash
python main.py test --model best_model --episodes 10 --visualize
```

Test without visualization:
```bash
python main.py test --model best_model --episodes 5 --no_visualize
```

### Command Line Arguments

#### Training Arguments
- `--episodes`: Number of training episodes (default: 10000)
- `--agents`: Number of agents (default: 3)
- `--landmarks`: Number of landmarks (default: 3)
- `--lr_actor`: Actor learning rate (default: 1e-4)
- `--lr_critic`: Critic learning rate (default: 1e-3)
- `--batch_size`: Batch size (default: 1024)
- `--buffer_size`: Replay buffer size (default: 100000)
- `--save_interval`: Model save interval (default: 1000)
- `--eval_interval`: Evaluation interval (default: 100)
- `--visualize`: Enable visualization during training

#### Testing Arguments
- `--model`: Model name to load (required)
- `--episodes`: Number of test episodes (default: 10)
- `--agents`: Number of agents (default: 3)
- `--landmarks`: Number of landmarks (default: 3)
- `--model_path`: Path to model files (default: models)
- `--no_visualize`: Disable visualization during testing
- `--save_video`: Save episode videos

## Project Structure

```
mini-proj-7th-sem/
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
├── README.md              # This file
├── src/
│   ├── models/
│   │   ├── attention.py   # Attention mechanism implementation
│   │   └── maddpg.py      # MADDPG agent implementation
│   ├── utils/
│   │   └── replay_buffer.py # Experience replay buffer
│   ├── visualization/
│   │   └── plotter.py     # Visualization tools
│   ├── train.py           # Training script
│   └── test.py            # Testing script
├── models/                # Saved models (created during training)
├── logs/                  # Training logs (created during training)
└── plots/                 # Generated plots (created during training)
```

## Algorithm Details

### Attention Mechanism

The attention mechanism processes multi-agent observations using:
- Multi-head scaled dot-product attention
- Layer normalization and residual connections
- Configurable number of heads and layers

### MADDPG Implementation

- **Actor Networks**: Generate actions for each agent
- **Critic Networks**: Evaluate state-action pairs
- **Target Networks**: Stable learning targets
- **Experience Replay**: Store and sample experiences
- **Soft Updates**: Gradual target network updates

### Training Process

1. Initialize agents and environment
2. Collect experiences through exploration
3. Store experiences in replay buffer
4. Sample batches and update networks
5. Evaluate performance periodically
6. Save models and generate visualizations

## Visualization

The implementation includes comprehensive visualization tools:

- **Training Progress**: Episode rewards, lengths, and losses
- **Attention Weights**: Heatmaps showing attention patterns
- **Episode Trajectories**: Agent movement paths
- **Testing Results**: Performance statistics and distributions
- **Animations**: Dynamic visualization of agent behavior

## Results

The ATT-MADDPG algorithm typically achieves:
- Improved coordination compared to standard MADDPG
- Better sample efficiency through attention mechanisms
- Stable learning with proper hyperparameter tuning
- Effective cooperation in multi-agent environments

## Dependencies

- PyTorch >= 2.0.0
- Gymnasium >= 0.29.0
- PettingZoo >= 1.24.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- TensorBoard >= 2.13.0
- tqdm >= 4.65.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PettingZoo: A Standard API for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2009.14471)

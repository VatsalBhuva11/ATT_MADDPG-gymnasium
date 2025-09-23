# Attention-based MADDPG (PyTorch) with PettingZoo/Gymnasium

## What is being trained?
- Decentralized actors: One policy network per agent maps its own observation to a discrete action (logits -> categorical policy).
- Centralized attention critic: A shared critic estimates Q-values for each agent given all agents' observations and actions. It uses multi-head attention to attend over other agents' encoded state-action features.
- Training uses actor-critic updates with target networks and replay buffer (off-policy), following the MADDPG paradigm adapted to discrete actions via Gumbel-Softmax.

## Environment
- Custom PettingZoo parallel gridworld: each agent moves on a grid to reach its own goal.
- Observation: own position, own goal, relative positions of others (padded), grid size.
- Actions: 5 discrete actions {stay, up, down, left, right}.
- Reward: small step penalty, + shaping toward goal, +10 on reaching goal. Episode truncates at max steps or if all reach goals.
- Adjustable via args: number of agents, grid size, max steps.

## Requirements
- See `requirements.txt`. Create and activate a venv and install:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python train_attn_maddpg.py
```
Key flags inside `train()` you can change: `num_agents`, `grid_size`, `max_steps`, `episodes`, `device`. During training, `training_curve.png` is saved and the model is saved to `models/attn_maddpg.pt.actors_critic`.

Expect the average team reward per episode to increase over time as agents learn to reach their goals more efficiently.

## Test & Visualize
```bash
python test_attn_maddpg.py
```
This loads saved actors/critic and renders the multi-agent interaction using `pygame`. Adjust `episodes`, `grid_size`, `num_agents`, `max_steps` in the script call if needed.

## Notes
- The attention critic helps each agent focus on relevant others when estimating its value.
- Because actors are decentralized and the critic is centralized, execution at test time uses only actors. 
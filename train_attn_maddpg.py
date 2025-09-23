from __future__ import annotations
import os
from typing import List
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from envs.simple_grid_attn import simple_grid_v0
from maddpg_attn.agent import MADDPG
from maddpg_attn.buffer import ReplayBuffer
from maddpg_attn.utils import set_seed


def train(
    num_agents: int = 3,
    grid_size: int = 7,
    max_steps: int = 75,
    episodes: int = 400,
    batch_size: int = 256,
    buffer_capacity: int = 20000,
    start_learning: int = 1000,
    learn_every: int = 1,
    device: str = "cpu",
    save_path: str = "models/attn_maddpg.pt",
    seed: int = 42,
):
    set_seed(seed)
    env = simple_grid_v0(num_agents=num_agents, grid_size=grid_size, max_steps=max_steps, random_spawn=True, seed=seed)
    agent_ids = env.agents
    obs_spaces = env.observation_spaces()
    act_spaces = env.action_spaces()
    obs_dims = [int(obs_spaces[aid].shape[0]) for aid in agent_ids]
    act_dims = [int(act_spaces[aid].n) for aid in agent_ids]

    algo = MADDPG(obs_dims, act_dims, device=device)
    buffer = ReplayBuffer(num_agents, obs_dims, capacity=buffer_capacity, device=device)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    episode_rewards: List[float] = []
    global_step = 0

    for ep in tqdm(range(episodes), desc="Training"):
        obs, _ = env.reset()
        ep_return = 0.0
        for t in range(max_steps):
            # action
            actions = algo.act(obs, agent_ids, explore=True)
            next_obs, rewards, terms, truncs, infos = env.step(actions)

            # collect lists in agent order
            obs_list = [obs[aid].astype(np.float32) for aid in agent_ids]
            act_list = [int(actions[aid]) for aid in agent_ids]
            rew_list = [float(rewards[aid]) for aid in agent_ids]
            next_obs_list = [next_obs[aid].astype(np.float32) for aid in agent_ids]
            done_list = [bool(terms[aid] or truncs[aid]) for aid in agent_ids]
            buffer.add(obs_list, act_list, rew_list, next_obs_list, done_list)

            obs = next_obs
            ep_return += sum(rew_list) / len(rew_list)
            global_step += 1

            if buffer.can_sample(batch_size) and global_step >= start_learning and (global_step % learn_every == 0):
                batch = buffer.sample(batch_size)
                logs = algo.train_step(batch)

            if all(done_list):
                break
        episode_rewards.append(ep_return)

        # simple smoothing and print
        if (ep + 1) % 10 == 0:
            avg_last = np.mean(episode_rewards[-10:])
            print(f"Episode {ep+1}: avg team reward (last10) = {avg_last:.2f}")

    torch.save({"model": algo.state_dict(), "config": {"obs_dims": obs_dims, "act_dims": act_dims}}, save_path)
    algo.save(save_path + ".actors_critic")

    # plot
    plt.figure(figsize=(6,4))
    xs = np.arange(len(episode_rewards))
    plt.plot(xs, episode_rewards, label="episode team reward")
    if len(episode_rewards) > 20:
        # simple moving average via convolution
        window = np.ones(20) / 20.0
        smoothed = np.convolve(episode_rewards, window, mode='same')
        plt.plot(xs, smoothed, label="moving avg (20)")
    plt.xlabel("Episode")
    plt.ylabel("Team reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curve.png")
    print("Saved model and training curve.")


if __name__ == "__main__":
    train() 
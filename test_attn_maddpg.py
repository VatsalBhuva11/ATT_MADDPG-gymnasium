from __future__ import annotations
import time
import torch

from envs.simple_grid_attn import simple_grid_v0
from maddpg_attn.agent import MADDPG


def test(model_path: str = "models/attn_maddpg.pt.actors_critic", episodes: int = 5, grid_size: int = 7, num_agents: int = 3, max_steps: int = 75):
    env = simple_grid_v0(num_agents=num_agents, grid_size=grid_size, max_steps=max_steps, random_spawn=True)
    agent_ids = env.agents
    obs_spaces = env.observation_spaces()
    act_spaces = env.action_spaces()
    obs_dims = [int(obs_spaces[aid].shape[0]) for aid in agent_ids]
    act_dims = [int(act_spaces[aid].n) for aid in agent_ids]
    algo = MADDPG(obs_dims, act_dims)
    algo.load(model_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        for t in range(max_steps):
            env.render("human")
            actions = algo.act(obs, agent_ids, explore=False)
            obs, rewards, terms, truncs, infos = env.step(actions)
            ep_ret += sum(rewards.values()) / len(rewards)
            time.sleep(0.05)
            if all([terms[a] or truncs[a] for a in agent_ids]):
                break
        print(f"Test episode {ep+1} team reward: {ep_ret:.2f}")
    env.close()


if __name__ == "__main__":
    test() 
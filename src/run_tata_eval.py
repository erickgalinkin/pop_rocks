from itertools import repeat
from multiprocessing import Pool

import numpy as np
import torch
from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.game_modes.game_mode_db import GameModeDB
from yawning_titan.networks.network_db import NetworkDB

from adaptive_red import AdaptiveRed
from multiagent_env import MultiAgentEnv
from multiagent_yt_run import PPO, HierarchicalPPO

TRIALS = 1
TIMESTEPS = 1000

gdb = GameModeDB()
ndb = NetworkDB()
GAME_MODE = gdb.get("d2c92fc4-119d-42e8-83ab-45be03c64d49")
NETWORK = ndb.get("3b921390-cd7b-41c5-8120-5e9ac587d2f2")

network_interface = NetworkInterface(GAME_MODE, NETWORK)
# Create dummy env to establish env parameters
RED = AdaptiveRed(network_interface)
BLUE = BlueInterface(network_interface)
ENV = MultiAgentEnv(RED, BLUE, network_interface)

blue_state_dim = ENV.observation_space.shape[0]
blue_action_dim = ENV.action_space.n
red_state_dim = ENV.observation_space.shape[0]
red_action_dim = len(RED.action_dict)


# Evaluation function
def evaluate(env, red_agent, blue_agent, red_type, name):
    red_rewards = list()
    blue_rewards = list()
    episode_lengths = list()
    with torch.no_grad():
        for i in range(TRIALS):
            state = env.reset()
            red_ep_reward = 0
            blue_ep_reward = 0
            for t in range(1, TIMESTEPS):
                red_action = red_agent.select_action(state)
                blue_action = blue_agent.select_action(state)
                state, red_reward, blue_reward, done, notes = env.step(red_action, blue_action, red_type)
                red_ep_reward += red_reward
                blue_ep_reward += blue_reward
                if done:
                    episode_lengths.append(t)
                    red_rewards.append(red_ep_reward)
                    blue_rewards.append(blue_ep_reward)
                    break

    red_avg_reward = np.average(red_rewards)
    blue_avg_reward = np.average(blue_rewards)
    avg_ep_length = np.average(episode_lengths)

    print(f"Evaluation Statistics:\n"
          f"Average episode length: {avg_ep_length:.0f}\n"
          f"Average red reward: {red_avg_reward:.2f}\t"
          f"Average blue reward: {blue_avg_reward:.2f}")

    return name, red_rewards, blue_rewards


if __name__ == "__main__":
    # 1k training runs; k_epochs=5, lr_actor=0.0003, lr_critic=0.0005
    multi_type = "48a25525-5dfd-4333-a3c1-6585ac843fb7"
    hierarchical = "92238a08-4f95-4317-b8b2-6b6c200dc458"
    ransomware = "7f2699af-bdc1-49a2-ad98-eefc158cb3b6"
    apt = "029841f2-d7e9-4b69-a1c0-c2ccfccf6f99"

    mtrw_path = f"./saved_models/{multi_type}/ransomware.pth"
    mtapt_path = f"./saved_models/{multi_type}/apt.pth"
    mt_defender_path = f"./saved_models/{multi_type}/blue.pth"
    hi_defender_path = f"./saved_models/{hierarchical}/blue.pth"
    hi_rw_path = "./saved_models/d2ff5e73-e9d3-48d6-a2c2-aa74bdd0aedc/blue.pth"
    hi_apt_path = "./saved_models/98a681af-bf5f-430c-9e88-88b8b035a654/blue.pth"
    rw_path = f"./saved_models/{ransomware}/red.pth"
    rw_defender_path = f"./saved_models/{ransomware}/blue.pth"
    apt_path = f"./saved_models/{apt}/red.pth"
    apt_defender_path = f"./saved_models/{apt}/blue.pth"

    # Create all attacking agents (4)
    mtrw_agent = PPO(red_state_dim, red_action_dim)
    mtrw_agent.load(mtrw_path)
    mtapt_agent = PPO(red_state_dim, red_action_dim)
    mtapt_agent.load(mtapt_path)
    rw_agent = PPO(red_state_dim, red_action_dim)
    rw_agent.load(rw_path)
    apt_agent = PPO(red_state_dim, red_action_dim)
    apt_agent.load(apt_path)
    # Create all defending agents (4)
    mt_blue_agent = PPO(blue_state_dim, blue_action_dim)
    mt_blue_agent.load(mt_defender_path)
    hi_blue_agent = HierarchicalPPO(blue_state_dim, blue_action_dim, hi_rw_path, hi_apt_path)
    hi_blue_agent.load(hi_defender_path)
    rw_blue_agent = PPO(blue_state_dim, blue_action_dim)
    rw_blue_agent.load(rw_defender_path)
    apt_blue_agent = PPO(blue_state_dim, blue_action_dim)
    apt_blue_agent.load(apt_defender_path)
    envs = repeat(ENV)
    red_agents = [(mtrw_agent, "mtrw"), (mtapt_agent, "mtapt"), (rw_agent, "rw"), (apt_agent, "apt")]
    blue_agents = [(mt_blue_agent, "mt"), (hi_blue_agent, "hi"), (rw_blue_agent, "rw"), (apt_blue_agent, "apt")]
    args = list()
    for blue_agent, blue_name in blue_agents:
        for red_agent, red_name in red_agents:
            name = f"{blue_name}_{red_name}"
            red_type = "Ransomware" if "rw" in red_name else "APT"
            args.append((envs, red_agent, blue_agent, red_type, name))

    with Pool(processes=8) as pool:
        results = pool.starmap_async(evaluate, args)
        for r in results:
            print(r.get())

import json
import copy
from multiprocessing import Pool
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.game_modes.game_mode_db import GameModeDB
from yawning_titan.networks.network_db import NetworkDB

from adaptive_red import AdaptiveRed
from multiagent_env import MultiAgentEnv
from multiagent_yt_run import PPO, HierarchicalPPO

import warnings
warnings.filterwarnings("ignore")

TRIALS = 500
TIMESTEPS = 2 * TRIALS
RESULTS_PATH = Path("./tata_results/")

parser = ArgumentParser()
parser.add_argument("--game", type=str, help="Game ID", default="d2c92fc4-119d-42e8-83ab-45be03c64d49")
parser.add_argument("--setting", type=str, help="Game setting", choices=["standard", "scaled", "tiny", "zeroscale", "zerotiny"])
parser.add_argument("--output", type=str, help="Output filename", required=True)


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

    return name, red_rewards, blue_rewards


if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parser.parse_args()
    
    output_path = f"{RESULTS_PATH.joinpath(args.output)}.json"

    gdb = GameModeDB()
    ndb = NetworkDB()
    GAME_MODE = gdb.get(args.game)
    NETWORK = ndb.get("3b921390-cd7b-41c5-8120-5e9ac587d2f2")

    network_interface = NetworkInterface(GAME_MODE, NETWORK)
    # Create dummy env to establish env parameters
    RED = AdaptiveRed(network_interface)
    BLUE = BlueInterface(network_interface)
    ENV = MultiAgentEnv(RED, BLUE, network_interface, print_metrics=False, print_per_ts_data=False)

    blue_state_dim = ENV.observation_space.shape[0]
    blue_action_dim = ENV.action_space.n
    red_state_dim = ENV.observation_space.shape[0]
    red_action_dim = len(RED.action_dict)
    
    if args.setting.lower() == "scaled":
    # Scaled
        multi_type = "81450b82-847a-4733-967e-842e0d96709f"
        hierarchical = "379b4b51-c438-4165-96a1-2fb339db24f1"
        ransomware = "bbfeecd7-3a88-4fa8-82ad-4ed39d70aba0"
        apt = "47c3ae21-6d3f-48be-bbe7-aab51a2f05a5"
    
    elif args.setting.lower() == "tiny":
    # Tiny
        multi_type = "96744406-1d3e-427c-9a9c-6431ec0ff21e"
        hierarchical = "e226ed04-e2b5-45b7-b03a-2257868e2553"
        ransomware = "2fecf24e-6775-4c59-9c1f-3ed7a7d175a2"
        apt = "d028e4fc-0b22-43b9-9c09-01fa21c1e145"
    elif args.setting.lower() == "zeroscale":
    # Zeroscale
        ransomware = "b93772ab-cb80-480c-9273-3c8aefc28f11"
        apt = "0c39e550-6f66-4b11-aea8-ac731391cb16"
        hierarchical = "5127634f-c655-4d86-80b0-d21675a4638a"
        multi_type = "c2b29221-8d55-42cc-b8bb-5c6a17cd6108"
    elif args.setting.lower() == "zerotiny":
    # Zerotiny
        ransomware = "4a24954f-c835-40ee-b0fd-94d8a99bc52c"
        apt = "9f2db5c8-bc83-4e3b-acb1-8127ca7db849"
        hierarchical = "d8df4d88-7c36-4745-aa9a-9962a5bfa681"
        multi_type = "2454ddd5-39fa-4fb7-a70f-a74c1c3bba85"
    else:
    # Improved 3.5k
        multi_type = "0e867a54-df55-40b0-99fd-86b22e3f23a2"
        hierarchical = "c2186cbc-a96d-4683-bf09-a565ac8743e3"
        ransomware = "a94e9c2f-8a6a-45c5-9556-6a392b5401a7"
        apt = "a19a9954-c220-42d6-a274-5567a19a31ad"

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
    red_agents = [(mtrw_agent, "mtrw"), (mtapt_agent, "mtapt"), (rw_agent, "rw"), (apt_agent, "apt")]
    blue_agents = [(mt_blue_agent, "mt"), (hi_blue_agent, "hi"), (rw_blue_agent, "rw"), (apt_blue_agent, "apt")]
    args = list()
    for blue_agent, blue_name in blue_agents:
        for red_agent, red_name in red_agents:
            name = f"{blue_name}_{red_name}"
            red_type = "Ransomware" if "rw" in red_name else "APT"
            args.append((copy.deepcopy(ENV), copy.deepcopy(red_agent), copy.deepcopy(blue_agent), copy.deepcopy(red_type), copy.deepcopy(name)))
    result_dict = dict()

    with Pool(processes=4) as pool:
        results = pool.starmap(evaluate, args)
        for name, red_rewards, blue_rewards in results:
            result_dict[f"{name}_red_rewards"] = red_rewards
            result_dict[f"{name}_blue_rewards"] = blue_rewards
            
    with open(output_path, "w") as f:
        data = json.dump(result_dict, f)

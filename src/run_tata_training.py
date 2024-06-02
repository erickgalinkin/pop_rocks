import warnings
from multiprocessing import Pool

import torch.multiprocessing as mp
from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.game_modes.game_mode_db import GameModeDB
from yawning_titan.networks.network_db import NetworkDB

from adaptive_red import AdaptiveRed
from multiagent_env import MultiAgentEnv
from multiagent_yt_run import MultiAgentYTRun

warnings.filterwarnings("ignore")

TRIALS = 2500

gdb = GameModeDB()
ndb = NetworkDB()
GAME_MODE = gdb.get("d2c92fc4-119d-42e8-83ab-45be03c64d49")
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


# Training function
def train(run, multi=False, hierarchical=False, red_agent_type=""):
    if multi and not hierarchical:
        run.multi_type_setup()
        run.multi_type_train()
        run.multi_type_save()
    elif hierarchical:
        run.multi_type_setup(hierarchical=hierarchical)
        run.multi_type_train()
        run.multi_type_save()
    else:
        run.setup(red_agent_type=red_agent_type)
        run.train()
        run.save()


if __name__ == "__main__":
    mp.set_start_method('spawn')

    runs = list()
    mt_run = MultiAgentYTRun(network=NETWORK,
                             game_mode=GAME_MODE,
                             red_agent_class=RED,
                             blue_agent_class=BLUE,
                             auto=False,
                             training_runs=TRIALS)
    hi_run = MultiAgentYTRun(network=NETWORK,
                             game_mode=GAME_MODE,
                             red_agent_class=RED,
                             blue_agent_class=BLUE,
                             auto=False,
                             training_runs=TRIALS)
    rw_run = MultiAgentYTRun(network=NETWORK,
                             game_mode=GAME_MODE,
                             red_agent_class=RED,
                             blue_agent_class=BLUE,
                             auto=False,
                             training_runs=TRIALS)
    apt_run = MultiAgentYTRun(network=NETWORK,
                              game_mode=GAME_MODE,
                              red_agent_class=RED,
                              blue_agent_class=BLUE,
                              auto=False,
                              training_runs=TRIALS)

    runs.append((mt_run, True, False, ""))
    runs.append((hi_run, True, True, ""))
    runs.append((rw_run, False, False, "Ransomware"))
    runs.append((apt_run, False, False, "APT"))
    with Pool(processes=4) as pool:
        results = pool.starmap(train, runs)

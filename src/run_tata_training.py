import warnings

from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.game_modes.game_mode_db import GameModeDB
from yawning_titan.networks.network_db import NetworkDB

from adaptive_red import AdaptiveRed
from multiagent_env import MultiAgentEnv
from multiagent_yt_run import MultiAgentYTRun
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--type", type=str, help="What type of training to run", required=True)
parser.add_argument("--steps", type=int, help="How many training steps to run", required=False)

warnings.filterwarnings("ignore")

TRIALS = 3500

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
rw_path = "./saved_models/d2ff5e73-e9d3-48d6-a2c2-aa74bdd0aedc/blue.pth"
apt_path = "./saved_models/98a681af-bf5f-430c-9e88-88b8b035a654/blue.pth"


# Training function
def train(run, multi=False, hierarchical=False, red_agent_type=""):
    if multi and not hierarchical:
        run.multi_type_setup()
        run.multi_type_train()
        run.multi_type_save()
    elif hierarchical:
        run.multi_type_setup(hierarchical=hierarchical, pretrained_rw=rw_path, pretrained_apt=apt_path)
        run.multi_type_train()
        run.multi_type_save()
    else:
        run.setup(red_agent_type=red_agent_type)
        run.train()
        run.save()


if __name__ == "__main__":
    args = parser.parse_args()
    red = AdaptiveRed
    blue = BlueInterface
    yt_run = MultiAgentYTRun(network=NETWORK,
                             game_mode=GAME_MODE,
                             red_agent_class=red,
                             blue_agent_class=blue,
                             auto=False,
                             training_runs=TRIALS)

    if args.type == "multitype":
        train(yt_run, True, False)
    elif args.type == "hierarchical":
        train(yt_run, True, True)
    elif args.type == "ransomware":
        train(yt_run, False, False, "Ransomware")
    elif args.type == "apt":
        train(yt_run, False, False, "APT")
    else:
        print(f"{args.type} is not a valid run type!")

from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.envs.generic.helpers.eval_printout import EvalPrintout
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv

from adaptive_red import AdaptiveRed
from nsa_red import NSARed

from gym import spaces
import copy
from stable_baselines3.common.utils import set_random_seed
from typing import Tuple, Union, Optional, Dict, List
from logging import getLogger
from collections import Counter
import numpy as np
import math
import json

_LOGGER = getLogger(__name__)

REMOVE_RED_POINTS = []
for i in range(0, 101):
    REMOVE_RED_POINTS.append(round(math.exp(-0.004 * i), 4))

REDUCE_VULNERABILITY_POINTS = []
for i in range(1, 20):
    REDUCE_VULNERABILITY_POINTS.append(2 / (10 + math.exp(4 - 10 * (i / 20))) + 0.5)

SCANNING_USAGE_POINTS = []
for i in range(0, 100):
    SCANNING_USAGE_POINTS.append(-math.exp(-i) + 1)


class MultiAgentEnv(GenericNetworkEnv):
    """
    MultiAgent gym environment for Red + Blue learning
    """
    metadata = {"render_modes": ["human"], "name": "yt_multiagent"}

    def __init__(self,
                 red_agent: Union[AdaptiveRed, NSARed],
                 blue_agent: BlueInterface,
                 network_interface: NetworkInterface,
                 print_metrics: bool = True,
                 show_metrics_every: int = 1,
                 collect_additional_per_ts_data: bool = True,
                 print_per_ts_data: bool = True
                 ):
        """
        Initialize multiagent network environment.

        Args:
            :param red_agent: Object from RedInterface class
            :param blue_agent: Object from BlueInterface class
            :param network_interface: Object from NetworkInterface class
            :param print_metrics: Whether to print metrics
            :param show_metrics_every: Number of timesteps to show summary statistics
            :param collect_additional_per_ts_data: Whether to collect additional per timestep data
            :param print_per_ts_data: Whether to print collected per timestep data
        """
        super(MultiAgentEnv, self).__init__(red_agent, blue_agent, network_interface)

        self.RED = red_agent
        self.BLUE = blue_agent
        self.blue_actions = blue_agent.get_number_of_actions()
        self.red_actions = red_agent.get_number_of_actions()
        self.network_interface = network_interface
        self.current_duration = 0
        self.game_stats_list = []
        self.num_games_since_avg = 0
        self.avg_every = show_metrics_every
        self.current_game_blue = {}
        self.current_game_red = {}
        self.current_game_stats = {}
        self.total_games = 0
        self.made_safe_nodes = []
        self.blue_reward = 0
        self.red_reward = 0
        self.print_metrics = print_metrics
        self.print_notes = print_per_ts_data

        self.random_seed = self.network_interface.random_seed

        self.graph_plotter = None
        self.eval_printout = EvalPrintout(self.avg_every)

        self.blue_action_space = spaces.Discrete(self.blue_actions)
        self.red_action_space = spaces.Discrete(self.red_actions)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.network_interface.get_observation_size(),),
            dtype=np.float32
        )

        self.collect_data = collect_additional_per_ts_data
        self.env_observation = self.network_interface.get_current_observation()

    def reset(self) -> np.array:
        if self.random_seed is not None:  # conditionally set random_seed
            set_random_seed(self.random_seed, True)
        self.network_interface.reset()
        self.current_duration = 0
        self.env_observation = self.network_interface.get_current_observation()
        self.current_game_blue = {}
        self.current_game_red = {}

        return self.env_observation

    def step(self, red_action_id: int, blue_action_id: int, red_type:str) -> Tuple[np.array, float, float, bool, Dict[str, dict]]:
        """
        Take a time step and executes the actions for both Blue and Red RL agents.

        Args:
            red_action_id: The action value generated from the Red RL agent (int)
            blue_action_id: the action value generated from the Blue RL agent (int)

        Returns:
             A four tuple containing the next observation as a numpy array,
             the reward for that timesteps, a boolean for whether complete and
             additional notes containing timestep information from the environment.
        """
        # sets the nodes that have been made safe this turn to an empty list
        self.made_safe_nodes = []

        # set up initial variables that are reassigned based on the actions taken
        done = False
        red_reward = 0
        red_action = ""
        blue_reward = 0
        blue_action = ""
        blue_node = None

        notes = {
            "initial_state": self.network_interface.get_all_node_compromised_states(),
            "initial_blue_view": self.network_interface.get_all_node_blue_view_compromised_states(),
            "initial_vulnerabilities": self.network_interface.get_all_vulnerabilities(),
            "initial_red_location": copy.deepcopy(self.network_interface.red_current_location),
            "initial_graph": self.network_interface.get_current_graph_as_dict(),
            "current_step": self.current_duration,
        }

        # resets the attack list for the red agent (so that only the current turns attacks are held)
        self.network_interface.reset_stored_attacks()

        # The red agent performs their turn
        if (
                self.network_interface.game_mode.game_rules.grace_period_length.value
                <= self.current_duration
        ):
            red_action, red_node = self.RED.perform_action(red_action_id)
        else:
            red_action, red_node = ("", copy.deepcopy(self.network_interface.red_current_location))

        # Gets the number of nodes that are safe
        number_uncompromised = len(self.network_interface.current_graph.get_nodes(filter_true_safe=True))

        # Collects data on red action taken
        if self.collect_data:
            notes["red_info"] = (red_action, red_node)

        if red_action in self.current_game_red:
            self.current_game_blue[red_action] += 1
        else:
            self.current_game_blue[red_action] = 1

        # The states of the nodes after red has had their turn (Used by the reward functions)
        notes["post_red_state"] = self.network_interface.get_all_node_compromised_states()
        # Blues view of the environment after red has had their turn
        notes["post_red_blue_view"] = self.network_interface.get_all_node_blue_view_compromised_states()
        # A dictionary of vulnerabilities after red has had their turn
        notes["post_red_vulnerabilities"] = self.network_interface.get_all_vulnerabilities()
        # The isolation status of all the nodes
        notes["post_red_isolation"] = self.network_interface.get_all_isolation()

        # collects extra data if turned on
        if self.collect_data:
            # The location of the red agent after red has had their turn
            notes["post_red_red_location"] = copy.deepcopy(self.network_interface.red_current_location)

        # Check if the game is over and red has won
        if self.network_interface.game_mode.game_rules.blue_loss_condition.all_nodes_lost.value \
                and red_type == "Ransomware":
            if number_uncompromised == 0:
                done = True
                blue_reward = self.network_interface.game_mode.rewards.for_loss.value
                red_reward = self.network_interface.game_mode.rewards.for_reaching_max_steps.value
                blue_action = "failed"
        if self.network_interface.game_mode.game_rules.blue_loss_condition.n_percent_nodes_lost.use.value \
                and red_type == "Ransomware":
            # calculate the number of safe nodes
            percent_comp = len(self.network_interface.current_graph.get_nodes(filter_true_compromised=True)) / \
                           self.network_interface.current_graph.number_of_nodes()
            if percent_comp >= self.network_interface.game_mode.game_rules.blue_loss_condition.n_percent_nodes_lost.value.value:
                done = True
                blue_reward = self.network_interface.game_mode.rewards.for_loss.value
                red_reward = self.network_interface.game_mode.rewards.for_reaching_max_steps.value / 2
                # If the game ends before blue has had their turn the blue action is set to failed
                blue_action = "failed"
        if self.network_interface.game_mode.game_rules.blue_loss_condition.high_value_node_lost.value and red_type == "APT":
            # check if a high value node was compromised
            compromised_hvn = False
            for hvn in self.network_interface.current_graph.high_value_nodes:
                if hvn.true_compromised_status == 1:
                    compromised_hvn = True
                    break

            if compromised_hvn:
                # If this mode is selected then the game ends if the high value node has been compromised
                done = True
                blue_reward = self.network_interface.game_mode.rewards.for_loss.value
                red_reward = self.network_interface.game_mode.rewards.for_reaching_max_steps.value / 2
                blue_action = "failed"

        # if self.network_interface.gr_loss_tn:
        tn = self.network_interface.get_target_node()
        if tn is not None and self.network_interface.game_mode.game_rules.blue_loss_condition.target_node_lost.value \
                and red_type == "APT":
            if tn.true_compromised_status == 1:
                # If this mode is selected then the game ends if the target node has been compromised
                done = True
                blue_reward = self.network_interface.game_mode.rewards.for_loss.value
                red_reward = self.network_interface.game_mode.rewards.for_reaching_max_steps.value / 2
                blue_action = "failed"

        if done:
            if self.network_interface.game_mode.rewards.reduce_negative_rewards_for_closer_fails.value:
                blue_reward = blue_reward * (1 - (self.current_duration / self.network_interface.game_mode.game_rules.max_steps.value))
        if not done:
            blue_action, blue_node = self.BLUE.perform_action(blue_action_id)

            if blue_action == "make_node_safe" or blue_action == "restore_node":
                self.made_safe_nodes.append(blue_node)

            if blue_action in self.current_game_blue:
                self.current_game_blue[blue_action] += 1
            else:
                self.current_game_blue[blue_action] = 1

            # Special actions for NSARed, derived from YAWNING-TITAN nsa_node_def.py
            # https://github.com/dstl/YAWNING-TITAN/blob/main/src/yawning_titan/envs/specific/nsa_node_def.py
#             if isinstance(self.RED, NSARed):
#                 compromised_nodes = self.network_interface.current_graph.get_nodes(filter_true_compromised=True)
#                 if blue_action in ["make_node_safe", "restore_node"] and blue_node in compromised_nodes:
#                     chance = np.random.rand()
#                     if chance <= self.RED.chance_to_spread_during_patch:
#                         self.RED.spread()

            # calculates the reward from the current state of the network
            reward_args = {
                "network_interface": self.network_interface,
                "red_action": red_action,
                "red_node": red_node,
                "blue_action": blue_action,
                "blue_node": blue_node,
                "start_state": notes["initial_state"],
                "end_state": self.network_interface.get_all_node_compromised_states(),
                "start_vulnerabilities": notes["post_red_vulnerabilities"],
                "end_vulnerabilities": self.network_interface.get_all_vulnerabilities(),
                "start_isolation": notes["post_red_isolation"],
                "end_isolation": self.network_interface.get_all_isolation(),
                "start_blue": notes["post_red_blue_view"],
                "end_blue": self.network_interface.get_all_node_blue_view_compromised_states(),
            }

            red_reward, blue_reward = multiagent_rewards(reward_args)
            # NSARed agent modifies the cost of actions
#             if isinstance(self.RED, NSARed):
#                 if blue_action in ["make_node_safe", "restore_node"]:
#                     blue_reward = blue_reward - self.RED.cost_of_patch
#                 if blue_action == "isolate":
#                     blue_reward = blue_reward - self.RED.cost_of_isolate

            # gets the current observation from the environment
            self.env_observation = (
                self.network_interface.get_current_observation().flatten()
            )
            self.current_duration += 1

            # if the total number of steps reaches the set end then the blue agent wins and is rewarded accordingly
            if self.current_duration == self.network_interface.game_mode.game_rules.max_steps.value:
                blue_reward = self.network_interface.game_mode.rewards.for_reaching_max_steps.value
                red_reward = self.network_interface.game_mode.rewards.for_loss.value
                done = True

        # Gets the state of the environment at the end of the current time step
        if self.collect_data:
            # The blues view of the network
            notes["end_blue_view"] = self.network_interface.get_all_node_blue_view_compromised_states()
            # The state of the nodes (safe/compromised)
            notes["end_state"] = self.network_interface.get_all_node_compromised_states()
            # A dictionary of vulnerabilities
            notes["final_vulnerabilities"] = self.network_interface.get_all_vulnerabilities()
            # The location of the red agent
            notes["final_red_location"] = copy.deepcopy(self.network_interface.red_current_location)

        if self.network_interface.game_mode.miscellaneous.output_timestep_data_to_json.value:
            current_state = self.network_interface.create_json_time_step()
            self.network_interface.save_json(current_state, self.current_duration)

        if self.print_metrics and done:
            # prints end of game metrics such as who won and how long the game lasted
            self.num_games_since_avg += 1
            self.total_games += 1

            # Populate the current game's dictionary of stats with the episode winner and the number of timesteps
            if self.current_duration == self.network_interface.game_mode.game_rules.max_steps.value:
                self.current_game_stats = {
                    "Winner": "blue",
                    "Duration": self.current_duration,
                }
            else:
                self.current_game_stats = {
                    "Winner": "red",
                    "Duration": self.current_duration,
                }

            # Add the actions taken by blue during the episode to the stats dictionary
            self.current_game_stats.update(self.current_game_blue)

            # Add the current game dictionary to the list of dictionaries to average over
            self.game_stats_list.append(Counter(dict(self.current_game_stats.items())))

            # Every self.avg_every episodes, print the stats to console
            if self.num_games_since_avg == self.avg_every:
                self.eval_printout.print_stats(self.game_stats_list, self.total_games)

                self.num_games_since_avg = 0
                self.game_stats_list = []

        self.blue_reward = blue_reward
        self.red_reward = red_reward

        if self.collect_data:
            notes["safe_nodes"] = len(self.network_interface.current_graph.get_nodes(filter_true_safe=True))
            notes["blue_action"] = blue_action
            notes["blue_node"] = blue_node
            notes["attacks"] = self.network_interface.true_attacks
            notes["end_isolation"] = self.network_interface.get_all_isolation()

        # Returns the environment information that AI gym uses and all the information collected in a dictionary
        return self.env_observation, red_reward, blue_reward, done, notes


def multiagent_rewards(args: dict, security: float = 1, efficiency: float = 1) -> Tuple[float, float]:
    """
    Calculate the reward for the current state of the environment.

    Actions cost a certain amount and blue gets rewarded for removing red nodes and
    reducing the vulnerability of nodes

    Args:
        args: A dictionary containing the following items:
            network_interface: Interface with the network
            blue_action: The action that the blue agent has taken this turn
            blue_node: The node that the blue agent has targeted for their action
            red_action: The action the red agent has taken this turn
            red_node: The node that the red agent has targeted for their action
            start_state: The state of the nodes before the agents have taken their action
            end_state: The state of the nodes after the agents have taken their action
            start_vulnerabilities: The vulnerabilities before the turn
            end_vulnerabilities: The vulnerabilities after the turn
            start_isolation: The isolation status of all the nodes at the start of a turn
            end_isolation: The isolation status of all the nodes at the end of a turn
            start_blue: The env as the blue agent can see it before the blue agents turn
            end_blue: The env as the blue agent can see it after the blue agents turn

    Returns:
        The reward earned for this specific turn for the blue agent
    """
    # Get information about the current state of the environment
    network_interface: NetworkInterface = args["network_interface"]
    blue_action = args["blue_action"]
    red_action = args["red_action"]
    start_state = args["start_state"]
    end_state = args["end_state"]
    start_blue = args["start_blue"]
    end_blue = args["end_blue"]

    # cost for actions
    blue_action_cost = {
        "reduce_vulnerability": 1.5,
        "restore_node": 3,
        "make_node_safe": 2,
        "scan": 0.5,
        "isolate": 5,
        "connect": 0.5,
        "do_nothing": 0,
        "add_deceptive_node": 3,
    }

    red_action_cost = {
        "basic_attack": 2,
        "do_nothing": 0,
        "random_move": 0.5,
        "zero_day": 6,
        "spread": 10,
        "intrude": 10,
    }

    blue_reward = -blue_action_cost[blue_action]
    red_reward = -red_action_cost[red_action]
    # calculating number of red nodes before and after the blue agents turn
    initial_cumulative_states = sum(start_state.values())
    final_cumulative_states = sum(end_state.values())
    # calculating the vulnerability of all the nodes before and after the blue agents turn
    # blue's view
    initial_blue_states = sum(start_blue.values())
    final_blue_states = sum(end_blue.values())

    # rewards for removing red nodes
    # gives between 0.5 and 1
    if blue_action == "make_node_safe" or blue_action == "restore_node":
        if initial_blue_states > final_blue_states:
            blue_reward += REMOVE_RED_POINTS[round(100 * final_cumulative_states / network_interface.current_graph.number_of_nodes())]
        # elif initial_cumulative_states >= final_cumulative_states:
        #     pass
        # else:
        #     blue_reward -= 1

    if final_cumulative_states > initial_cumulative_states:
        red_reward += 1

    # punish agent for doing nothing if there are large numbers or red controlled nodes in the environment
    if blue_action == "do_nothing":
        blue_reward = blue_reward - (0.5 * final_cumulative_states)

    if red_action == "do_nothing":
        red_reward = red_reward - (0.5 * final_cumulative_states)

    vision_score = 0.01 * (final_cumulative_states - final_blue_states) ** 2

    blue_reward -= vision_score
    red_reward += vision_score

    return red_reward, blue_reward

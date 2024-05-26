from yawning_titan.envs.generic.core.red_action_set import RedActionSet
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.networks.node import Node
import random

from typing import Tuple, Dict, List, Union, Set

from logging import getLogger

_LOGGER = getLogger(__name__)


class AdaptiveRed(RedActionSet):
    def __init__(self, network_interface: NetworkInterface):
        """
        Initialise the red interface.

        Args:
            network_interface: Object from the NetworkInterface class
        """
        self.network_interface = network_interface
        self.non_attacking_actions = ["do_nothing", "random_move"]

        self.action_dict = {}
        action_set = []
        action_number = 0

        # Goes through the actions that the red agent can perform
        if self.network_interface.game_mode.red.action_set.spread.use.value:
            # If the action is enabled in the settings files then add to list of possible actions
            self.action_dict[action_number] = self.spread
            action_set.append(action_number)
            action_number += 1
        if self.network_interface.game_mode.red.action_set.random_infect.use.value:
            self.action_dict[action_number] = self.intrude
            action_set.append(action_number)
            action_number += 1
        if self.network_interface.game_mode.red.action_set.basic_attack.use.value:
            self.action_dict[action_number] = self.basic_attack
            action_set.append(action_number)
            action_number += 1
        if self.network_interface.game_mode.red.action_set.do_nothing.use.value:
            self.action_dict[action_number] = self.do_nothing
            action_set.append(action_number)
            action_number += 1
        if self.network_interface.game_mode.red.action_set.move.use.value:
            self.action_dict[action_number] = self.random_move
            action_set.append(action_number)
            action_number += 1
        if self.network_interface.game_mode.red.action_set.zero_day.use.value:
            self.action_dict[action_number] = self.zero_day_attack
            action_set.append(action_number)
            action_number += 1

        super().__init__(network_interface, action_set, [])

    def choose_target_node(self) -> Union[Tuple[Node, Node], Tuple[bool, bool]]:
        """
        Choose a target node.

        Returns:
            The target node (False if no possible nodes to attack)
            The node attacking the target node (False if no possible nodes to attack)
        """
        # creates a set of nodes that the red agent could attack
        possible_to_attack: Set[Node] = set()
        original_node = {}
        if self.network_interface.game_mode.red.agent_attack.attack_from.any_red_node.value:
            nodes = self.network_interface.current_graph.get_nodes(filter_true_compromised=True)
            # runs through the connected nodes and adds the safe nodes to a set of possible nodes to attack
            for node in nodes:
                # If red can attack from any compromised node
                connected = self.network_interface.get_current_connected_nodes(node)
                for connected_node in connected:
                    if connected_node.true_compromised_status == 0:
                        original_node[connected_node] = node
                        possible_to_attack.add(connected_node)
        elif self.network_interface.game_mode.red.agent_attack.attack_from.only_main_red_node.value:
            if self.network_interface.red_current_location is None:
                return
            # If red can only attack from the central red node
            connected = self.network_interface.get_current_connected_nodes(self.network_interface.red_current_location)
            for node in connected:
                if node.true_compromised_status == 0:
                    original_node[node] = self.network_interface.red_current_location
                    possible_to_attack.add(node)
        # also adds entry nodes into the set of possible nodes. This is the red agents entrance into the network

        for node in self.network_interface.current_graph.entry_nodes:
            if node.true_compromised_status == 0:
                possible_to_attack.add(node)
                original_node[node] = None

        possible_to_attack = sorted(list(possible_to_attack))

        weights = []
        # red can prioritise nodes based on some different parameters chosen in the settings menu
        if self.network_interface.game_mode.red.target_mechanism.random.value:
            # equal weighting for all nodes
            weights = [1] * len(possible_to_attack)
        elif self.network_interface.game_mode.red.target_mechanism.prioritise_connected_nodes.value:
            for node in possible_to_attack:
                # more connections means a higher weight
                weights.append(len(self.network_interface.get_current_connected_nodes(node)))
        elif self.network_interface.game_mode.red.target_mechanism.prioritise_unconnected_nodes.value:
            for node in possible_to_attack:
                # higher connections means a lower weight
                current_connected = len(self.network_interface.get_current_connected_nodes(node))
                if current_connected == 0:
                    current_connected = 0.1
                weights.append(1 / current_connected)
        elif self.network_interface.game_mode.red.target_mechanism.prioritise_vulnerable_nodes.value:
            for node in possible_to_attack:
                # higher vulnerability means a higher weight
                weights.append(1 / node.vulnerability_score)
        elif self.network_interface.game_mode.red.target_mechanism.prioritise_resilient_nodes.value:
            for node in possible_to_attack:
                # higher vulnerability means a lower weight
                weights.append(1 / node.vulnerability_score)
        elif self.network_interface.game_mode.red.target_mechanism.target_specific_node.use.value or self.network_interface.game_mode.red.target_mechanism.target_specific_node.target.value is not None:
            distances = self.network_interface.get_shortest_distances_to_target(possible_to_attack)
            for dist in distances:
                if self.network_interface.game_mode.red.target_mechanism.target_specific_node.always_choose_shortest_distance.value:
                    weight = 1 if dist == min(distances) else 0
                else:
                    weight = 1 if dist == 0 else dist / sum(distances)
                weights.append(weight)
        else:
            # if using the configuration checker then this should never happen
            raise Exception(
                "Red should have have a method for how it chooses nodes to attack (enable "
                "red_chooses_targets_at_random in the config file if you are unsure)"
            )

        if len(possible_to_attack) == 0:
            # If the red agent cannot attack anything then return False showing that the attack has failed
            return False, False
        if sum(weights) == 0:
            for counter, _ in enumerate(weights):
                weights[counter] = 1
        weights_normal = [float(i) / sum(weights) for i in weights]
        # Chooses a target with some being more likely than others
        target = random.choices(population=possible_to_attack, weights=weights_normal, k=1)[0]

        # get the node that red attacked from
        attacking_node = original_node[target]
        return target, attacking_node

    def zero_day_attack(self) -> Dict[str, List[Union[bool, str, None]]]:
        """
        Execute a zero-day attack if available.

        Returns:
            The name of the action taken
            If the action succeeded
            The target node
            The attacking node
        """
        if self.get_amount_zero_day() >= 1:
            # Can only use this if there are available zero days
            target, attacking_node = self.choose_target_node()
            if target is False:
                return {
                    "Action": "no_possible_targets",
                    "Attacking_Nodes": [],
                    "Target_Nodes": [],
                    "Successes": [False],
                }
            self.zero_day_amount -= 1
            self.network_interface.attack_node(target, guarantee=True)
            # Moves the red agent to the attacked location
            if self.network_interface.red_current_location is None:
                # moves the red agent into the network if it is not currently
                if target in self.network_interface.current_graph.entry_nodes:
                    self.network_interface.red_current_location = target
            elif target in self.network_interface.get_current_connected_nodes(self.network_interface.red_current_location):
                self.network_interface.red_current_location = target
            return {
                "Action": "zero_day",
                "Attacking_Nodes": [attacking_node],
                "Target_Nodes": [target],
                "Successes": [True],
            }
        else:
            return {
                "Action": "zero_day",
                "Attacking_Nodes": [],
                "Target_Nodes": [],
                "Successes": [False],
            }

    def perform_action(self, action: int) -> Tuple[str, Node]:
        """
        Chooses and then performs an action.

        This is called for every one of the red agents turns

        Returns:
            A tuple containing the name of the action, the success status, the target, the attacking node and any natural spreading attacks
        """

        # increments the day for the zero day
        if self.network_interface.game_mode.red.action_set.zero_day.use.value:
            self.increment_day()

        if action >= self.get_number_of_actions():
            action_info = self.do_nothing()
            red_action = action_info["Action"]
            target_node = action_info["Target_Nodes"]

            return red_action, target_node

        if self.network_interface.game_mode.red.natural_spreading.capable.value:
            action_info = self.natural_spread()

        # performs the action
        action_info = self.action_dict[action]()
        red_action = action_info["Action"]
        attacking_node = action_info["Attacking_Nodes"]
        target_node = action_info["Target_Nodes"]
        success = action_info["Successes"]

        # If there are no possible targets for an attack then red will attempt to move to a new node
        if action_info["Action"] == "no_possible_targets":
            action_info = self.random_move()
            red_action = action_info["Action"]
            attacking_node = action_info["Attacking_Nodes"]
            target_node = action_info["Target_Nodes"]
            success = action_info["Successes"]

        all_attacking = attacking_node
        all_target = target_node
        all_success = [success]
        try:
            self.network_interface.update_stored_attacks(all_attacking, all_target, all_success)
            
        except Exception as e:
            print(all_attacking, all_target, all_success, action_info)
            pass

        return red_action, target_node

    def get_number_of_actions(self):
        return len(self.action_dict)


class BayesHurwiczRed(AdaptiveRed):
    def __init__(self, network_interface):
        super().__init__(network_interface)

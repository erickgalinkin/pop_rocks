import logging
import random
import numpy as np
from typing import Dict, List, Set, Tuple, Union
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.envs.generic.core.red_action_set import RedActionSet
from yawning_titan.networks.node import Node

logger = logging.getLogger(__name__)


class NSARed(RedActionSet):
    """
    Modification of:
    https://github.com/dstl/YAWNING-TITAN/blob/main/src/yawning_titan/agents/nsa_red.py#L8

    Provides the red agent behaviour within the 50-node ransomware multi-agent environments.

    The agent is a loose replication of:
    https://www.nsa.gov/portals/70/documents/resources/everyone/digital-media-center/publications/the-next-wave/TNW-22-1.pdf#page=9
    """

    def __init__(self, network_interface: NetworkInterface):
        self.network_interface = network_interface
        # Derived from nsa_node def. See YAWNING-TITAN github:
        # https://github.com/dstl/YAWNING-TITAN/blob/main/src/yawning_titan/envs/specific/nsa_node_def.py#L107
        self.skill: float = 1.0
        self.chance_to_spread_during_patch: float = 0.01
        self.chance_to_randomly_compromise: float = 0.15
        self.spread_vs_random_intrusion: float = 0.5
        self.cost_of_isolate: float = 10
        self.cost_of_patch: float = 5
        self.chance_to_spread: float = 0.01
        action_set, action_probabilities, action_dict = self._get_action_dict()
        self.action_dict = action_dict
        super().__init__(network_interface, action_set, action_probabilities)

    def _get_action_dict(self) -> Tuple[List, List, Dict]:
        action_set = list()
        action_probabilities = list()
        action_dict = dict()
        action_number = 0
        # Assign actions, probabilities derived from NodeEnv
        action_dict[action_number] = self.spread
        action_set.append(action_number)
        action_probabilities.append(self.spread_vs_random_intrusion)
        action_number += 1
        action_dict[action_number] = self.intrude
        action_set.append(action_number)
        action_probabilities.append(1 - self.spread_vs_random_intrusion)
        action_number += 1
        action_dict[action_number] = self.basic_attack
        action_set.append(action_number)
        action_probabilities.append(0.1)
        action_number += 1
        action_dict[action_number] = self.zero_day_attack
        action_set.append(action_number)
        action_probabilities.append(0.1)

        # Normalize action_probabilities
        action_probabilities = [float(prob)/sum(action_probabilities) for prob in action_probabilities]

        return action_set, action_probabilities, action_dict

    def choose_target_node(self) -> Union[Tuple[Node, Node], Tuple[bool, bool]]:
        """
        Choose a node to act on.

        Returns:
            The node to act on

        """
        possible_to_attack: Set[Node] = set()
        original_node = dict()
        nodes = self.network_interface.current_graph.get_nodes(
            filter_true_compromised=True
        )
        # runs through the connected nodes and adds the safe nodes to a set of possible nodes to attack
        for node in nodes:
            # If red can attack from any compromised node
            connected = self.network_interface.get_current_connected_nodes(node)
            for connected_node in connected:
                if connected_node.true_compromised_status == 0:
                    original_node[connected_node] = node
                    possible_to_attack.add(connected_node)
        for node in self.network_interface.current_graph.entry_nodes:
            if node.true_compromised_status == 0:
                possible_to_attack.add(node)
                original_node[node] = None

        possible_to_attack = sorted(list(possible_to_attack))

        if len(possible_to_attack) == 0:
            # If the red agent cannot attack anything then return False showing that the attack has failed
            return False, False
        else:
            target = random.choices(possible_to_attack)[0]
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
            self.network_interface.red_current_location = target
            return {
                "Action": "zero_day",
                "Attacking_Nodes": [attacking_node],
                "Target_Nodes": [target],
                "Successes": [True],
            }
        else:
            # Never fail open on zero day
            return self.basic_attack()

    def basic_attack(self) -> Dict[str, List[Union[bool, str, None]]]:
        """
        Perform a basic attack on a targeted node.

        The red agent will attempt to compromise a target node using the predefined attack method.

        Returns:
            The name of the action taken
            If the action succeeded
            The target node
            The attacking node
        """
        target, attacking_node = self.choose_target_node()
        if target is False:
            return {
                "Action": "no_possible_targets",
                "Attacking_Nodes": [],
                "Target_Nodes": [],
                "Successes": [False],
            }
        attack_status = self.network_interface.attack_node(
            target,
            skill=self.skill,
            use_skill=self.network_interface.game_mode.red.agent_attack.skill.use.value,
            use_vulnerability=(
                not self.network_interface.game_mode.red.agent_attack.ignores_defences.value
            ),
            guarantee=self.network_interface.game_mode.red.agent_attack.always_succeeds.value,
        )
        if attack_status:
            # update the location of the red agent if applicable
            if self.network_interface.red_current_location is None:
                if target in self.network_interface.current_graph.entry_nodes:
                    self.network_interface.red_current_location = target
            elif target in self.network_interface.get_current_connected_nodes(
                self.network_interface.red_current_location
            ):
                self.network_interface.red_current_location = target
            return {
                "Action": "basic_attack",
                "Attacking_Nodes": [attacking_node],
                "Target_Nodes": [target],
                "Successes": [True],
            }
        else:
            return {
                "Action": "basic_attack",
                "Attacking_Nodes": [attacking_node],
                "Target_Nodes": [target],
                "Successes": [False],
            }

    def move(self, target: int, args: Tuple[List[List[float]], int, int, bool]):
        """
        Move the red agent from one node to another.

        Args:
            target: the node the agent is moving to
            args: A tuple containing the following:
                * "red_current_node": the current node of the agent
                * "able_to_move": if the agent is able to move
        """
        [_, red_current_node, _, able_to_move] = args
        if able_to_move:
            logger.debug(f"Red Team: Moved to target {target + 1}")
            self.update_location(target, red_current_node)

    def spread(self) -> Dict[str, List[Union[bool, str, None]]]:
        """
        Attempt to spread to all nodes connected to a compromised node.
        """
        chance_to_spread = self.chance_to_spread
        nodes = list()
        attacking_nodes = list()
        success = list()
        compromised_nodes = self.network_interface.current_graph.get_nodes(filter_true_compromised=True)
        # runs through all compromised nodes
        for node in compromised_nodes:
            if node is None:
                # If red does not control any nodes then the entry nodes are used
                connected_nodes = self.network_interface.current_graph.entry_nodes
                connected_nodes = [
                    n for n in connected_nodes if n.true_compromised_status == 0
                ]
                attacking_nodes.extend([None] * len(connected_nodes))
            else:
                connected_nodes = self.network_interface.get_current_connected_nodes(node)
                connected_nodes = [n for n in connected_nodes if n.true_compromised_status == 0]
                attacking_nodes.extend([node] * len(connected_nodes))
            for connected_node in connected_nodes:
                chance = random.random()
                if chance <= chance_to_spread:
                    attack_status = self.network_interface.attack_node(
                        connected_node,
                        skill=self.network_interface.game_mode.red.action_set.spread.chance.value,
                        use_skill=True,
                        use_vulnerability=(
                            not self.network_interface.game_mode.red.agent_attack.ignores_defences.value
                        ),
                        guarantee=self.network_interface.game_mode.red.agent_attack.always_succeeds.value,
                    )
                    if attack_status:
                        # If the attack succeeds
                        if node == self.network_interface.red_current_location:
                            self.network_interface.red_current_location = connected_node
                        # Spread can attack multiple nodes in one go; the agent remembers the success of each attack
                        success.append(True)
                    else:
                        success.append(False)

        return {
            "Action": "spread",
            "Attacking_Nodes": attacking_nodes,
            "Target_Nodes": nodes,
            "Successes": success,
        }

    def intrude(self) -> Dict[str, List[Union[bool, str, None]]]:
        """
        Attempt to randomly intrude every uncompromised node.

                The red agent will try to infect every safe node at once (regardless of connectivity).
        The chance for the red agent to compromise a node is independent to each of the other nodes

        Returns:
            The name of the action
            A list of success status for each node attacked
            A list of the target nodes
            A list of the attacking nodes
        """
        chance_to_randomly_compromise = self.chance_to_randomly_compromise
        safe_nodes = self.network_interface.current_graph.get_nodes(
            filter_true_safe=True
        )
        success = []
        nodes = []
        attacking_nodes = []
        # tries to attack the safe nodes
        for node in safe_nodes:
            attack_status = self.network_interface.attack_node(
                node,
                skill=chance_to_randomly_compromise,
                use_skill=True,
            )
            nodes.append(node)
            if attack_status:
                # Agent remembers each of the successes or failures for each attempt
                success.append(True)
            else:
                success.append(False)
            attacking_nodes.append(None)
        return {
            "Action": "intrude",
            "Attacking_Nodes": attacking_nodes,
            "Target_Nodes": nodes,
            "Successes": success,
        }

    def perform_action(self, action: int) -> Tuple[str, Node]:
        """
        Chooses and then performs an action.

        This is called for every one of the red agents turns

        Returns:
            A tuple containing the name of the action, the success status, the target,
            the attacking node and any natural spreading attacks
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
        self.network_interface.update_stored_attacks(all_attacking, all_target, all_success)

        return red_action, target_node

    def get_number_of_actions(self):
        return len(self.action_dict)

    def select_action(self):
        """
        Uses np.random.choice to return action from action set with weight equal to action probabilities
        """
        action = np.random.choice(a=self.action_set, p=self.action_probabilities)
        return action

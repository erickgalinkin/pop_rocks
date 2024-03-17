from yawning_titan.envs.generic.core.blue_interface import BlueActionSet
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.networks.node import Node

from typing import Tuple

import logging

logger = logging.getLogger(__name__)


class LazyBlue(BlueActionSet):
    def __init__(self, network_interface: NetworkInterface):
        super().__init__(network_interface)

    def perform_action(self, action: int) -> Tuple[str, Node]:
        """No matter what, do nothing."""
        blue_action, blue_node = self.do_nothing()
        return blue_action, blue_node

    @staticmethod
    def get_number_of_actions():
        return 1

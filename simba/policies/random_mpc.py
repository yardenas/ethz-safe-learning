import numpy as np

from simba.policies.policy import PolicyBase
from simba.infrastructure.logging_utils import logger


class RandomMpc(PolicyBase):
    def __init__(self,
                 action_space):
        super().__init__()
        self.action_space_sample = action_space.sample

    def generate_action(self, state):
        return self.action_space_sample()

    def build(self):
        pass

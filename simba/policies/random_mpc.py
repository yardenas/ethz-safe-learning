import numpy as np
from simba.policies.policy import PolicyBase
from simba.infrastructure.logging_utils import logger


class RandomMpc(PolicyBase):
    def __init__(self,
                 action_space):
        super().__init__()
        self.action_space = action_space

    def generate_action(self, state):
        return np.expand_dims(np.random.uniform(self.action_space.low, self.action_space.high), axis=0)

    def build(self):
        pass

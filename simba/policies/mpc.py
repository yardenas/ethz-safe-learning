from simba.policies.policy import PolicyBase
from simba.infrastructure.logging_utils import logger


class Mpc(PolicyBase):
    def __init__(self,
                 model,
                 reward,
                 cost,
                 horizon):
        super().__init__()
        self.model = model
        self.reward = reward
        self.cost = cost
        self.horizon = horizon
        # TODO (yarden): write a build optimizer function.
        self.optimizer = None
        pass

    def generate_action(self, state):
        logger.debug("Taking action.")
        pass

    def build(self):
        logger.debug("Building policy.")
        pass

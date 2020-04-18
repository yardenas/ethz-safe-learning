from simba.policies.policy import PolicyBase
from simba.infrastructure.logging_utils import logger


class CemMpc(PolicyBase):
    def __init__(self,
                 model,
                 environment,
                 horizon):
        super().__init__()
        # TODO (yarden): not sure if it should take the model just as input to a function.
        self.model = model
        self.reward = environment.get_rewards
        self.cost = None
        self.horizon = horizon
        pass

    def generate_action(self, state):
        logger.debug("Taking action.")
        # TODO (yarden): if env.is_done == True stop propagating (or at least add 0 to rewards...)
        return 0.0

    def build(self):
        logger.debug("Building policy.")
        pass

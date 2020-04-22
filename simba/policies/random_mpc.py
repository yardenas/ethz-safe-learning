from simba.policies.policy import PolicyBase
from simba.infrastructure.logging_utils import logger


class RandomMpc(PolicyBase):
    def __init__(self):
        super().__init__()
        pass

    def generate_action(self, state):
        logger.debug("Taking action.")
        # TODO (yarden): if env.is_done == True stop propagating (or at least add 0 to rewards...)
        return 0.0

    def build(self):
        logger.debug("Building policy.")
        pass

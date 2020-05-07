import numpy as np
from simba.environment_utils.mbrl_env import MbrlEnv


class MbrlInvertedPendulum(MbrlEnv):
    def __init__(self):
        super().__init__('InvertedPendulum-v2')

    # Copy-pasted from the InvertedPendulumEnv 'step' function.
    def get_reward(self, obs, *args, **kwargs):
        assert obs.ndim == 2, \
            "Expected inputs with shape (batch_size, dim), got shapes {}" .format(obs.shape)
        notdone = np.logical_and(np.isfinite(obs).all(axis=1), (np.abs(obs[:, 1]) <= .2))
        return np.ones(shape=(obs.shape[0],)), np.logical_not(notdone)

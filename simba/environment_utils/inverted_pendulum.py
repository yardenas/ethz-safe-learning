import numpy as np
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from simba.environment_utils.mbrl_env import MbrlEnv


class MbrlInvertedPendulumEnv(InvertedPendulumEnv, MbrlEnv):
    def __init__(self):
        super().__init__()

    # Copy-pasted from the InvertedPendulumEnv 'step' function.
    def get_reward(self, obs, acs):
        assert obs.ndim == 2 and acs.ndim == 2, \
            "Expected inputs with shape (batch_size, dim), got shapes {} and {}" .format(obs.shape, acs.shape)
        notdone = np.logical_and(np.isfinite(obs).all(axis=1), (np.abs(obs[:, 1]) <= .2))
        return np.ones(shape=(obs.shape[0],)), np.logical_not(notdone)

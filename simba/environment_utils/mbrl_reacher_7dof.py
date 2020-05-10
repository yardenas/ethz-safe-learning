import tensorflow as tf
from simba.environment_utils.mbrl_env import MbrlEnv


class MbrlReacher7DOF(MbrlEnv):
    def __init__(self):
        super().__init__('Reacher7DOF-v0')

    def get_reward(self, observations, actions, *args, **kwargs):
        hand_pos = observations[:, -6:-3]
        target_pos = observations[:, -3:]
        dist = tf.linalg.norm(hand_pos - target_pos, axis=1)
        dones = tf.zeros((observations.shape[0],), dtype=tf.bool)
        return -10.0 * dist, dones

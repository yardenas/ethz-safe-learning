import tensorflow as tf

from simba.environment_utils.mbrl_env import MbrlEnv


class MbrlInvertedPendulum(MbrlEnv):
    def __init__(self):
        super().__init__('InvertedPendulum-v2')

    # Copy-pasted from the InvertedPendulumEnv 'step' function.
    def get_reward(self, obs, *args, **kwargs):
        notdone = tf.logical_and(tf.reduce_all(tf.math.is_finite(obs), axis=1),
                                 tf.greater_equal(2.0, tf.math.abs(obs[:, 1])))
        return tf.ones(shape=(tf.shape(obs)[0],)), tf.logical_not(notdone)

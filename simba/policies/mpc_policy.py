import tensorflow as tf
from gym import spaces as spaces
from simba.policies.policy import PolicyBase
from simba.infrastructure.logging_utils import logger


class MpcPolicy(PolicyBase):
    def __init__(self,
                 model,
                 environment,
                 horizon,
                 n_samples,
                 particles):
        super().__init__()
        self.model = model
        self.reward = environment.get_reward
        self.action_space = environment.action_space
        assert isinstance(self.action_space, spaces.Box), "Expecting only box as action space."
        self.horizon = horizon
        self.n_samples = n_samples
        self.particles = particles

    def generate_action(self, state):
        raise NotImplementedError

    def compute_objective(self, trajectories, action_sequences):
        cumulative_rewards = tf.zeros((tf.shape(trajectories)[0],))
        done_trajectories = tf.zeros((tf.shape(trajectories)[0],), dtype=bool)
        horizon = trajectories.shape[1]
        for t in range(horizon - 1):
            s_t = trajectories[:, t, ...]
            s_t_1 = trajectories[:, t + 1, ...]
            a_t = action_sequences[:, t, ...]
            reward, dones = self.reward(s_t, a_t, s_t_1)
            cumulative_rewards += reward * (1.0 - tf.cast(done_trajectories, dtype=tf.float32))
            done_trajectories = tf.logical_or(
                dones, done_trajectories)
        rewards_per_sample = tf.reshape(cumulative_rewards, (self.particles, self.n_samples))
        return tf.reduce_mean(rewards_per_sample, axis=0)

    def build(self):
        logger.debug("Building policy.")
        pass

    @property
    def sampling_params(self):
        if self.action_space.is_bounded():
            mean = (self.action_space.high + self.action_space.low) / 2.0
            stddev = (self.action_space.high - self.action_space.low) / 2.0
            lower_bound = self.action_space.low
            upper_bound = self.action_space.high
        else:
            lower_bound = -100
            upper_bound = 100
            mean = 0.0
            stddev = 100
        return lower_bound, upper_bound, mean, stddev


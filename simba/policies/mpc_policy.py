import tensorflow as tf
from gym import spaces as spaces
from simba.policies.policy import PolicyBase
from simba.infrastructure.logging_utils import logger


class MpcPolicy(PolicyBase):
    def __init__(self,
                 model,
                 environment,
                 horizon,
                 objective,
                 n_samples,
                 particles):
        super().__init__()
        self.model = model
        self.reward = environment.get_reward
        self.action_space = environment.action_space
        assert isinstance(self.action_space, spaces.Box), "Expecting only box as action space."
        self.horizon = horizon
        self.objective = self.pets_objective
        self.n_samples = n_samples
        self.particles = particles

    def generate_action(self, state):
        raise NotImplementedError

    def compute_cumulative_rewards(self, trajectories, action_sequences):
        cumulative_rewards = tf.zeros((trajectories.shape[0],))
        done_trajectories = tf.zeros((trajectories.shape[0]), dtype=bool)
        horizon = trajectories.shape[1]
        for t in range(horizon - 1):
            s_t = trajectories[:, t, ...]
            s_t_1 = trajectories[:, t + 1, ...]
            a_t = action_sequences[:, t, ...]
            reward, dones = self.reward(s_t, a_t, s_t_1)
            done_trajectories = tf.logical_or(
                dones, done_trajectories)
            cumulative_rewards += reward * (1.0 - tf.cast(done_trajectories, dtype=tf.float32))
        return cumulative_rewards

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
            # For large enough bounds, the truncated normal dist. converges to a standard normal dist.
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html#scipy.stats.truncnorm
            lower_bound = -100
            upper_bound = 100
            mean = 0.0
            stddev = 100
        return lower_bound, upper_bound, mean, stddev

    def pets_objective(self, cumulative_rewards):
        rewards_per_sample = tf.reshape(cumulative_rewards, (self.particles, self.n_samples))
        return tf.reduce_mean(rewards_per_sample, axis=0)

    def pets_with_exploration_bonus(self, cumulative_rewards):
        rewards_per_sample_per_net = cumulative_rewards.reshape((5, -1, self.n_samples))
        particle_mean = rewards_per_sample_per_net.mean(axis=1)
        epistemic = particle_mean.std(axis=0)
        return self.pets_objective(cumulative_rewards) + epistemic

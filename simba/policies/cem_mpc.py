import numpy as np
import tensorflow as tf
from scipy.stats import truncnorm
from gym import spaces as spaces
from simba.policies.policy import PolicyBase
from simba.infrastructure.logging_utils import logger


class CemMpc(PolicyBase):
    def __init__(self,
                 model,
                 environment,
                 horizon,
                 iterations,
                 objective,
                 smoothing,
                 n_samples,
                 n_elite,
                 particles):
        super().__init__()
        self.model = model
        self.reward = environment.get_reward
        self.action_space = environment.action_space
        assert isinstance(self.action_space, spaces.Box), "Expecting only box as action space."
        self.horizon = horizon
        self.iterations = iterations
        self.objective = self.pets_objective
        self.smoothing = smoothing
        self.n_samples = n_samples
        self.elite = n_elite
        self.particles = particles

    def generate_action(self, state):
        return self.do_generate_action(tf.constant(state, dtype=tf.float32)).numpy()

    @tf.function
    def do_generate_action(self, state):
        lb, ub, mu, sigma = self.sampling_params
        action_dim = self.action_space.shape[0]
        mu = tf.broadcast_to(mu, (self.horizon, action_dim))
        sigma = tf.broadcast_to(sigma, (self.horizon, action_dim))
        for i in tf.range(self.iterations):
            action_sequences = tf.random.normal(
                shape=(self.n_samples, self.horizon, action_dim),
                mean=mu, stddev=sigma
            )
            action_sequences = tf.clip_by_value(action_sequences, lb, ub)
            action_sequences_batch = tf.tile(
                action_sequences, (self.particles, 1, 1)
            )
            trajectories = self.model.unfold_sequences(
                tf.broadcast_to(state, (action_sequences_batch.shape[0], state.shape[0])), action_sequences_batch
            )
            cumulative_rewards = self.compute_cumulative_rewards(trajectories, action_sequences_batch)
            scores = self.objective(cumulative_rewards)
            _, elite = tf.nn.top_k(scores, self.elite, sorted=False)
            best_actions = tf.gather(action_sequences, elite, axis=0)
            mean, variance = tf.nn.moments(best_actions, axes=0)
            stddev = tf.sqrt(variance)
            mu = self.smoothing * mu + (1.0 - self.smoothing) * mean
            sigma = self.smoothing * sigma + (1.0 - self.smoothing) * stddev
            if tf.less_equal(tf.reduce_mean(sigma), 0.25):
                break
        return mu[0, ...]


    # def generate_action(self, state):
    #     lb, ub, mu, sigma = self.sampling_params
    #     action_sequences = np.random.uniform(lb, ub, (self.n_samples, self.horizon, self.action_space.shape[0]))
    #     action_sequences_batches = np.tile(action_sequences, (self.particles, 1, 1))
    #     trajectories = self.model.simulate_trajectories(
    #         np.broadcast_to(state, (action_sequences_batches.shape[0], state.shape[0])), action_sequences_batches
    #     )
    #     cumulative_rewards = self.compute_cumulative_rewards(trajectories, action_sequences_batches)
    #     scores = self.objective(cumulative_rewards)
    #     best_trajectory_id = np.argmax(scores)
    #     return action_sequences[best_trajectory_id, 0, :]

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

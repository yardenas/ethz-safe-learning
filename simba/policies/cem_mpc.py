import numpy as np
import tensorflow.compat.v1 as tf
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
        self.reward = environment.get_rewards
        self.is_done = environment.is_done
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
        lb, ub, mu, sigma = self.sampling_params
        elite = mu
        for i in range(self.iterations):
            action_sequences = truncnorm.rvs(
                a=lb, b=ub, loc=mu, scale=sigma,
                size=(self.n_samples, self.horizon, self.action_space.shape[0])
            )
            # Propagate the same action sequences for #particles to get better statistics estimates.
            action_sequences_batch = np.tile(action_sequences, (self.particles, 1, 1))
            trajectories = self.model.simulate_trajectories(
                np.broadcast_to(state, (action_sequences_batch.shape[0], state.shape[0])), action_sequences_batch)
            rewards_along_trajectories = self.compute_rewards_along_trajectories(trajectories, action_sequences_batch)
            scores = self.objective(rewards_along_trajectories)
            trajectories_ranking = np.argsort(scores)
            elite = action_sequences[trajectories_ranking[-self.elite:], ...]
            elite_mu, elite_sigma = elite.mean(axis=0), elite.std(axis=0)
            mu = self.smoothing * mu + (1.0 - self.smoothing) * elite_mu
            sigma = self.smoothing * sigma + (1.0 - self.smoothing) * elite_sigma
            if np.max(sigma) < 1e-1:
                break
        return elite[0, 0, ...]

    def compute_rewards_along_trajectories(self, trajectories, action_sequences):
        rewards = np.zeros((trajectories.shape[0], trajectories.shape[1]))
        done_trajectories = np.zeros((trajectories.shape[0]), dtype=bool)
        for t in range(self.horizon):
            s_t = trajectories[:, t, ...]
            a_t = action_sequences[:, t, ...]
            done_trajectories = np.logical_or(
                self.is_done(s_t, a_t), done_trajectories)
            rewards[:, t, ...] = self.reward(s_t, a_t) * (1 - done_trajectories)
        return rewards

    def build(self):
        logger.debug("Building policy.")
        pass

    @property
    def sampling_params(self):
        if self.action_space.is_bounded():
            mean = (self.action_space.high + self.action_space.low) / 2.0
            stddev = self.action_space.high - self.action_space.low
            lower_bound = (self.action_space.low - mean) / stddev
            upper_bound = (self.action_space.high - mean) / stddev
        else:
            # For large enough bounds, the truncated normal dist. converges to a standard normal dist.
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html#scipy.stats.truncnorm
            lower_bound = -100
            upper_bound = 100
            mean = 0.0
            stddev = 1.0
        return lower_bound, upper_bound, mean, stddev

    def pets_objective(self, rewards_along_trajectories):
        rewards_per_sample = rewards_along_trajectories.reshape((self.n_samples, self.particles, self.horizon))
        return np.mean(rewards_per_sample.sum(axis=2), axis=1)

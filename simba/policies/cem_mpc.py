import numpy as np
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
        lb, ub, mu, sigma = self.sampling_params
        elite = mu
        for i in range(self.iterations):
            # Following instructions from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm
            # .html
            action_sequences = truncnorm.rvs(
                a=(lb - mu) / sigma, b=(ub - mu) / sigma, loc=mu, scale=sigma,
                size=(self.n_samples, self.horizon, self.action_space.shape[0])
            )
            # Propagate the same action sequences for #particles to get better statistics estimates.
            action_sequences_batch = np.tile(action_sequences, (self.particles, 1, 1))
            trajectories = self.model.simulate_trajectories(
                np.broadcast_to(state, (action_sequences_batch.shape[0], state.shape[0])), action_sequences_batch)
            assert np.isfinite(trajectories).all(), "Got a non-finite trajectory."
            cumulative_rewards = self.compute_cumulative_rewards(trajectories, action_sequences_batch)
            assert np.isfinite(cumulative_rewards.all()), "Got non-finite rewards."
            scores = self.objective(cumulative_rewards)
            elite = action_sequences[np.argsort(scores)[-self.elite:], ...]
            elite_mu, elite_sigma = elite.mean(axis=0), elite.std(axis=0)
            mu = self.smoothing * mu + (1.0 - self.smoothing) * elite_mu
            sigma = self.smoothing * sigma + (1.0 - self.smoothing) * elite_sigma
            if np.max(sigma) < 1e-1:
                break
        return elite[0, 0, ...]

    def compute_cumulative_rewards(self, trajectories, action_sequences):
        cumulative_rewards = np.zeros((trajectories.shape[0],))
        done_trajectories = np.zeros((trajectories.shape[0]), dtype=bool)
        for t in range(self.horizon - 1):
            s_t = trajectories[:, t, ...]
            s_t_1 = trajectories[:, t + 1, ...]
            a_t = action_sequences[:, t, ...]
            reward, dones = self.reward(s_t, a_t, s_t_1)
            done_trajectories = np.logical_or(
                dones, done_trajectories)
            cumulative_rewards += reward * (1 - done_trajectories)
        return cumulative_rewards

    def build(self):
        logger.debug("Building policy.")
        pass

    @property
    def sampling_params(self):
        if self.action_space.is_bounded():
            mean = (self.action_space.high + self.action_space.low) / 2.0
            stddev = self.action_space.high - self.action_space.low
            lower_bound = self.action_space.low
            upper_bound = self.action_space.high
        else:
            # For large enough bounds, the truncated normal dist. converges to a standard normal dist.
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html#scipy.stats.truncnorm
            lower_bound = -100
            upper_bound = 100
            mean = 0.0
            stddev = 200
        return lower_bound, upper_bound, mean, stddev

    def pets_objective(self, cumulative_rewards):
        rewards_per_sample = cumulative_rewards.reshape((self.particles, self.n_samples))
        return np.mean(rewards_per_sample, axis=0)

    def pets_with_exploration_bonus(self, cumulative_rewards):
        rewards_per_sample_per_net = cumulative_rewards.reshape((5, -1, self.n_samples))
        particle_mean = rewards_per_sample_per_net.mean(axis=1)
        epistemic = particle_mean.std(axis=0)
        return self.pets_objective(cumulative_rewards) + epistemic

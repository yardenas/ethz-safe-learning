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
                 n_elite):
        super().__init__()
        self.model = model
        self.reward = environment.get_rewards
        self.is_done = environment.is_done
        self.action_space = environment.action_space
        assert isinstance(self.action_space, spaces.Box), "Expecting only box as action space."
        self.horizon = horizon
        self.iterations = iterations
        self.objective = lambda t: np.mean(np.sum(t, axis=0), axis=(1, 2, 3))
        self.smoothing = smoothing
        self.n_samples = n_samples
        self.elite = n_elite

    def generate_action(self, state):
        # TODO (yarden): if env.is_done == True stop propagating (or at least add 0 to rewards...)
        lb, ub, mu, sigma = self.sampling_params
        for i in range(self.iterations):
            action_sequences = truncnorm.rvs(
                a=lb, b=ub, loc=mu, scale=sigma,
                size=(self.n_samples, self.horizon, self.action_space.shape[0])
            )
            rewards_along_trajectories = self.simulate_trajectories(state, action_sequences)
            trajectories_scores = np.argsort(self.objective(rewards_along_trajectories))
            elite = action_sequences[trajectories_scores[-self.elite:], ...]
            elite_mu, elite_sigma = elite.mean(axis=0), elite.std(axis=0)
            mu = self.smoothing * mu + (1.0 - self.smoothing) * elite_mu
            sigma = self.smoothing * sigma + (1.0 - self.smoothing) * elite_sigma
            if np.max(sigma) < 1:
                break
        return mu[0]

    def build(self):
        logger.debug("Building policy.")
        pass

    def simulate_trajectories(self, current_state, action_sequences):
        particles = 20
        samples = 1
        # TODO (yarden): not sure about this copy.
        s_t = np.broadcast_to(current_state.copy(),
                              (particles * self.n_samples, current_state.shape[0]))
        action_batches = np.broadcast_to(action_sequences,
                                         (particles, self.n_samples, self.horizon, self.action_space.shape[0]))
        action_batches = np.reshape(action_batches,
                                    (particles * self.n_samples, self.horizon, self.action_space.shape[0]))
        rewards = []
        done_trajectories = np.zeros((particles, self.n_samples), dtype=bool)
        for t in range(self.horizon - 1):
            a_t = action_batches[:, t, ...]
            # If a trajectory was already predicted to be over in previous timesteps, it should remain done.
            done_trajectories = np.logical_and(
                np.reshape(self.is_done(s_t, a_t), (particles, self.n_samples)), done_trajectories)
            # s_t_1_samples is of shape: (samples, n_mlps ,particles_per_mlp, observation_space_dim)
            _, _, s_t_1_samples = \
                self.model.predict(np.concatenate([s_t, a_t], axis=1), samples=samples, distribute=True)
            # Predict outcomes of future states conditioned on a_t_1.
            s_t_1_samples_batches = np.reshape(s_t_1_samples, (-1, current_state.shape[0]))
            a_t_1 = action_batches[:, t + 1, ...]
            rewards_batch = self.reward(s_t_1_samples_batches, a_t_1)
            reward_per_particle = np.reshape(rewards_batch, (-1, particles, self.n_samples))
            # Arrange rewards in shape: (n_action_seqs, n_mlps, particles_per_mlp, samples_per_particle)
            # 'dead' particles recieve 0 reward.
            rewards.append(np.reshape(reward_per_particle * (1 - done_trajectories),
                                      (self.n_samples, s_t_1_samples.shape[1], -1, samples)))
            # TODO (yarden): decide if we propagate the mean or a random sample (I think a random sample.)
            random_state = np.random.choice(samples)
            s_t = np.reshape(s_t_1_samples[random_state, ...], (-1, current_state.shape[0]))
        return np.array(rewards, copy=False)


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

import numpy as np
import tensorflow as tf

from simba.policies.cem_mpc import CemMpc


class SafeCemMpc(CemMpc):
    def __init__(self,
                 model,
                 environment,
                 horizon,
                 iterations,
                 smoothing,
                 n_samples,
                 n_elite,
                 particles,
                 stddev_threshold,
                 noise_stddev,
                 posterior_mean_threashold):
        super().__init__(
            model,
            environment,
            horizon,
            iterations,
            smoothing,
            n_samples,
            n_elite,
            particles,
            stddev_threshold,
            noise_stddev
        )
        self.cost = environment.get_cost
        self.posterior_mean_threashold = posterior_mean_threashold
        self.last_action = tf.zeros((self.action_space.shape[0],), dtype=tf.float32)

    def generate_action(self, state):
        action, _ = self.do_generate_action(tf.constant(state, dtype=tf.float32))
        return action.numpy()

    @tf.function
    def optimize_for_safety(self, state):
        lb, ub, mu, sigma = self.sampling_params
        action_dim = self.action_space.shape[0]
        mu = tf.broadcast_to(mu, (self.horizon, action_dim))
        sigma = tf.broadcast_to(sigma, (self.horizon, action_dim))
        best_so_far = tf.zeros((action_dim,), dtype=tf.float32)
        best_so_far_score = -np.inf * tf.ones((), dtype=tf.float32)
        for _ in tf.range(self.iterations):
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
            mean_costs = self.compute_mean_costs(trajectories, action_sequences_batch)
            scores = -mean_costs
            elite_scores, elite = tf.nn.top_k(scores, self.elite, sorted=False)
            best_of_elite = tf.argmax(elite_scores)
            if tf.greater(elite_scores[best_of_elite], best_so_far_score):
                best_so_far = action_sequences[elite[best_of_elite], 0, ...]
                best_so_far_score = elite_scores[best_of_elite]
            elite_actions = tf.gather(action_sequences, elite, axis=0)
            mean, variance = tf.nn.moments(elite_actions, axes=0)
            stddev = tf.sqrt(variance)
            mu = self.smoothing * mu + (1.0 - self.smoothing) * mean
            sigma = self.smoothing * sigma + (1.0 - self.smoothing) * stddev
            if tf.less_equal(tf.reduce_mean(sigma), self.stddev_threshold):
                break
        return best_so_far + tf.random.normal(best_so_far.shape, stddev=self.noise_stddev)

    def compute_objective(self, trajectories, action_sequences):
        cumulative_rewards = tf.zeros((self.n_samples * self.particles,), dtype=tf.float32)
        done_trajectories = tf.zeros((self.n_samples * self.particles,), dtype=tf.bool)
        safe_trajectories = tf.ones((self.n_samples,), dtype=tf.bool)
        horizon = trajectories.shape[1]
        mu, sigma = tf.linspace(0.5, 0.5, horizon - 1), tf.linspace(0.25, 0.25, horizon - 1)
        for t in range(horizon - 1):
            s_t = trajectories[:, t, ...]
            s_t_1 = trajectories[:, t + 1, ...]
            a_t = action_sequences[:, t, ...]
            reward, dones = self.reward(s_t, a_t, s_t_1)
            done_trajectories = tf.logical_or(
                dones, done_trajectories)
            cost = self.cost(s_t, a_t, s_t_1) * (1.0 - tf.cast(done_trajectories, dtype=tf.float32))
            probably_safe = self.bayesian_safety_beta_inference(cost, mu[t], sigma[t])
            safe_trajectories = tf.logical_and(
                probably_safe, safe_trajectories)
            cumulative_rewards += reward * (1.0 - tf.cast(done_trajectories, dtype=tf.float32))
        rewards_per_sample = tf.reshape(cumulative_rewards, (self.particles, self.n_samples))
        trajectories_returns = tf.reduce_mean(rewards_per_sample, axis=0)
        return trajectories_returns - tf.cast(tf.logical_not(safe_trajectories), tf.float32) * 100.0

    def compute_mean_costs(self, trajectories, action_sequences):
        cumulative_costs = tf.zeros((tf.shape(trajectories)[0],))
        horizon = trajectories.shape[1]
        for t in range(horizon - 1):
            s_t = trajectories[:, t, ...]
            s_t_1 = trajectories[:, t + 1, ...]
            a_t = action_sequences[:, t, ...]
            cost = self.cost(s_t, a_t, s_t_1)
            cumulative_costs += cost
        costs_per_sample = tf.reshape(cumulative_costs, (self.particles, self.n_samples))
        return tf.reduce_mean(costs_per_sample, axis=0)

    def bayesian_safety_beta_inference(self, costs, mu=0.5, sigma=0.20):
        costs_per_sample = tf.reshape(costs, (self.particles, self.n_samples))
        counts = tf.reduce_sum(costs_per_sample, axis=0)
        # Computing parameters for the prior.
        alpha = (((1.0 - mu) / sigma ** 2) - 1.0 / mu) * (mu ** 2)
        beta = alpha * (1.0 / mu - 1)
        posterior_mean = (alpha + counts) / (alpha + beta + self.particles)
        # From a Bayesian decision theory perspective, the posterior_mean_threshold is
        # c_FP / (c_FP + c_FN) where c_FP, c_FN are the false-positive and false-negative
        # error costs.
        return tf.less_equal(posterior_mean, self.posterior_mean_threashold)

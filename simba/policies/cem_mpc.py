import numpy as np
import tensorflow as tf
from simba.policies.mpc_policy import MpcPolicy
from simba.infrastructure.logging_utils import logger


class CemMpc(MpcPolicy):
    def __init__(self,
                 model,
                 environment,
                 horizon,
                 iterations,
                 objective,
                 smoothing,
                 n_samples,
                 n_elite,
                 particles,
                 stddev_threshold,
                 noise_stddev):
        super().__init__(
            model,
            environment,
            horizon,
            objective,
            n_samples,
            particles
        )
        self.iterations = iterations
        self.smoothing = smoothing
        self.elite = n_elite
        self.stddev_threshold = stddev_threshold
        self.noise_stddev = noise_stddev

    def generate_action(self, state):
        return self.do_generate_action(tf.constant(state, dtype=tf.float32)).numpy()

    @tf.function
    def do_generate_action(self, state):
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
            wtf_flag = tf.reduce_all(tf.greater(trajectories, 1e3))
            if wtf_flag:
                tf.print("wtffffffffff")
            cumulative_rewards = self.compute_cumulative_rewards(trajectories, action_sequences_batch)
            scores = self.objective(cumulative_rewards)
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

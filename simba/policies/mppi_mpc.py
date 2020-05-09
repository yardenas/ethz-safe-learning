import tensorflow as tf
from simba.policies.mpc_policy import MpcPolicy
from simba.infrastructure.logging_utils import logger


class CemMpc(MpcPolicy):
    def __init__(self,
                 model,
                 environment,
                 horizon,
                 objective,
                 n_samples,
                 particles):
        super().__init__(
            model,
            environment,
            horizon,
            objective,
            n_samples,
            particles
        )

    def generate_action(self, state):
        return self.do_generate_action(tf.constant(state, dtype=tf.float32)).numpy()

    @tf.function
    def do_generate_action(self, state):
        lb, ub, mu, sigma = self.sampling_params
        action_dim = self.action_space.shape[0]
        mu = tf.broadcast_to(mu, (self.horizon, action_dim))
        sigma = tf.broadcast_to(sigma, (self.horizon, action_dim))
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
        returns = self.objective(cumulative_rewards)
        scores = tf.exp(5.0 * (returns - tf.reduce_max(returns, axis=0)))
        weighted_action_sequences = tf.expand_dims(scores, axis=1) * action_sequences[:, 0, ...]
        return tf.reduce_sum(weighted_action_sequences, axis=0) / tf.reduce_sum(scores + 1e-5, axis=0)



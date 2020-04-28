import tensorflow.compat.v1 as tf
import numpy as np

from simba.infrastructure.common import standardize_name
from simba.models import TensorFlowBaseModel, AnchoredMlpEnsemble, MlpEnsemble


class TransitionModel(TensorFlowBaseModel):
    def __init__(self,
                 sess,
                 model,
                 observation_space_dim,
                 action_space_dim,
                 **kwargs):
        super().__init__(sess,
                         observation_space_dim + action_space_dim,
                         observation_space_dim)
        self.model_scope = model
        self.model = eval(standardize_name(model))(
            self._sess,
            self.model_scope,
            self.inputs_dim,
            self.outputs_dim,
            **kwargs)
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim
        self.inputs_mean = tf.zeros((self.inputs_dim,), name='inputs_mean')
        self.inputs_stddev = tf.ones((self.inputs_dim,), name='inputs_stddev')
        self.action_seqs_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, None, self.action_space_dim),
            name='action_sequences_ph'
        )
        self.observations_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self.observation_space_dim),
            name='observations_ph'
        )
        self.targets_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self.outputs_dim),
            name='targets_ph'
        )
        self.propagated_trajectories = None

    def build(self):
        self.model.build()
        self.propagated_trajectories = self.propagation_op(self.observations_ph, self.action_seqs_ph)

    def fit(self, inputs, targets):
        self.inputs_mean = inputs.mean(axis=0)
        self.inputs_stddev = inputs.std(axis=0)
        self.model.fit(
            (inputs - self.inputs_mean) / (self.inputs_stddev + 1e-8),
            targets)

    def predict(self, inputs):
        return self.simulate_trajectories(
            current_state=inputs[..., :self.observation_space_dim],
            action_sequences=np.expand_dims(inputs[..., -self.action_space_dim:], axis=1)
        )

    def predict_ops(self, inputs):
        return self.model.predict_ops(
            (inputs - self.inputs_mean) / (self.inputs_stddev + 1e-7))

    def simulate_trajectories(self, current_state, action_sequences):
        return self._sess.run(
            self.propagated_trajectories,
            feed_dict={
                self.action_seqs_ph: action_sequences,
                self.observations_ph: current_state
            }
        )

    def propagation_op(self, s_t, action_sequences):
        horizon = 1
        trajectories = tf.expand_dims(s_t, axis=1, name='trajectories')

        def per_timestep(s_t_a_t, t, trajectories):
            _, _, s_t_1_cond_s_t_a_t = self.predict_ops(s_t_a_t)
            a_t_1 = action_sequences[:, t + 1, ...]
            return tf.concat([s_t_1_cond_s_t_a_t, a_t_1], axis=1), (t + 1), \
                   tf.concat([trajectories, tf.expand_dims(s_t_1_cond_s_t_a_t, axis=1)], axis=1)

        s_0_a_0 = tf.concat([s_t, action_sequences[:, 0, ...]], axis=1)
        t_0 = tf.constant(0, dtype=tf.int32)
        tf.TensorShape([None, None, self.outputs_dim])
        tf.while_loop(
            cond=lambda _, t, *args: tf.less(t, horizon),
            body=per_timestep,
            loop_vars=[s_0_a_0, t_0, trajectories],
            shape_invariants=[s_0_a_0.get_shape(),
                              t_0.get_shape(),
                              tf.TensorShape([None, None, self.outputs_dim])],
            back_prop=False
        )

        return trajectories

    def save(self):
        pass

    def load(self):
        pass

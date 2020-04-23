import tensorflow.compat.v1 as tf
import numpy as np

from simba.infrastructure.common import standardize_name
from simba.models import TensorFlowBaseModel, AnchoredMlpEnsemble


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
        self.model = eval(standardize_name(model))(
            self._sess,
            self.inputs_dim,
            self.outputs_dim,
            **kwargs)
        self.observation_space_dim = observation_space_dim
        self.inputs_mean = None
        self.inputs_stddev = None

    def build(self):
        self.model.build()

    def fit(self, inputs, targets):
        self.inputs_mean = inputs.mean(axis=0)
        self.inputs_stddev = inputs.std(axis=0)
        # Our model predicts s_(t + 1) - s_(t)
        losses = self.model.fit(
            (inputs - self.inputs_mean) / (self.inputs_stddev + 1e-8),
            targets - inputs[:, :self.observation_space_dim])

    def predict(self, inputs, *args, **kwargs):
        mus, sigmas, samples = self.model.predict(
            (inputs - self.inputs_mean) / (self.inputs_stddev + 1e-8), *args, **kwargs)
        # Our model predicts the difference s_(t + 1) - s_(t) hence we add s_(t) to the mean
        # and samples of the predictions.
        # TODO (yarden): what about the standard deviation in this case????
        mus = mus + np.reshape(inputs[:, :self.observation_space_dim], mus.shape)
        samples = samples + np.reshape(inputs[:, :self.observation_space_dim], mus.shape)
        return mus, sigmas, samples

    def simulate_trajectories(self, current_state, action_sequences):
        predict_op = self.model.predict_ops
        particles = 20
        def per_timestep(s_t_a_t, _):
            s_t, a_t = s_t_a_t
            s_t_1 = predict_op(tf.concat([s_t, a_t], axis=1))

    def save(self):
        pass

    def load(self):
        pass

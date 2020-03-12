import tensorflow as tf
from simba.models import BaseModel
from simba.infrastructure.ensembled_anchored_nn import build_mlp


class MBRLModel(BaseModel):
    def __init(self,
               action_space_dimension,
               observation_space_dimension,
               approximator,
               **kwargs):
        # TODO (yarden): don't forget to make the predictions the mean and variance of a normal distribution.
        super().__init__(**kwargs)
        self.observations_ph = tf.placeholder(
            shape=[None, observation_space_dimension],
            name='observations_ph', dtype=tf.float32
        )
        self.actions_ph = tf.placeholder(
            shape=[None, action_space_dimension],
            name='actions_ph', dtype=tf.float32
        )
        self.next_observations_ph = tf.placeholder(
            shape=[None, observation_space_dimension],
            name='next_actions_ph', dtype=tf.float32
        )
        self.input_mean_ph = tf.placeholder(
            shape=[None, action_space_dimension + observation_space_dimension],
            name='input_mean', dtype=tf.float32
        )
        self.input_std_ph = tf.placeholder(
            shape=[None, action_space_dimension + observation_space_dimension],
            name='input_std', dtype=tf.float32
        )

    def build(self):
        pass

    def define_prediction_op(self):
        inputs = tf.concat([self.observations_ph, self.actions_ph], axis=1)
        assert tf.shape(inputs) == tf.shape(self.input_mean_ph) == tf.shape(self.input_std_ph)
        standardized_inputs = tf.divide(
            (inputs - self.input_mean_ph), self.input_std_ph + 1e-8)
        self.prediction_op = build_mlp(standardized_inputs, self.out)


    def define_traininig_op(self):
        labels = self.next_observations_ph - self.observations_ph
        assert self.prediction_op is not None
        self.loss = tf.losses.mean_squared_error(
            labels=labels, predictions=self.prediction_op)




    def fit(self, fit_feed_dict):
        pass

    def predict(self, prediction_feed_dict):
        pass

    def save(self):
        pass

    def load(self):
        pass


import tensorflow as tf
from simba.models import BaseModel
from simba.infrastructure.ensembled_anchored_nn import InitializationAnchoredNN


class MBRLModel(BaseModel):
    def __init(self,
               action_space_dimension,
               observation_space_dimension,
               ensemble_size,
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

    def fit(self, fit_feed_dict):
        pass

    def predict(self, prediction_feed_dict):
        pass

    def save(self):
        pass

    def load(self):
        pass

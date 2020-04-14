import numpy as np
import tensorflow as tf


class BaseLayer(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 activation,
                 dropout_rate):
        super().__init__()
        self._dense = tf.keras.layers.Dense(units=units)
        self._batch_norm = tf.keras.layers.BatchNormalization()
        self._dropout = tf.keras.layers.Dropout(dropout_rate)
        self._activation = activation

    def call(self, inputs, training=None):
        x = self._dense(inputs)
        x = self._batch_norm(x, training=training)
        x = self._activation(x)
        return self._dropout(x, training=training)


class GaussianHead(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self._mu = tf.keras.layers.Dense(1)
        self._var = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        return tf.concat((self._mu(inputs),
                         tf.math.softplus(self._var(inputs)) + 1e-4), axis=1)


def gaussian_dist_mlp(inputs_dim,
                      name,
                      n_layers,
                      units,
                      activation,
                      dropout_rate):
    inputs = tf.keras.Input(shape=(inputs_dim,))
    x = tf.keras.Sequential([
        BaseLayer(units, activation, dropout_rate) for _ in range(n_layers)
    ])(inputs)
    outputs = GaussianHead()(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


@tf.function
def negative_log_likelihood(y_true, y_pred):
    prediction_dim = y_pred.shape[1] // 2
    mu, var = y_pred[..., :prediction_dim], y_pred[..., prediction_dim:]
    return 0.5 * tf.reduce_mean(tf.math.log(2.0 * np.pi * var)) + \
           0.5 * tf.reduce_mean(tf.math.divide(tf.math.squared_difference(y_true, mu), var))


class MlpEnsemble(object):
    def __init__(self,
                 inputs_dim,
                 outputs_dim,
                 ensemble_size,
                 n_epochs,
                 batch_size,
                 validation_split,
                 learning_rate,
                 mlp_params):
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim
        self.ensemble_size = ensemble_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.mlp_params = mlp_params
        self.ensemble = None

    def build(self):
        inputs = [tf.keras.Input(shape=(self.inputs_dim,)) for _ in range(self.ensemble_size)]
        outputs = [gaussian_dist_mlp(name='ensemble/id_' + str(i), **self.mlp_params, inputs_dim=self.inputs_dim)
                   (inputs[i]) for i in range(self.ensemble_size)]
        self.ensemble = tf.keras.Model(inputs=inputs, outputs=outputs, name='ensemble')
        self.ensemble.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss=[negative_log_likelihood for _ in range(self.ensemble_size)]
        )
        self.ensemble.summary()

    def fit(self, inputs, targets):
        assert inputs.shape[0] == targets.shape[0], "Inputs batch size ({}) "
        "doesn't match targets batch size ({})".format(inputs.shape[0], targets.shape[0])
        inputs_per_mlp = [None] * self.ensemble_size
        targets_per_mlp = [None] * self.ensemble_size
        for i in range(self.ensemble_size):
            shuffle_per_mlp = np.random.permutation(inputs.shape[0])
            inputs_per_mlp[i] = inputs[shuffle_per_mlp]
            targets_per_mlp[i] = targets[shuffle_per_mlp]
        self.ensemble.fit(
            inputs_per_mlp,
            targets_per_mlp,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            validation_split=self.validation_split
        )

    def predict(self, inputs):
        broadcasted = [inputs.astype(np.float32)] * self.ensemble_size
        preds = self.ensemble.predict(broadcasted)
        preds = np.array(preds, copy=False)
        mus = preds[..., :self.outputs_dim]
        sigmas = np.sqrt(preds[..., self.outputs_dim:])
        return mus, sigmas, np.random.normal(mus, sigmas)


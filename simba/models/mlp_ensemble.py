import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


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

    @tf.function
    def call(self, inputs, training=None):
        x = self._dense(inputs)
        # x = self._batch_norm(x, training=training)
        x = self._activation(x)
        return self._dropout(x, training=training)


class GaussianHead(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super().__init__()
        self._mu = tf.keras.layers.Dense(output_dim)
        self._var = tf.keras.layers.Dense(output_dim,
                                          activation=lambda t: tf.math.softplus(t) + 1e-4)

    @tf.function
    def call(self, inputs, training=None):
        return self._mu(inputs), self._var(inputs)


class GaussianDistMlp(tf.keras.Model):
    def __init__(self,
                 outputs_dim,
                 n_layers,
                 units,
                 activation,
                 dropout_rate):
        super().__init__()
        self.forward = tf.keras.Sequential([
            BaseLayer(units, activation, dropout_rate) for _ in range(n_layers)
        ])
        self.head = GaussianHead(outputs_dim)

    @tf.function
    def call(self, inputs, training=None):
        x = self.forward(inputs, training)
        return self.head(x, training)


@tf.function
def negative_log_likelihood(y_true, mu, var):
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
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def build(self):
        self.ensemble = [GaussianDistMlp(outputs_dim=self.outputs_dim, **self.mlp_params)
                         for _ in range(self.ensemble_size)]

    @tf.function
    def forward(self, inputs):
        inputs_per_mlp = tf.split(inputs, self.ensemble_size, axis=0)
        ensemble_mus = []
        ensemble_vars = []
        for mlp_inputs, mlp in zip(inputs_per_mlp, self.ensemble):
            mu, var = mlp(mlp_inputs)
            ensemble_mus.append(mu)
            ensemble_vars.append(var)
        cat_mus = tf.concat(ensemble_mus, axis=0)
        cat_vars = tf.concat(ensemble_vars, axis=0)
        return cat_mus, cat_vars

    @tf.function
    def training_step(self, inputs, targets):
        losses = []
        for i, mlp in enumerate(self.ensemble):
            with tf.GradientTape() as tape:
                mu, var = mlp(inputs[i, ...])
                loss = negative_log_likelihood(targets[i, ...], mu, var)
                losses.append(loss)
                grads = tape.gradient(loss, mlp.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, mlp.trainable_variables))
        return losses

    def fit(self, inputs, targets):
        assert inputs.shape[0] == targets.shape[0], "Inputs batch size ({}) "
        "doesn't match targets batch size ({})".format(inputs.shape[0], targets.shape[0])
        losses = np.empty((self.n_epochs, self.ensemble_size))
        n_batches = int(np.ceil(inputs.shape[0] / self.batch_size))
        for epoch in range(self.n_epochs):
            avg_loss = 0.0
            shuffles_per_mlp = np.array([np.random.permutation(inputs.shape[0])
                                         for _ in range(self.ensemble_size)])
            x_batches = np.array_split(inputs[shuffles_per_mlp], n_batches, axis=1)
            y_batches = np.array_split(targets[shuffles_per_mlp], n_batches, axis=1)
            for x_batch, y_batch in zip(x_batches, y_batches):
                loss_per_mlp = self.training_step(x_batch, y_batch)
                avg_loss += np.array(loss_per_mlp) / n_batches
            if epoch % 20 == 0:
                print('Epoch {} | Losses {}'.format(epoch, avg_loss))
            losses[epoch] = avg_loss
        return losses

    def predict(self, inputs):
        mu, var = self.forward(inputs)
        distribution = tfp.distributions.Normal(loc=mu, scale=tf.sqrt(var))
        return distribution.mean(), distribution.stddev(), distribution.sample()

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from simba.infrastructure.logging_utils import logger


class BaseLayer(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 activation,
                 dropout_rate):
        super().__init__()
        self._dense = tf.keras.layers.Dense(units=units)
        self._activation = eval(activation) if isinstance(activation, str) else activation
        self._dropout = tf.keras.layers.Dropout(dropout_rate)

    @tf.function
    def call(self, inputs, training=None):
        x = self._dense(inputs)
        x = self._activation(x)
        x = self._dropout(x, training=training)
        return x


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
                 inputs_dim,
                 outputs_dim,
                 n_layers,
                 units,
                 activation,
                 dropout_rate):
        super().__init__()
        self.forward = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(inputs_dim,))] +
            [BaseLayer(units, activation, dropout_rate) for _ in range(n_layers)]
        )
        self.head = GaussianHead(outputs_dim)
        self.output_dim = outputs_dim

    # Following https://stackoverflow.com/questions/58577713/tf-function-with-input-signature-errors-out-when-calling
    # -a-sub-layer
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.bool)]
    )
    def call(self, inputs, training=None):
        x = self.forward(inputs, training)
        return self.head(x, training)


@tf.function
def negative_log_likelihood(y_true, mu, var):
    return 0.5 * tf.reduce_mean(tf.math.log(2.0 * np.pi * var)) + \
           0.5 * tf.reduce_mean(tf.math.divide(tf.square(mu - y_true), var))


class EpochLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                 init_learning_rate,
                 training_steps_per_epoch,
                 train_epochs):
        super().__init__()
        self._learning_rate = tf.constant(init_learning_rate, tf.float32)
        self._train_epochs = tf.constant(train_epochs, tf.float32)
        self._training_steps_per_epoch = tf.constant(training_steps_per_epoch, tf.int32)

    def __call__(self, step):
        train_epochs_so_far = tf.floor(tf.cast(step, tf.int32) / self._training_steps_per_epoch)
        return tf.maximum(self._learning_rate * (1.0 - tf.cast(train_epochs_so_far, tf.float32) / self._train_epochs),
                          0.0)

    def get_config(self):
        return dict(_learning_rate=self._learning_rate,
                    _train_epochs=self._train_epochs,
                    _training_steps_per_epoch=self._training_steps_per_epoch)


class MlpEnsemble(tf.Module):
    def __init__(self,
                 inputs_dim,
                 outputs_dim,
                 ensemble_size,
                 batch_size,
                 validation_split,
                 learning_rate,
                 learning_rate_schedule,
                 training_steps,
                 mlp_params,
                 train_epochs):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim
        self.ensemble_size = ensemble_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.training_steps = training_steps
        self.mlp_params = mlp_params
        self.ensemble = [GaussianDistMlp(inputs_dim=self.inputs_dim, outputs_dim=self.outputs_dim, **self.mlp_params)
                         for _ in range(self.ensemble_size)]
        self.optimizer = tf.keras.optimizers.Adam(
            EpochLearningRateSchedule(learning_rate, training_steps, train_epochs)
            if learning_rate_schedule else learning_rate,
            clipvalue=1.0,
            epsilon=1e-5)

    def build(self):
        pass

    def forward(self, inputs):
        inputs_per_mlp = tf.split(inputs, self.ensemble_size, axis=0)
        ensemble_mus = []
        ensemble_vars = []
        for mlp_inputs, mlp in zip(inputs_per_mlp, self.ensemble):
            mu, var = mlp(mlp_inputs, training=tf.constant(False))
            ensemble_mus.append(mu)
            ensemble_vars.append(var)
        cat_mus = tf.concat(ensemble_mus, axis=0)
        cat_vars = tf.concat(ensemble_vars, axis=0)
        return cat_mus, cat_vars

    @tf.function
    def training_step(self, inputs, targets):
        loss = 0.0
        ensemble_trainable_variables = []
        with tf.GradientTape() as tape:
            for i, mlp in enumerate(self.ensemble):
                mu, var = mlp(inputs[i, ...], training=tf.constant(True))
                loss += negative_log_likelihood(targets[i, ...], mu, var) / self.ensemble_size
                ensemble_trainable_variables += mlp.trainable_variables
            grads = tape.gradient(loss, ensemble_trainable_variables)
            self.optimizer.apply_gradients(zip(grads, ensemble_trainable_variables))
        return loss

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)] * 2
    )
    def validation_step(self, inputs, targets):
        loss = 0.0
        for i, mlp in enumerate(self.ensemble):
            mu, var = mlp(inputs, training=tf.constant(False))
            loss += negative_log_likelihood(targets, mu, var) / self.ensemble_size
        return loss

    def split_train_validate(self, inputs, targets):
        indices = np.random.permutation(inputs.shape[0])
        num_val = int(inputs.shape[0] * self.validation_split)
        train_idx, val_idx = indices[num_val:], indices[:num_val]
        return inputs[train_idx, ...], targets[train_idx, ...], inputs[val_idx, ...], targets[val_idx, ...]

    def fit(self, inputs, targets):
        assert inputs.shape[0] == targets.shape[0], "Inputs batch size ({}) "
        "doesn't match targets batch size ({})".format(inputs.shape[0], targets.shape[0])
        assert np.isfinite(inputs).all() and np.isfinite(targets).all(), "Training data is not finite."
        losses = np.empty((self.training_steps,))
        train_inputs, train_targets, validate_inputs, validate_targets = self.split_train_validate(inputs, targets)
        n_batches = int(np.ceil(train_inputs.shape[0] / self.batch_size))
        step = 0
        while step < self.training_steps:
            shuffles_per_mlp = np.array([np.random.permutation(train_inputs.shape[0])
                                         for _ in range(self.ensemble_size)])
            x_batches = np.array_split(train_inputs[shuffles_per_mlp], n_batches, axis=1)
            y_batches = np.array_split(train_targets[shuffles_per_mlp], n_batches, axis=1)
            for x_batch, y_batch in zip(x_batches, y_batches):
                loss = self.training_step(tf.constant(x_batch),
                                          tf.constant(y_batch))
                losses[step] = loss
                step += 1
                if step % int(self.training_steps / 10) == 0:
                    validation_loss = self.validation_step(validate_inputs, validate_targets).numpy()
                    logger.debug(
                        "Step {} | Training Loss {} | Validation Loss {}".format(step, loss, validation_loss))
                if step == self.training_steps:
                    break
        return losses

    @tf.function
    def __call__(self, inputs, *args, **kwargs):
        mu, var = self.forward(inputs)
        distribution = tfp.distributions.Normal(loc=mu, scale=tf.sqrt(var))
        return distribution.mean(), distribution.stddev(), distribution.sample()

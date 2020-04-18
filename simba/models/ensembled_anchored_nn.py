import numpy as np
import tensorflow.compat.v1 as tf
from simba.infrastructure.logging_utils import logger


class InitializationAnchoredNn(object):
    def __init__(self,
                 sess,
                 inputs,
                 targets,
                 scope,
                 learning_rate,
                 n_layers,
                 hidden_size,
                 activation,
                 anchor,
                 init_std_bias,
                 init_std_weights,
                 data_noise):
        self._sess = sess
        activation = eval(activation) if type(activation) is str else activation
        with tf.variable_scope(scope):
            layer = inputs
            self._layers = []
            lamda_anchors = []
            if anchor:
                bias_init_first_layer = \
                    tf.random_normal_initializer(0.0, init_std_bias)
                weights_init_first_layer = \
                    tf.random_normal_initializer(0.0, init_std_weights)
                bias_init_deep_layers = \
                    tf.random_normal_initializer(0.0, 1.0 / np.sqrt(hidden_size))
                weights_init_deep_layers = bias_init_deep_layers
                lamda_anchors.append((data_noise / init_std_weights ** 2,
                                      data_noise / init_std_bias ** 2))
            else:
                bias_init_first_layer, weights_init_first_layer,\
                    bias_init_deep_layers, weights_init_deep_layers = \
                    None, None, None, None
            self._layers.append(tf.layers.Dense(
                units=hidden_size,
                activation=activation,
                bias_initializer=bias_init_first_layer,
                kernel_initializer=weights_init_first_layer
            ))
            layer = self._layers[0].apply(layer)
            for _ in range(n_layers - 1):
                self._layers.append(tf.layers.Dense(
                    units=hidden_size,
                    activation=activation,
                    bias_initializer=bias_init_deep_layers,
                    kernel_initializer=weights_init_deep_layers
                ))
                layer = self._layers[-1].apply(layer)
                if anchor:
                    lamda_anchors.append((data_noise / hidden_size,
                                         data_noise / hidden_size))
            self._mu = tf.layers.dense(inputs=layer, units=1)
            var = tf.layers.dense(inputs=layer, units=1,
                                  activation=lambda t: tf.math.softplus(t) + 1e-4)
            self._sigma = tf.sqrt(var)
            self._loss = 0.5 * tf.reduce_mean(tf.log(var)) + 0.5 * tf.reduce_mean(
                tf.divide(tf.squared_difference(targets, self._mu), var)
            )
            if anchor:
                self._anchor_weights(lamda_anchors)
            self._training_op = \
                tf.train.AdamOptimizer(learning_rate).minimize(self._loss)

    def _anchor_weights(self, lamba_anchors):
        """
        Based on "Uncertainty in Neural Networks: Approximately Bayesian Ensembling"
        https://arxiv.org/abs/1810.05546
        """
        weights = []
        for layer in self._layers:
            weights += layer.weights
        init_weights = tf.variables_initializer(weights)
        self._sess.run(init_weights)
        for i in range(len(self._layers)):
            kernel = self._layers[i].kernel.eval().copy()
            bias = self._layers[i].bias.eval().copy()
            self._loss += \
                lamba_anchors[i][0] * tf.losses.mean_squared_error(
                    labels=kernel,
                    predictions=self._layers[i].kernel
                )
            self._loss += \
                lamba_anchors[i][1] * tf.losses.mean_squared_error(
                    labels=bias,
                    predictions=self._layers[i].bias
                )

    @property
    def loss(self):
        return self._loss

    @property
    def training_op(self):
        return self._training_op

    @property
    def predict_op(self):
        return self._mu, self._sigma


class MlpEnsemble(object):
    def __init__(self,
                 sess,
                 inputs_dim,
                 outputs_dim,
                 ensemble_size,
                 n_epochs,
                 batch_size,
                 mlp_params):
        self.sess = sess
        self.mlp_params = mlp_params
        self.ensemble_size = ensemble_size
        self.inputs_dim = inputs_dim
        self.epochs = n_epochs
        self.batch_size = batch_size
        self.inputs_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(ensemble_size, None, inputs_dim)
        )
        self.targets_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(ensemble_size, None, outputs_dim)
        )
        self.mlps = []
        self.predict_ops = []
        self.training_ops = []
        self.losses_ops = []

    def fit(self, inputs, targets):
        assert inputs.shape[0] == targets.shape[0], "Inputs batch size ({}) "
        "doesn't match targets batch size ({})".format(inputs.shape[0], targets.shape[0])
        losses = np.empty((self.epochs, self.ensemble_size))
        n_batches = int(np.ceil(inputs.shape[0] / self.batch_size))
        for epoch in range(self.epochs):
            avg_loss = 0.0
            shuffles_per_mlp = np.array([np.random.permutation(inputs.shape[0])
                                         for _ in self.mlps])
            x_batches = np.array_split(inputs[shuffles_per_mlp], n_batches, axis=1)
            y_batches = np.array_split(targets[shuffles_per_mlp], n_batches, axis=1)
            for i in range(n_batches):
                _, loss_per_mlp = self.sess.run([self.training_ops, self.losses_ops],
                                                feed_dict={
                                                    self.inputs_ph: x_batches[i],
                                                    self.targets_ph: y_batches[i]
                                                })
                avg_loss += np.array(loss_per_mlp) / n_batches
            if epoch % 20 == 0:
                logger.debug('Epoch {} | Losses {}'.format(epoch, avg_loss))
            losses[epoch] = avg_loss
        return losses

    def predict(self, inputs):
        mus, sigmas = zip(*self.sess.run(self.predict_ops, feed_dict={
            self.inputs_ph: np.broadcast_to(
                inputs, (self.ensemble_size, inputs.shape[0], self.inputs_dim))
        }))
        mus = np.array(mus).squeeze()
        sigmas = np.array(sigmas).squeeze()
        return mus, sigmas, tf.distributions.Normal(mus, sigmas).sample().eval()

    def build(self):
        for i in range(self.ensemble_size):
            self.mlps.append(InitializationAnchoredNn(
                self.sess,
                self.inputs_ph[i, ...],
                self.targets_ph[i, ...],
                str(i),
                **self.mlp_params

            ))
            self.predict_ops.append(self.mlps[i].predict_op)
            self.training_ops.append(self.mlps[i].training_op)
            self.losses_ops.append(self.mlps[i].loss)


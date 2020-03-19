import numpy as np
import tensorflow as tf


class InitializationAnchoredNN(object):
    def __init__(self,
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
        with tf.variable_scope(scope):
            layer = inputs
            for _ in range(3):
                layer = tf.layers.dense(inputs=layer, units=12, activation=tf.nn.tanh)
            self._mu = tf.layers.dense(inputs=layer, units=1)
            self._sigma = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.elu(x) + 1)
            dist = tf.distributions.Normal(loc=self._mu, scale=self._sigma)
            self._loss = tf.reduce_mean(-dist.log_prob(targets))
            self._training_op = tf.train.AdamOptimizer(learning_rate).minimize(self._loss)

    def _anchor_weights(self, lamba_anchors):
        """
        Based on "Uncertainty in Neural Networks: Approximately Bayesian Ensembling"
        https://arxiv.org/abs/1810.05546
        """
        # for i in range(len(self._layers)):
        #     kernel = self._layers[i].kernel.eval().copy()
        #     bias = self._layers[i].bias.eval().copy()
        #     self.loss += \
        #         lamba_anchors[i][0] * tf.losses.mean_squared_error(
        #             labels=kernel,
        #             predictions=self._layers[i].kernel
        #         )
        #     self.loss += \
        #         lamba_anchors[i][1] * tf.losses.mean_squared_error(
        #             labels=bias,
        #             predictions=self._layers[i].bias
        #         )
        pass

    @property
    def loss(self):
        return self._loss

    @property
    def training_op(self):
        return self._training_op

    @property
    def predict_op(self):
        return self._mu, self._sigma


class MLPEnsemble(object):
    def __init__(self,
                 sess,
                 inputs_dim,
                 outputs_dim,
                 ensemble_size,
                 n_epochs,
                 batch_size,
                 **mlp_kwargs):
        self.sess = sess
        self.ensemble_size = ensemble_size
        self.inputs_dim = inputs_dim
        self.epochs = n_epochs
        self.batch_size = batch_size
        self.log = True
        self.inputs_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(ensemble_size, None, inputs_dim)
        )
        self.targets_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(ensemble_size, None, outputs_dim)
        )
        self.mlps = [None] * ensemble_size
        self.predict_ops = [None] * ensemble_size
        self.training_ops = [None] * ensemble_size
        self.losses_ops = [None] * ensemble_size
        for i in range(self.ensemble_size):
            self.mlps[i] = InitializationAnchoredNN(
                self.inputs_ph[i, ...],
                self.targets_ph[i, ...],
                scope=str(i),
                **mlp_kwargs
            )
            self.predict_ops[i] = self.mlps[i].predict_op
            self.training_ops[i] = self.mlps[i].training_op
            self.losses_ops[i] = self.mlps[i].loss

    def fit(self, inputs, targets):
        assert inputs.shape[0] == targets.shape[0]
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
            if self.log and epoch % 20 == 0:
                print('Epoch ', epoch,  ' | Losses =', avg_loss)
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

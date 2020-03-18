import numpy as np
import tensorflow as tf


class InitializationAnchoredNN(object):
    def __init__(self,
                 sess,
                 scope,
                 input_dim,
                 targets_dim,
                 learning_rate,
                 n_layers,
                 hidden_size,
                 activation,
                 anchor,
                 init_std_bias,
                 init_std_weights,
                 data_noise):
        self._sess = sess
        self.inputs_ph = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name=(scope + '/inputs'))
        self.targets_ph = tf.placeholder(dtype=tf.float32, shape=(None, targets_dim), name=(scope + '/targets'))
        self._layers = []
        layer = self.inputs_ph
        # Define forward-pass.
        with tf.variable_scope(scope):
            lambda_anchor = []
            for i in range(n_layers):
                if anchor:
                    if i == 0:
                        std_weights = init_std_weights
                        std_bias = init_std_bias
                    else:
                        std_weights = 1. / np.sqrt(hidden_size)
                        std_bias = std_weights
                    kernel_initializer = tf.random_normal_initializer(0.0, std_weights)
                    bias_initializer = tf.random_normal_initializer(0.0, std_bias)
                    lambda_anchor.append((data_noise / (std_weights ** 2), data_noise / (std_bias ** 2)))
                else:
                    kernel_initializer = None
                    bias_initializer = tf.zeros_initializer()
                self._layers.append(tf.layers.Dense(
                    hidden_size,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer
                ))
                layer = self._layers[-1].apply(layer)
            # Define loss & train op.
            self._mu = tf.layers.dense(
                layer,
                1,
                activation=activation,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer
            )
            log_sigma = tf.layers.dense(
                layer,
                1
            )
            max_logsigma = tf.get_variable(
                'max_logsigma', shape=(targets_dim, ), initializer=tf.constant_initializer(1.0 / np.sqrt(targets_dim)))
            min_logsigma = tf.get_variable(
                'min_logsigma', shape=(targets_dim, ), initializer=tf.constant_initializer(-10.0 / np.sqrt(targets_dim))
            )
            log_sigma = max_logsigma - tf.nn.softplus(
                max_logsigma - log_sigma)
            log_sigma = min_logsigma + tf.nn.softplus(
                log_sigma - min_logsigma
            )
            self._sigma = tf.exp(log_sigma)
            prediction_dist = tf.distributions.Normal(self._mu, self._sigma)
            self.loss = tf.reduce_mean(-prediction_dist.log_prob(self.targets_ph)) + \
                        0.01 * tf.reduce_sum(max_logsigma) - \
                        0.01 * tf.reduce_sum(min_logsigma)
            # Anchor to weights to priors.
        if anchor:
            weights = []
            for layer in self._layers:
                weights += layer.weights
            init_weights = tf.variables_initializer(weights)
            self._sess.run(init_weights)
            self._anchor_weights(lambda_anchor)
        self.training_op = tf.train.AdamOptimizer(learning_rate).\
            minimize(self.loss)

    def _anchor_weights(self, lamba_anchors):
        """
        Based on "Uncertainty in Neural Networks: Approximately Bayesian Ensembling"
        https://arxiv.org/abs/1810.05546
        """
        for i in range(len(self._layers)):
            kernel = self._layers[i].kernel.eval().copy()
            bias = self._layers[i].bias.eval().copy()
            self.loss += \
                lamba_anchors[i][0] * tf.losses.mean_squared_error(
                    labels=kernel,
                    predictions=self._layers[i].kernel
                )
            self.loss += \
                lamba_anchors[i][1] * tf.losses.mean_squared_error(
                    labels=bias,
                    predictions=self._layers[i].bias
                )

    def fit(self, inputs, targets):
        loss, _ = self._sess.run([self.loss, self.training_op],
                                 feed_dict={self.inputs_ph: inputs,
                                            self.targets_ph: targets})
        return loss

    def predict(self, inputs):
        mu, sigma = self._sess.run([self._mu, self._sigma],
                                   feed_dict={self.inputs_ph: inputs})
        print("Stds: ", sigma)
        return tf.distributions.Normal(mu, sigma).sample().eval()

    @property
    def predict_op(self):
        return self._mu, self._sigma


class MLPEnsemble(object):
    def __init__(self,
                 sess,
                 ensemble_size,
                 n_epochs,
                 batch_size,
                 **mlp_kwargs):
        self.sess = sess
        self.ensemble_size = ensemble_size
        self.epochs = n_epochs
        self.batch_size = batch_size
        self.log = True
        self.mlps = []
        for i in range(self.ensemble_size):
            self.mlps.append(InitializationAnchoredNN(
                sess=sess,
                scope=str(i),
                **mlp_kwargs
            ))

    def fit(self, inputs, targets):
        assert inputs.shape[0] == targets.shape[0]
        # Collect training ops and losses.
        training_ops = [mlp.training_op for mlp in self.mlps]
        loss_ops = [mlp.loss for mlp in self.mlps]
        # Create data set for each mlp.
        losses = np.empty((self.epochs, self.ensemble_size))
        n_batches = int(np.ceil(inputs.shape[0] / self.batch_size))
        batches_per_mlp = []
        for _ in range(self.ensemble_size):
            rand_indices = np.random.permutation(inputs.shape[0])
            input_batches = np.array(np.array_split(inputs[rand_indices], n_batches))
            targets_batches = np.array(np.array_split(targets[rand_indices], n_batches))
            batches_per_mlp.append((input_batches, targets_batches))
        for epoch in range(self.epochs):
            average_loss_per_mlp = np.zeros(self.ensemble_size)
            for i, batches_list in enumerate(batches_per_mlp):
                shuffle_batches = np.random.permutation(n_batches)
                batches_per_mlp[i] = (batches_list[0][shuffle_batches], batches_list[1][shuffle_batches])
            for batch_index in range(n_batches):
                feed_dict = dict()
                for i, mlp in enumerate(self.mlps):
                    feed_dict.update({
                        mlp.inputs_ph: batches_per_mlp[i][0][batch_index],
                        mlp.targets_ph: batches_per_mlp[i][1][batch_index]
                    })
                _, loss_per_mlp = self.sess.run([training_ops, loss_ops], feed_dict=feed_dict)
                average_loss_per_mlp += np.array(loss_per_mlp) / n_batches
            losses[epoch] = average_loss_per_mlp
            if self.log and epoch % 20 == 0:
                print('Epoch ', epoch,  ' | Losses =', loss_per_mlp)
        return losses

    def predict(self, inputs):
        predict_ops = [mlp.predict_op for mlp in self.mlps]
        feed_dict = {mlp.inputs_ph: inputs for mlp in self.mlps}
        out = self.sess.run(predict_ops, feed_dict=feed_dict)
        mus = np.array([prediction[0] for prediction in out])
        sigmas = np.array([prediction[1] for prediction in out])
        return tf.distributions.Normal(mus, sigmas).sample().eval()

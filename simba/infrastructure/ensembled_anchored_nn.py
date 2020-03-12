import tensorflow as tf


class InitializationAnchoredNN(object):
    def __init__(self,
                 sess,
                 scope,
                 input_dim,
                 targets_dim,
                 learning_rate,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 anchor,
                 init_std_bias,
                 init_std_weights,
                 data_noise):
        self.sess = sess
        self.inputs_ph = tf.placeholder(tf.float64, [None, None, input_dim], name='inputs')
        self.targets_ph = tf.placehold(tf.float64, [None, targets_dim], name='targets')
        self.scope = scope
        self.layers = []
        layer = self.inputs_ph
        # Define forward-pass.
        with tf.variable_scope(scope):
            for _ in range(n_layers):
                self.layers.append(tf.layers.Dense(
                    size,
                    activation=activation,
                    kernel_initializer=tf.random_normal_initializer(0.0, init_std_weights),
                    bias_initializer=tf.random_normal_initializer(0.0, init_std_bias)
                ))
                layer = self.layers[-1].apply(layer)
            self.layers.append((
                tf.layers.Dense(
                    1,
                    activation=output_activation,
                    kernel_initializer=tf.random_normal_initializer(0.0, init_std_weights),
                    bias_initializer=tf.random_normal_initializer(0.0, init_std_bias)
                )
            ))
            self.layers.append((
                tf.layers.Dense(
                    1,
                    activation=output_activation,
                    kernel_initializer=tf.random_normal_initializer(0.0, init_std_weights),
                    bias_initializer=tf.random_normal_initializer(0.0, init_std_bias)
                )
            ))
        # TODO (yarden): might need here the softplus thing from
        #  https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/modeling/models/BNN.py
        #  line 414.
        # Define loss & train op.
        self.mu = self.layers[-1].apply(layer)
        self.log_std = self.layers[-2].apply(layer)
        prediction_dist = tf.distributions.Normal(self.mu, tf.exp(self.log_std))
        self._loss = tf.reduce_mean(-prediction_dist.log_prob(self.targets_ph))
        # Anchor to weights to priors.
        if anchor:
            weights = []
            for layer in self.layers:
                weights += layer.weights
            init_weights = tf.variables_initializer(weights)
            self.sess.run(init_weights)
            lambda_anchor_bias = data_noise / init_std_bias
            lambda_anchor_weights = data_noise / init_std_weights
            self._anchor_weights(lambda_anchor_bias, lambda_anchor_weights)
        self._training_op = tf.train.AdamOptimizer(learning_rate).\
            minimize(self._loss)


    def _anchor_weights(self, lambda_anchor_bias, lambda_anchor_weights):
        """
        Based on "Uncertainty in Neural Networks: Approximately Bayesian Ensembling"
        https://arxiv.org/abs/1810.05546
        """
        for layer in self.layers:
            for weight in layer.get_weights():
                kernel = weight[0].copy()
                bias = weight[1].copy()
                self._loss += \
                    lambda_anchor_weights * tf.losses.mean_squared_error(
                        labels=kernel,
                        predictions=layer.kernel
                    )
                self._loss += \
                    lambda_anchor_bias * tf.losses.mean_squared_error(
                        labels=bias,
                        predictions=layer.bias
                    )

    def fit(self, inputs, targets):
        loss, _ = self.sess.run([self._loss, self._training_op],
                                feed_dict=dict(inputs_ph=inputs,
                                               target_ph=targets))
        return loss

    def predict(self, inputs):
        mu, std = self.sess.run([self.mu, self.std],
                                feed_dict=dict(inputs_ph=inputs))
        return tf.distributions.Normal(mu, std).sample()

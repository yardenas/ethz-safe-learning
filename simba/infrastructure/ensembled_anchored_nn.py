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
        layer = self.inputs_ph
        for _ in range(3):
            layer = tf.layers.dense(inputs=layer, units=12, activation=tf.nn.tanh)
        mu = tf.layers.dense(inputs=layer, units=1)
        sigma = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.elu(x) + 1)
        dist = tf.distributions.Normal(loc=mu, scale=sigma)
        self.loss = tf.reduce_mean(-dist.log_prob(self.targets_ph))
        self.training_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

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
        mlp = self.mlps[0]
        for epoch in range(self.epochs):
            avg_loss = 0.0
            rand_ind = np.random.permutation(inputs.shape[0])
            x_batches = np.array_split(inputs[rand_ind], n_batches)
            y_batches = np.array_split(targets[rand_ind], n_batches)
            for i in range(n_batches):
                x_batch = np.expand_dims(np.squeeze(x_batches[i]), axis=1)
                y_batch = np.expand_dims(np.squeeze(y_batches[i]), axis=1)
                _, loss = self.sess.run([mlp.training_op, mlp.loss],
                                        feed_dict={
                                            mlp.inputs_ph: x_batch,
                                            mlp.targets_ph: y_batch})
                avg_loss += loss / n_batches
            if self.log and epoch % 20 == 0:
                print('Epoch ', epoch,  ' | Losses =', avg_loss)
        return losses

    def predict(self, inputs):
        predict_ops = [mlp.predict_op for mlp in self.mlps]
        feed_dict = {mlp.inputs_ph: inputs for mlp in self.mlps}
        out = self.sess.run(predict_ops, feed_dict=feed_dict)
        mus = np.array([prediction[0] for prediction in out])
        sigmas = np.array([prediction[1] for prediction in out])
        return mus, sigmas, tf.distributions.Normal(mus, sigmas).sample().eval()
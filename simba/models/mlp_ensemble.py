import numpy as np
import tensorflow.compat.v1 as tf
from simba.infrastructure.logging_utils import logger
tf.disable_v2_behavior()


def base_layer(inputs,
               units,
               activation,
               dropout_rate,
               training):
    output = tf.layers.dense(inputs, units)
    output = tf.layers.batch_normalization(output, training=training)
    output = activation(output)
    output = tf.cond(training,
                     lambda: tf.nn.dropout(output, rate=dropout_rate),
                     lambda: output)
    return output


class GaussianDistMlp(object):
    def __init__(self,
                 sess,
                 inputs,
                 targets,
                 scope,
                 learning_rate,
                 n_layers,
                 units,
                 activation,
                 dropout_rate,
                 training):
        self._sess = sess
        with tf.variable_scope(scope):
            x = inputs
            for _ in range(n_layers):
                x = base_layer(x, units, activation, dropout_rate, training)
            self._mu = tf.layers.dense(inputs=x, units=1)
            var = tf.layers.dense(inputs=x, units=1,
                                  activation=lambda t: tf.math.softplus(t) + 1e-4)
            self._sigma = tf.sqrt(var)
            self._loss = 0.5 * tf.reduce_sum(tf.log(2.0 * np.pi * var)) + 0.5 * tf.reduce_sum(
                tf.divide(tf.squared_difference(targets, self._mu), var)
            )
            self._training_op = tf.group([tf.get_collection(tf.GraphKeys.UPDATE_OPS),
                                          tf.train.AdamOptimizer(learning_rate).minimize(self._loss)]
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
                 validation_split,
                 mlp_params):
        self.sess = sess
        self.mlp_params = mlp_params
        self.ensemble_size = ensemble_size
        self.inputs_dim = inputs_dim
        self.epochs = n_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.training_ph = tf.placeholder(
            dtype=tf.bool
        )
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
                                                    self.targets_ph: y_batches[i],
                                                    self.training_ph: True
                                                })
                avg_loss += np.array(loss_per_mlp) / n_batches
            if epoch % 20 == 0:
                print('Epoch {} | Losses {}'.format(epoch, avg_loss))
            losses[epoch] = avg_loss
        return losses

    def predict(self, inputs):
        mus, sigmas = zip(*self.sess.run(self.predict_ops, feed_dict={
            self.inputs_ph: np.broadcast_to(
                inputs, (self.ensemble_size, inputs.shape[0], self.inputs_dim)),
            self.training_ph: False
        }))
        mus = np.array(mus).squeeze()
        sigmas = np.array(sigmas).squeeze()
        return mus, sigmas, tf.distributions.Normal(mus, sigmas).sample().eval()

    def build(self):
        for i in range(self.ensemble_size):
            self.mlps.append(GaussianDistMlp(
                sess=self.sess,
                inputs=self.inputs_ph[i, ...],
                targets=self.targets_ph[i, ...],
                scope=str(i),
                training=self.training_ph,
                **self.mlp_params

            ))
            self.predict_ops.append(self.mlps[i].predict_op)
            self.training_ops.append(self.mlps[i].training_op)
            self.losses_ops.append(self.mlps[i].loss)


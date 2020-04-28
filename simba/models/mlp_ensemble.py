import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def base_layer(inputs,
               units,
               activation,
               dropout_rate,
               training,
               reuse=None):
    output = tf.layers.dense(inputs, units, reuse=reuse)
    # TODO (yarden): not sure about BN
    # output = tf.layers.batch_normalization(output, training=training)
    output = activation(output)
    # output = tf.cond(training,
    #                  lambda: tf.nn.dropout(output, rate=dropout_rate),
    #                  lambda: output)
    return output


class GaussianDistMlp(object):
    def __init__(self,
                 output_dim,
                 scope,
                 learning_rate,
                 n_layers,
                 units,
                 activation,
                 dropout_rate):
        self.output_dim = output_dim
        self.scope = scope
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.units = units
        self.activation = eval(activation) if isinstance(activation, str) else activation
        self.dropout_rate = dropout_rate

    def training_ops(self, inputs, targets):
        # TODO (yarden): make sure that the training actually works here... (probably no.)
        mu, var = self.predict_ops(inputs, False, reuse=True)
        print("losssss muu", mu)
        loss = 0.5 * tf.reduce_sum(tf.log(2.0 * np.pi * var)) + 0.5 * tf.reduce_sum(
            tf.divide(tf.squared_difference(targets, mu), var)
        )
        return loss, tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

    def predict_ops(self, inputs, training, reuse=None):
        with tf.variable_scope(self.scope):
            x = inputs
            training = tf.constant(False)
            for _ in range(self.n_layers):
                x = base_layer(x, self.units, self.activation, self.dropout_rate, training, reuse=reuse)
            mu = tf.layers.dense(inputs=x, units=self.output_dim, name='mu', reuse=reuse)
            var = tf.layers.dense(inputs=x, units=self.output_dim,
                                  activation=lambda t: tf.math.softplus(t) + 1e-4, name='sigma', reuse=reuse)
            print("dfieneee muu", mu)
            return mu, var


class MlpEnsemble(object):
    def __init__(self,
                 sess,
                 scope,
                 inputs_dim,
                 outputs_dim,
                 ensemble_size,
                 n_epochs,
                 batch_size,
                 validation_split,
                 mlp_params):
        self.sess = sess
        self.scope = scope
        self.outputs_dim = outputs_dim
        self.mlp_params = mlp_params
        self.ensemble_size = ensemble_size
        self.inputs_dim = inputs_dim
        self.epochs = n_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.mlps = []

    def training_ops(self, inputs, targets):
        losses_ops = []
        training_ops = []
        for i, mlp in enumerate(self.mlps):
            print("dkdksldlksdfs", inputs[i])
            loss_op, training_op = mlp.training_ops(inputs[i], targets)
            training_ops.append(training_op)
            losses_ops.append(loss_op)
        return losses_ops, training_ops

    def fit(self, inputs, targets):
        pass

    def predict_ops(self, inputs):
        mu, var = [], []
        # Distribute inputs to the mlps.
        inputs_per_mlp = tf.split(inputs, self.ensemble_size, axis=0, name='distribute_inputs')
        for i, mlp in enumerate(self.mlps):
            # TODO (yarden): training shouldn't necessarily be false.
            mlp_mu, mlp_var = mlp.predict_ops(inputs_per_mlp[i], False)
            mu.append(mlp_mu)
            var.append(mlp_var)
        # This assumes an independant gaussian distribution.
        mus = tf.concat(mu, axis=0, name='merge_mus')
        vars = tf.concat(var, axis=0, name='merge_vars')
        return mus, vars

    def build(self):
        for i in range(self.ensemble_size):
            self.mlps.append(GaussianDistMlp(
                output_dim=self.outputs_dim,
                scope="mlp_" + str(i),
                **self.mlp_params
            ))

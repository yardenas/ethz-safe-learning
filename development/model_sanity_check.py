#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Spread Toy Example
# ## Initilization Anchored NN Ensemble Sanity Check
# 

# In this notebook I want to demonstrate that my tensorflow implementation of the ensemble neural network is
# actually working and useful. In the spirit of times, I will try to learn the _hypothetical_ spreading of the COVID-19
# disease in the _hypothetical_ island of Wakanda through the period of one year.
from simba.models.mlp_ensemble import MlpEnsemble
import tensorflow.compat.v1 as tf
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from simba.infrastructure.logging_utils import TrainingLogger

tf.set_random_seed(0)
np.random.seed(0)


# First we generate some data using the [SIR model]
# (https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html) of covid19:

def generate_covid_19_infection_rate_data():
    # https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html
    # https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
    population = 15000
    days = 365
    i_0, r_0 = 2, 0
    s_0 = population - i_0 - r_0
    beta, gamma = 0.3, 0.02
    t = np.linspace(0, days, days)

    def deriv(y, t, population, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / population
        dIdt = beta * S * I / population - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    y_0 = s_0, i_0, r_0
    ret = odeint(deriv, y_0, t, args=(population, beta, gamma))
    _, infected_people, _ = ret.transpose()
    return t, infected_people


# Say we have only have access to noisy measurements of how many people were sick on a certain day: 
time, infected_people = generate_covid_19_infection_rate_data()
n_samples = 25
noise = 0.01
time_augmented = np.array([])
infected_people_samples = np.array([])
for day, sick_people_that_day in zip(time, infected_people):
    time_augmented = np.append(time_augmented, np.full(n_samples, day))
    infected_people_samples = np.append(infected_people_samples, np.random.normal(
        sick_people_that_day, noise * sick_people_that_day, n_samples))
time_val = np.linspace(0, 365, 900)


# Some hyperparameters
def make_model(sess):
    mlp_dict = dict(
        learning_rate=0.0007,
        n_layers=5,
        units=64,
        activation=tf.nn.relu,
        dropout_rate=0.0
    )
    ensemble = MlpEnsemble(
        sess=sess,
        inputs_dim=1,
        scope="mlp_ensemble",
        outputs_dim=1,
        ensemble_size=5,
        n_epochs=250,
        batch_size=64,
        validation_split=0.1,
        mlp_params=mlp_dict
    )
    ensemble.build()
    return ensemble


def fit(inputs, targets, inputs_ph, targets_ph, training_ops, losses_ops,
        model):
    assert inputs.shape[0] == targets.shape[0], "Inputs batch size ({}) "
    "doesn't match targets batch size ({})".format(inputs.shape[0], targets.shape[0])
    losses = np.empty((model.epochs, model.ensemble_size))
    n_batches = int(np.ceil(inputs.shape[0] / model.batch_size))
    for epoch in range(model.epochs):
        avg_loss = 0.0
        shuffles_per_mlp = np.array([np.random.permutation(inputs.shape[0])
                                     for _ in model.mlps])
        x_batches = np.array_split(inputs[shuffles_per_mlp], n_batches, axis=1)
        y_batches = np.array_split(targets[shuffles_per_mlp], n_batches, axis=1)
        for i in range(n_batches):
            _, loss_per_mlp = model.sess.run([training_ops, losses_ops],
                                             feed_dict={
                                                 inputs_ph: x_batches[i],
                                                 targets_ph: y_batches[i]
                                             })
            avg_loss += np.array(loss_per_mlp) / n_batches
        if epoch % 20 == 0:
            print('Epoch {} | Losses {}'.format(epoch, avg_loss))
        losses[epoch] = avg_loss


def ensemble_negloglikelihood(mus, vars, targets, model):
    mus_per_mlp = tf.split(mus, model.ensemble_size, axis=0, name='split_mus')
    vars_per_mlp = tf.split(vars, model.ensemble_size, axis=0, name='split_vars')
    losses = []
    grad_ops = []
    for i, (mu, var) in enumerate(zip(mus_per_mlp, vars_per_mlp)):
        losses.append(0.5 * tf.reduce_sum(tf.log(2.0 * np.pi * var)) + 0.5 * tf.reduce_sum(
            tf.divide(tf.squared_difference(targets[i], mu), var)
        ))
        grad_ops.append(tf.train.AdamOptimizer(learning_rate=model.mlp_params['learning_rate']).minimize(losses[-1]))
    return losses, grad_ops

# Run the training loop:
n_particles = 20
x_test = np.broadcast_to(time_val, (n_particles, time_val.shape[0]))
x_test = np.reshape(x_test, (n_particles * time_val.shape[0]))
data_mean = time_augmented.mean()
data_std = time_augmented.std()
x = np.squeeze((time_augmented - data_mean) / (data_std + 1e-8))
x_test = (x_test - data_mean) / (data_std + 1e-8)
import time as t

t0 = t.time()
with tf.Session() as sess:
    model = make_model(sess)
    inputs_ph = tf.placeholder(
        dtype=tf.float32,
        shape=(None, model.inputs_dim)
    )
    targets_ph = tf.placeholder(
        dtype=tf.float32,
        shape=(model.ensemble_size, None, model.outputs_dim)
    )
    with tf.name_scope("inference"):
        mus, vars = model.predict_ops(inputs=inputs_ph)
        dist = tf.distributions.Normal(loc=mus, scale=tf.sqrt(vars))
        predict_ops = dist.mean(), dist.stddev(), dist.sample()
    with tf.name_scope("training"):
        lossess, grad_ops = ensemble_negloglikelihood(mus, vars, targets_ph, model)
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs', sess.graph)
    writer.close()
    fit(inputs=x[:, np.newaxis], targets=infected_people_samples[:, np.newaxis],
        inputs_ph=inputs_ph, targets_ph=targets_ph, losses_ops=lossess, training_ops=grad_ops,
        model=model)
    t1 = t.time()
    print("train time:", t1 - t0)
    dist = sess.run(predict_ops, feed_dict={
        inputs_ph: np.reshape(x_test[:, np.newaxis],
                              (model.ensemble_size, -1, model.inputs_dim))
    })
    t2 = t.time()
    print("inferecne time:", t2 - t1)
    mus, sigmas, preds = dist

# The total uncertainty (epistemic and aleatoric) using monte-carlo estimation
# using data sampled from _ensemble_size_ and _n\_particles_
# For more details on decomposition of uncertainties: http://proceedings.mlr.press/v80/depeweg18a/depeweg18a.pdf 
preds = np.reshape(preds,
                   (model.ensemble_size, -1, time_val.shape[0]))
aleatoric_monte_carlo_uncertainty = np.mean(np.std(preds, axis=1) ** 2, axis=0)
epistemic_monte_carlo_uncertainty = np.std(np.mean(preds, axis=1), axis=0) ** 2
total_monte_carlo_uncertainty = aleatoric_monte_carlo_uncertainty + epistemic_monte_carlo_uncertainty

fig = plt.figure(figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
ax = fig.subplots()
ax.set_ylim([-100, 12.5e3])
ax.scatter(time_augmented, infected_people_samples, color='#FF9671', alpha=0.09,
           s=20, label='Infected today people a day')
ax.plot(time_val, np.mean(preds, axis=(0, 1)), '-', color='#845EC2', linewidth=1.5,
        label='Mean over all particles and MLPs', alpha=0.8)
ax.fill_between(time_val, np.mean(preds, axis=(0, 1)) - np.sqrt(total_monte_carlo_uncertainty),
                np.mean(preds, axis=(0, 1)) + np.sqrt(total_monte_carlo_uncertainty),
                color='#FF6F91', alpha=0.5, label='Total monte-carlo standard deviation')
ax.legend(loc='upper right', fontsize='medium')
plt.xlabel("Days")
plt.ylabel("Infectious people")
plt.show()

mus = np.reshape(mus,
                 (model.ensemble_size, -1, time_val.shape[0]))
sigmas = np.reshape(sigmas,
                    (model.ensemble_size, -1, time_val.shape[0]))
aleatoric_explicit_uncertainty = np.mean(sigmas ** 2, axis=(0, 1))
fig = plt.figure(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(211)
ax1.plot(time_val, aleatoric_monte_carlo_uncertainty, label='Monte-carlo estimated aleatoric uncertainty')
ax1.plot(time_val, aleatoric_explicit_uncertainty, label='Explicit aleatoric uncertainty')
ax1.plot(time, (infected_people * noise) ** 2, label='Ground truth')
ax1.legend(loc='upper right', fontsize='medium')
ax2 = fig.add_subplot(212)
ax2.plot(time_val, epistemic_monte_carlo_uncertainty, label='Monte-carlo epistemic uncertainty')
ax2.legend(loc='upper right', fontsize='medium')
plt.show()

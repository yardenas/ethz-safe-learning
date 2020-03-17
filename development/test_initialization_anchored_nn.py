from simba.infrastructure import MLPEnsemble, InitializationAnchoredNN
import tensorflow as tf
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns



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


def make_model(args,
               sess,
               input_dim,
               targets_dim):
    mlp_dict = dict(
        input_dim=input_dim,
        targets_dim=targets_dim,
        learning_rate=args.learning_rate,
        n_layers=args.n_layers,
        hidden_size=args.hidden_size,
        activation=args.activation,
        anchor=args.anchor,
        init_std_bias=args.init_std_bias,
        init_std_weights=args.init_std_weights,
        data_noise=args.data_noise
    )
    return MLPEnsemble(
        sess,
        args.ensemble_size,
        args.n_epochs,
        args.batch_size,
        **mlp_dict
    )


def main(args):
    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)
    time, infected_poeple = generate_covid_19_infection_rate_data()
    n_samples = 5
    noise = 0.1
    inputs = np.array([])
    targets = np.array([])
    for day, sick_people_that_day in zip(time, infected_poeple):
        inputs = np.append(inputs, np.full(n_samples, day))
        targets = np.append(targets, np.random.normal(sick_people_that_day, noise * sick_people_that_day, n_samples))
    with tf.Session() as sess:
        vars(args).update({'activation': tf.nn.relu})
        vars(args)['data_noise'] = noise
        model = make_model(args, sess, 1, 1)
        sess.run(tf.global_variables_initializer())
        mean = inputs.mean()
        std = inputs.std()
        x = (inputs - mean) / (std + 1e-8)
        model.fit(x[:, np.newaxis], targets[:, np.newaxis])
        pred = np.squeeze(model.predict((time[:, np.newaxis] - mean) / (std + 1e-8)))
    fig, ax = plt.subplots()
    ax.scatter(inputs, targets, color='#FF764D', alpha=0.6,
               s=5, label='Infectious people a day')
    pred_mean = pred.mean()
    pred_std = pred.std()
    ax.plot(time, pred, '-', color='#C20093', linewidth=1, label='Predictions')
    ax.fill_between(time, pred - pred_std, pred + pred_std,
                    color='#FC206C', alpha=0.15, label='Confidence interval')
    legend = ax.legend(loc='upper right', fontsize='medium')
    plt.xlabel("Days")
    plt.ylabel("Infectious people")
    plt.show()

# def main(args):
#     X_train = np.atleast_2d([1., 4.5, 5.1, 6., 8., 9.]).T
#     X_train = X_train / 5. - 1
#     X_train = np.squeeze(X_train)
#     X_train = np.tile(X_train, 50)
#     y_train = X_train * np.sin(X_train * 5.)
#     y_train = np.random.normal(y_train, 1e-8 * np.abs(y_train))
#     X_val = np.atleast_2d(np.linspace(-3, 3, 100)).T
#     x_dim = 1
#     y_dim = 1
#     hidden_size = 100
#     n_epochs = 200
#     learning_rate = 0.01
#     n_ensembels = 5
#     data_noise = 0.001
#     nns = []
#     with tf.Session() as sess:
#         for ens in range(n_ensembels):
#             nns.append(InitializationAnchoredNN(
#                 sess,
#                 str(ens)+"ddld",
#                 x_dim,
#                 y_dim,
#                 learning_rate,
#                 2,
#                 hidden_size,
#                 tf.nn.tanh,
#                 True,
#                 0.1,
#                 0.1,
#                 data_noise
#             ))
#
#         sess.run(tf.global_variables_initializer())
#         for ens in range(n_ensembels):
#             for _ in range(n_epochs):
#                 loss = nns[ens].fit(X_train[:, np.newaxis], y_train[:, np.newaxis])
#                 print("Ensemble: ", ens, " loss: ", loss)
#         y_pred = []
#         for ens in range(n_ensembels):
#             y_pred.append(nns[ens].predict(X_val))
#         fig = plt.figure(figsize=(10, 4))
#         ax = fig.add_subplot(111)
#         for ens in range(n_ensembels):
#             ax.plot(X_val, y_pred[ens], 'k')
#         ax.plot(X_train, y_train, 'r.', markersize=14,
#                 markeredgecolor='k', markeredgewidth=0.5)
#         ax.set_ylim(-4, 2)
#         ax.set_xlim(-2.5, 2.5)
#         plt.show()
#
#         # combine ensembles estimates properly
#         y_preds = np.array(y_pred)
#         y_preds = y_preds[:, :, 0]
#         y_pred_mu = np.mean(y_preds, axis=0)
#         y_pred_std = np.std(y_preds, axis=0, ddof=1)
#
#         # add on data noise
#         y_pred_std = np.sqrt(np.square(y_pred_std) + data_noise)
#
#         # plot predictive distribution
#         fig = plt.figure(figsize=(10, 4))
#         ax = fig.add_subplot(111)
#
#         ax.plot(X_val, y_pred_mu, 'b-', linewidth=2., label=u'Prediction')
#         ax.plot(X_val, y_pred_mu + 2 * y_pred_std, 'b', linewidth=0.5)
#         ax.plot(X_val, y_pred_mu - 2 * y_pred_std, 'b', linewidth=0.5)
#         ax.plot(X_val, y_pred_mu + 1 * y_pred_std, 'b', linewidth=0.5)
#         ax.plot(X_val, y_pred_mu - 1 * y_pred_std, 'b', linewidth=0.5)
#         ax.fill(np.concatenate([X_val, X_val[::-1]]),
#                 np.concatenate([y_pred_mu - 2 * y_pred_std,
#                                 (y_pred_mu + 2 * y_pred_std)[::-1]]),
#                 alpha=1, fc='lightskyblue', ec='None')
#         ax.fill(np.concatenate([X_val, X_val[::-1]]),
#                 np.concatenate([y_pred_mu - 1 * y_pred_std,
#                                 (y_pred_mu + 1 * y_pred_std)[::-1]]),
#                 alpha=1, fc='deepskyblue', ec='None')
#
#         ax.plot(X_train, y_train, 'r.', markersize=14,
#                 markeredgecolor='k', markeredgewidth=0.5)
#         ax.set_ylim(-4, 2)
#         ax.set_xlim(-2.5, 2.5)
#         plt.show()

# def f(x):
#     return x ** 2 - 6 * x + 9
#
#
# def data_generator(x, sigma_0, samples):
#     return np.random.normal(f(x), sigma_0 * x, samples)
#
#
# def main(args):
#     sigma_0 = 0.1
#     x_vals = np.arange(1,5.2,0.2)
#     x_arr = np.array([])
#     y_arr = np.array([])
#     samples = 50
#     for x in x_vals:
#         x_arr = np.append(x_arr, np.full(samples,x))
#         y_arr = np.append(y_arr, data_generator(x,sigma_0,samples))
#     idi = np.random.permutation(x_arr.shape[0])
#     x_arr, y_arr = x_arr[idi], y_arr[idi]
#     x_test = np.arange(1.1,5.1,0.2)
#     fig, ax = plt.subplots(figsize=(10, 10))
#     plt.grid(True)
#     plt.xlabel('x')
#     plt.ylabel('g(x)')
#     ax.scatter(x_arr, y_arr, label='sampled data')
#     ax.plot(x_vals, f(x_vals), c='m', label='f(x)')
#     ax.legend(loc='upper center', fontsize='large', shadow=True)
#     plt.show()
#     with tf.Session() as sess:
#         vars(args).update({'activation': tf.nn.relu})
#         model = make_model(args, sess, 1, 1)
#         sess.run(tf.global_variables_initializer())
#         model.fit(x_arr[:, np.newaxis], y_arr[:, np.newaxis])
#         pred = model.predict(x_test[:, np.newaxis])
#     fig, ax = plt.subplots(figsize=(10,10))
#     plt.grid(True)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.scatter(x_test, pred, alpha=1.0, label='predidcic')
#     ax.scatter(x_arr,y_arr,c='b',alpha=0.05,label='sampled data')
#     ax.errorbar(x_vals,f(x_vals),yerr=sigma_0*x_vals,c='b',lw=2,ls='None',marker='.',ms=10,label='true distributions')
#     ax.plot(x_vals,f(x_vals),c='m',label='f(x)')
#     ax.legend(loc='upper center',fontsize='large',shadow=True)
#     plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--anchor', action='store_true')
    parser.add_argument('--init_std_bias', type=float, default=0.1)
    parser.add_argument('--init_std_weights', type=float, default=0.1)
    parser.add_argument('--data_noise', type=float, default=0.01)
    parser.add_argument('--ensemble_size',  type=int, default=5)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    main(args)

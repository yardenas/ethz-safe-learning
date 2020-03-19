from simba.infrastructure import MLPEnsemble
import tensorflow as tf
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


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
               outputs_dim):
    mlp_dict = dict(
        learning_rate=args.learning_rate,
        n_layers=args.n_layers,
        hidden_size=args.hidden_size,
        activation=eval(args.activation),
        anchor=args.anchor,
        init_std_bias=args.init_std_bias,
        init_std_weights=args.init_std_weights,
        data_noise=args.data_noise
    )
    return MLPEnsemble(
        sess,
        input_dim,
        outputs_dim,
        args.ensemble_size,
        args.n_epochs,
        args.batch_size,
        **mlp_dict
    )


def main(args):
    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)
    time, infected_poeple = generate_covid_19_infection_rate_data()
    n_samples = 3
    noise = 0.1
    inputs = np.array([])
    targets = np.array([])
    for day, sick_people_that_day in zip(time, infected_poeple):
        inputs = np.append(inputs, np.full(n_samples, day))
        targets = np.append(targets, np.random.normal(sick_people_that_day, noise * sick_people_that_day, n_samples))
    with tf.Session() as sess:
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--activation', type=str, default='tf.nn.relu')
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--anchor', action='store_true')
    parser.add_argument('--init_std_bias', type=float, default=0.1)
    parser.add_argument('--init_std_weights', type=float, default=0.1)
    parser.add_argument('--data_noise', type=float, default=0.1)
    parser.add_argument('--ensemble_size',  type=int, default=5)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    main(args)


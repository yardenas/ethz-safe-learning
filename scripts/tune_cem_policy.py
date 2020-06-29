import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from simba.policies.cem_mpc import CemMpc


def make_new_policy(model, environment, horizon, iterations, n_samples, elite_ratio, policy_kwargs):
    return CemMpc(
        model=model,
        environment=environment,
        horizon=horizon,
        iterations=iterations,
        n_samples=n_samples,
        n_elite=round(elite_ratio * n_samples),
        particles=policy_kwargs['particles'],
        stddev_threshold=policy_kwargs['stddev_threshold'],
        noise_stddev=policy_kwargs['noise_stddev'],
        smoothing=policy_kwargs['smoothing']
    )


def visualize_grid_search(horizon_scores,
                          horizons,
                          proposals_with_iterations,
                          elite_ratios,
                          logdir,
                          name,
                          cmap='Blues'):
    assert len(horizon_scores) == 4
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
    plt.setp(axes, xticks=[0.5, 1.5, 2.5], yticks=[0.5, 1.5, 2.5],
             xticklabels=list(map(str, elite_ratios)),
             yticklabels=list(map(lambda x: str(x).strip('(').strip(')'), proposals_with_iterations)))
    max_value = np.max(np.asarray(horizon_scores))
    min_value = np.min(np.asarray(horizon_scores))
    fig.text(0.5, 0.94, 'Horizon length', ha='center', fontsize=13)
    fig.text(0.005, 0.275, '#Proposals, #Iterations', rotation='vertical', fontsize=13)
    fig.text(0.51, 0.04, 'Elite ratio', ha='center', fontsize=13)
    for ax, horizon, horizon_score in zip(axes.flatten(), horizons, horizon_scores):
        ax.pcolor(np.reshape(np.asarray(horizon_score), (3, 3)), cmap=cmap,
                  vmin=min_value, vmax=max_value)
        ax.set_rasterized(True)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_title(str(horizon))
    save_path = os.path.join(logdir, name + '.svg')
    plt.savefig(save_path)


def main():
    from config.config import load_config_or_die, pretty_print
    from simba.infrastructure.logging_utils import init_loggging
    from simba.infrastructure.common import dump_string, get_git_hash
    from simba.agents.agent_factory import make_agent
    from simba.environment_utils.environment_factory import make_environment
    from simba.infrastructure.trainer import RLTrainer
    parser = argparse.ArgumentParser()
    log_dir_suffix = time.strftime("%d-%m-%Y_%H-%M-%S")
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='experiments')
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--config_dir', type=str, required=True)
    parser.add_argument('--config_basename', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--cuda_device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    experiment_log_dir = args.log_dir + '/' + args.name + '_' + log_dir_suffix
    os.makedirs(experiment_log_dir, exist_ok=True)
    init_loggging(args.log_level)
    params = load_config_or_die(args.config_dir, args.config_basename)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    logging.getLogger('simba').info("Startning a training session with parameters:\n" +
                                    pretty_print(params))
    dump_string(pretty_print(params) + '\n' +
                'git hash: ' + get_git_hash(), experiment_log_dir + '/params.txt')
    env = make_environment(params)
    agent = make_agent(params, env)
    trainer_options = params['options'].get('trainer_options')
    trainer_options['training_logger_params'].update({
        'log_dir': (experiment_log_dir + '/training_data')
    })
    trainer = RLTrainer(agent=agent,
                        environemnt=env,
                        **trainer_options)
    trainer.train(params['options'].pop('train_iterations'))
    horizons = [8, 10, 12, 15]
    proposals_with_iterations = [(100, 15), (150, 10), (300, 5)]
    elites_ratio = [0.05, 0.1, 0.2]
    horizon_scores = []
    horizon_costs = []
    for horizon in horizons:
        scores = []
        costs = []
        for proposal_iteration in proposals_with_iterations:
            for ratio in elites_ratio:
                logging.getLogger('simba').info(
                    "Evaluating {} ratio with {} (proposal, iterations) on horizon of length {}.".format(
                        ratio, proposal_iteration, horizon)
                )
                # Making new policy to trigger tensorflow's retracing and update optimization parameters.
                agent.policy = make_new_policy(agent.model,
                                               env,
                                               horizon,
                                               proposal_iteration[1],
                                               proposal_iteration[0],
                                               ratio,
                                               params['policies']['cem_mpc'])
                evaluation_metrics = trainer.evaluate_agent(7000, 1000)
                mean_rewards, stddev_rewards = \
                    evaluation_metrics['training_rl_objective'], evaluation_metrics['sum_rewards_stddev']
                mean_costs, stddev_costs = \
                    evaluation_metrics['sum_costs_mean'], evaluation_metrics['sum_costs_stddev']
                scores.append(mean_rewards)
                costs.append(mean_costs)
                logging.getLogger('simba') \
                    .info("Mean score is: {}. Standard deviation is: {}\nMean cost is: {}. Standard deviation is {}"
                          .format(mean_rewards, stddev_rewards, mean_costs, stddev_costs))
        horizon_costs.append(costs)
        horizon_scores.append(scores)
    for name, color, metric in zip(['scores', 'costs'], ['Blues', 'Reds'], [horizon_scores, horizon_costs]):
        visualize_grid_search(metric,
                              horizons,
                              proposals_with_iterations,
                              elites_ratio,
                              experiment_log_dir,
                              name,
                              color)
    if args.visualize:
        plt.show()


if __name__ == '__main__':
    main()

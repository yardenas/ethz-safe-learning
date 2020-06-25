import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def parse_tf_event_file(file_path):
    print('Parsing event file {}'.format(file_path))
    ea = event_accumulator.EventAccumulator(file_path)
    ea.Reload()
    rl_objective, mean_sum_costs, sum_costs, timesteps = [], [], [], []
    for objective, mean_sum_cost, sum_cost in zip(ea.Scalars('eval_rl_objective'), ea.Scalars('eval_mean_sum_costs'),
                                                  ea.Scalars('sum_costs')):
        rl_objective.append(objective.value)
        mean_sum_costs.append(mean_sum_cost.value)
        sum_costs.append(sum_cost.value)
        timesteps.append(mean_sum_cost.step)
    return rl_objective, mean_sum_costs, sum_costs, timesteps


def parse_experiment_data(experiment_path):
    files = list(Path(experiment_path).glob('**/events.out.tfevents.*'))
    assert len(files) == 4, 'Expected four seeds per experiment.'
    rl_objectives, mean_sum_costs, sum_costs, timesteps = [], [], [], []
    for file in files:
        run_rl_objective, run_mean_sum_costs, run_sum_costs, run_timesteps = parse_tf_event_file(str(file))
        rl_objectives.append(run_rl_objective)
        mean_sum_costs.append(run_mean_sum_costs)
        sum_costs.append(run_sum_costs)
        timesteps.append(run_timesteps)
    return np.asarray(rl_objectives), np.asarray(mean_sum_costs), np.asarray(sum_costs), np.asarray(timesteps)


def median_percentiles(metric):
    median = np.median(metric, axis=0)
    upper_percentile = np.percentile(metric, 95, axis=0, interpolation='linear')
    lower_percentile = np.percentile(metric, 5, axis=0, interpolation='linear')
    return median, upper_percentile, lower_percentile


def make_statistics(eval_rl_objectives, eval_mean_sum_costs, sum_costs, timesteps):
    assert np.all(np.equal(timesteps[0, :], timesteps)), 'All experiments should have the same amount of steps.'
    objectives_median, objectives_upper, objectives_lower = median_percentiles(eval_rl_objectives)
    mean_sum_costs_median, mean_sum_costs_upper, mean_sum_costs_lower = median_percentiles(eval_mean_sum_costs)
    average_costs_median, average_costs_upper, average_costs_lower = median_percentiles(
        (sum_costs / timesteps))
    return dict(objectives_median=objectives_median,
                objectives_upper=objectives_upper,
                objectives_lower=objectives_lower,
                mean_sum_costs_median=mean_sum_costs_median,
                mean_sum_costs_upper=mean_sum_costs_upper,
                mean_sum_costs_lower=mean_sum_costs_lower,
                average_costs_median=average_costs_median,
                average_costs_upper=average_costs_upper,
                average_costs_lower=average_costs_lower,
                timesteps=timesteps[0, :]
                )


def draw(ax, timesteps, median, upper, lower, label):
    ax.plot(timesteps, median, label=label)
    ax.fill_between(timesteps, lower, upper, alpha=0.2)


def resolve_name(name):
    if name == 'unaware':
        return 'CEM-MPC'
    elif name == 'aware':
        return 'Safe CEM-MPC'
    elif name == 'no_sample':
        return 'Deterministic Safe CEM-MPC'
    else:
        return name


def draw_experiments_results(data_path):
    fig, (rl_objective_ax, mean_sum_cost_ax, average_cost_ax) = plt.subplots(1, 3, figsize=(11, 3.5))
    for root, experiment_dirs, _ in os.walk(data_path):
        for experiment_name in experiment_dirs:
            print('Processing experiment {}...'.format(experiment_name))
            experiment_statistics = make_statistics(*parse_experiment_data(
                os.path.join(root, experiment_name)))
            draw(rl_objective_ax,
                 experiment_statistics['timesteps'],
                 experiment_statistics['objectives_median'],
                 experiment_statistics['objectives_upper'],
                 experiment_statistics['objectives_lower'],
                 label=resolve_name(experiment_name))
            draw(mean_sum_cost_ax,
                 experiment_statistics['timesteps'],
                 experiment_statistics['mean_sum_costs_median'],
                 experiment_statistics['mean_sum_costs_upper'],
                 experiment_statistics['mean_sum_costs_lower'],
                 label=resolve_name(experiment_name))
            draw(average_cost_ax,
                 experiment_statistics['timesteps'],
                 experiment_statistics['average_costs_median'],
                 experiment_statistics['average_costs_upper'],
                 experiment_statistics['average_costs_lower'],
                 label=resolve_name(experiment_name))
        break
    rl_objective_ax.set_ylim([-1.0, 30.0])
    mean_sum_cost_ax.set_ylim([0.0, 100.0])
    average_cost_ax.set_ylim([0.0, 0.1])
    rl_objective_ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    rl_objective_ax.set_xlabel('Interaction time steps')
    rl_objective_ax.set_ylabel('Average sum of rewards')
    mean_sum_cost_ax.set_xlabel('Interaction time steps')
    mean_sum_cost_ax.set_ylabel('Average sum of costs')
    average_cost_ax.set_xlabel('Interaction time steps')
    average_cost_ax.set_ylabel('Temporal average cost')
    mean_sum_cost_ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    average_cost_ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    rl_objective_ax.plot(experiment_statistics['timesteps'],
                         np.ones_like(experiment_statistics['timesteps']) * 17.5,
                         color='orangered', ls='--', label='TRPO-Lagrangian')
    mean_sum_cost_ax.plot(experiment_statistics['timesteps'],
                          np.ones_like(experiment_statistics['timesteps']) * 25.0,
                          color='orangered', ls='--', label='TRPO-Lagrangian')
    average_cost_ax.plot(experiment_statistics['timesteps'],
                         np.ones_like(experiment_statistics['timesteps']) * 0.025,
                         color='orangered', ls='--', label='TRPO-Lagrangian')

    fig.legend(*rl_objective_ax.get_legend_handles_labels(), loc='upper center', mode='extand')
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, wspace=0.23)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()
    draw_experiments_results(args.data_path)

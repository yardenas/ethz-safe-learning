import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_experiment_data(experiment_path):
    files = list(Path(experiment_path).glob('**/*.csv'))
    assert len(files) == 8, 'Expected only four seeds per experiment.'
    rl_objectives, mean_sum_costs, timesteps = [], [], []
    for file in files:
        data = pd.read_csv(file)
        if 'eval_mean_sum_costs' in str(file):
            mean_sum_costs.append(data['Value'])
            timesteps.append(data['Step'])
        elif 'eval_rl_objective' in str(file):
            rl_objectives.append(data['Value'])
        else:
            raise NameError(file)
    return np.asarray(rl_objectives), np.asarray(mean_sum_costs), np.asarray(timesteps)


def median_percentiles(metric):
    median = np.median(metric, axis=0)
    upper_percentile = np.percentile(metric, 95, axis=0, interpolation='linear')
    lower_percentile = np.percentile(metric, 5, axis=0, interpolation='linear')
    return median, upper_percentile, lower_percentile


def temporal_average_costs(timesteps, mean_sum_costs):
    all_average_costs = []
    for times, costs in zip(timesteps, mean_sum_costs):
        experiment_average_costs = []
        cost_so_far = 0.0
        for time_point, cost in zip(times, costs):
            cost_so_far += cost * 5.0
            experiment_average_costs.append(cost_so_far / time_point)
        all_average_costs.append(experiment_average_costs)
    return np.asarray(all_average_costs)


def make_statistics(eval_rl_objectives, eval_mean_sum_costs, timesteps):
    assert np.all(np.equal(timesteps[0, :], timesteps)), 'All experiments should have the same amount of steps.'
    objectives_median, objectives_upper, objectives_lower = median_percentiles(eval_rl_objectives)
    mean_sum_costs_median, mean_sum_costs_upper, mean_sum_costs_lower = median_percentiles(eval_mean_sum_costs)
    average_costs_median, average_costs_upper, average_costs_lower = median_percentiles(
        temporal_average_costs(timesteps, eval_mean_sum_costs))
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
        return 'Safe-CEM-MPC'
    else:
        return name


def draw_experiments_results(data_path):
    fig, (rl_objective_ax, mean_sum_cost_ax, average_cost_ax) = plt.subplots(1, 3, figsize=(11, 3.5))
    for root, experiment_dirs, _ in os.walk(data_path):
        for experiment_name in experiment_dirs:
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
    rl_objective_ax.set_ylim([-1.0, 30.0])
    mean_sum_cost_ax.set_ylim([0.0, 100.0])
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
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()
    draw_experiments_results(args.data_path)

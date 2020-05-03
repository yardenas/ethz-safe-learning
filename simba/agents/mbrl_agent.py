import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from simba.infrastructure.common import standardize_name
from simba.infrastructure.logging_utils import logger
from simba.agents import BaseAgent
from simba.policies import CemMpc, RandomMpc
from simba.models.transition_model import TransitionModel


class MbrlAgent(BaseAgent):
    def __init__(self,
                 seed,
                 environment,
                 warmup_timesteps,
                 train_batch_size,
                 train_interaction_steps,
                 episode_length,
                 replay_buffer_size,
                 **kwargs
                 ):
        super().__init__(
            seed,
            replay_buffer_size)
        self.observation_space_dim = environment.observation_space.shape[0]
        self.actions_space_dim = environment.action_space.shape[0]
        self.train_batch_size = train_batch_size
        self.train_interaction_steps = train_interaction_steps
        self.episode_length = episode_length
        self.warmup_policy = self._make_policy('random_mpc', kwargs['policy_params'], environment)
        self.warmup_timesteps = warmup_timesteps
        self.total_warmup_timesteps_so_far = 0
        assert all(key in kwargs.keys() for key in ('policy', 'policy_params', 'model', 'model_params')), \
            "Did not specify a policy or a model."
        kwargs['model_params']['scale_features'] = kwargs['scale_features']
        self.model = self._make_model(kwargs.pop('model'), kwargs.pop('model_params'), environment)
        self.policy = self._make_policy(kwargs.pop('policy'), kwargs.pop('policy_params'), environment)

    def set_random_seeds(self, seed):
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

    @property
    def warm(self):
        return self.total_warmup_timesteps_so_far >= self.warmup_timesteps

    def update(self):
        # TODO (yarden): not sure about random data, maybe everything, maybe sample N trajectories.
        observations, actions, _, next_observations, _ = \
            self.replay_buffer.sample_recent_data(self.train_batch_size)
        observations_with_actions = np.concatenate([
            observations,
            actions], axis=1
        )
        losses = self.model.fit(observations_with_actions, next_observations)
        self.training_report['losses'] = losses

    def _interact(self, environment):
        if not self.warm:
            samples, timesteps_this_batch = self.sample_trajectories(
                environment,
                self.warmup_policy,
                self.warmup_timesteps,
                self.episode_length
            )
            self.total_warmup_timesteps_so_far += timesteps_this_batch
        else:
            samples, timesteps_this_batch = self.sample_trajectories(
                environment,
                self.policy,
                self.train_interaction_steps,
                self.episode_length
            )
        return samples, timesteps_this_batch

    def _build(self):
        self.model.build()
        self.policy.build()
        logger.info("Done building Mbrl agent computational graph.")

    def _load(self):
        raise NotImplementedError

    def report(self,
               environment,
               eval_interaction_steps,
               eval_episode_length):
        logger.info("Evaluating policy.")
        evaluation_trajectories, _ = self.sample_trajectories(
            environment,
            self.policy,
            eval_interaction_steps,
            eval_episode_length)
        eval_return_values = np.array([trajectory['reward'].sum() for
                                       trajectory in evaluation_trajectories])
        ground_truth_states = evaluation_trajectories[0]['observation']
        action_sequences = np.tile(evaluation_trajectories[0]['action'], (20, 1, 1))
        start_states = np.tile(ground_truth_states[0, ...], (20, 1))
        predicted_states = self.model.simulate_trajectories(
            start_states, action_sequences).reshape((20, ground_truth_states.shape[0], ground_truth_states.shape[1]))
        self.training_report['predicted_states_vs_ground_truth'] = make_prediction_error_figure(predicted_states,
                                                                                                ground_truth_states)
        self.training_report.update(dict(
            eval_rl_objective=eval_return_values.mean(),
            sum_rewards_stddev=eval_return_values.std()
        ))
        return self.training_report

    def _make_policy(self, policy, policy_params, environment):
        eval_policy = eval(standardize_name(policy))
        policy_params['environment'] = environment
        if eval_policy == RandomMpc:
            return RandomMpc(policy_params['environment'].action_space)
        if policy_params is None:
            return eval((standardize_name(policy)))()
        return eval((standardize_name(policy)))(model=self.model, **policy_params)

    def _make_model(self, model, model_params, environment):
        return TransitionModel(
            model=model,
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            **model_params)


def make_prediction_error_figure(predicted_states, ground_truth_states):
    observation_dim = ground_truth_states.shape[1]
    cols = 2
    rows = int(np.ceil(observation_dim / cols))
    fig = plt.figure()
    t = np.arange(predicted_states.shape[1])
    for dim in range(observation_dim):
        ax = fig.add_subplot(rows, cols, dim + 1)
        ax.errorbar(t, predicted_states[..., dim].mean(axis=0),
                    yerr=predicted_states[..., dim].std(axis=0),
                    c='bisque', ls='None', marker='.', ms=3,
                    label='predicted distributions', alpha=0.7)
        ax.plot(ground_truth_states[..., dim], 'skyblue',
                label='ground truth')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize='medium')
    mse = np.mean((predicted_states - ground_truth_states) ** 2)
    fig.suptitle('Predicted states vs. true states \n' + 'Mean squared error: ' + str(mse))
    return fig

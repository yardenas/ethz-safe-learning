import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from simba.infrastructure.common import standardize_name
from simba.infrastructure.logging_utils import logger
from simba.agents import BaseAgent
from simba.policies import CemMpc, MppiMpc, RandomShootingMpc, RandomMpc
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
            replay_buffer_size,
            **kwargs)
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
        observations, actions, next_observations, _, _ = \
            self.replay_buffer.sample_recent_data(self.train_batch_size)
        observations_with_actions = np.concatenate([
            observations,
            actions], axis=1
        )
        loss = self.model.fit(observations_with_actions, next_observations)
        self.training_report['loss'] = loss

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
        self.make_evaluation_metrics(evaluation_trajectories)
        self.training_report.update(dict(
            eval_rl_objective=eval_return_values.mean(),
            sum_rewards_stddev=eval_return_values.std(),
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

    def make_evaluation_metrics(self, ground_truth_trajectories):
        squared_errors = []
        for evaluation_trajecty in ground_truth_trajectories:
            ground_truth_states = evaluation_trajecty['observation']
            action_sequences = evaluation_trajecty['action']
            # Breaking into horizon-length trajectories.
            ground_truth_states_split = np.array_split(
                ground_truth_states, ground_truth_states.shape[0] // self.policy.horizon, axis=0)
            action_sequences_split = np.array_split(
                action_sequences, action_sequences.shape[0] // self.policy.horizon, axis=0)
            for (states_split, actions_split) in zip(ground_truth_states_split, action_sequences_split):
                start_state = np.tile(states_split[0, ...], (5, 1))
                action_sequence = np.tile(actions_split, (5, 1, 1))
                predicted_states = self.model.simulate_trajectories(
                    start_state, action_sequence).reshape(
                    (5, states_split.shape[0] + 1, states_split.shape[1]))
                predicted_states = predicted_states[:, :-1, :]
                squared_errors.append((predicted_states.mean(axis=0) - states_split) ** 2)
        self.training_report['mse'] = np.array(squared_errors).mean(axis=(0, 1, 2))

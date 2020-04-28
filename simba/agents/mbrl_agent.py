import numpy as np
import tensorflow.compat.v1 as tf
from simba.infrastructure.common import create_tf_session, standardize_name
from simba.infrastructure.logging_utils import logger
from simba.agents import BaseAgent
from simba.policies import CemMpc, RandomMpc
from simba.models.transition_model import TransitionModel
tf.disable_v2_behavior()


class MbrlAgent(BaseAgent):
    def __init__(self,
                 seed,
                 observation_space_dim,
                 action_space_dim,
                 warmup_timesteps,
                 train_batch_size,
                 train_interaction_steps,
                 eval_interaction_steps,
                 episode_length,
                 replay_buffer_size,
                 **kwargs
                 ):
        self._sess = create_tf_session(tf.config.list_physical_devices('GPU'))
        super().__init__(
            seed,
            replay_buffer_size)
        self.observation_space_dim = observation_space_dim
        self.actions_space_dim = action_space_dim
        self.train_batch_size = train_batch_size
        self.train_interaction_steps = train_interaction_steps
        self.eval_batch_size = eval_interaction_steps
        self.episode_length = episode_length
        self.warmup_policy = self._make_policy('random_mpc', kwargs['policy_params'])
        self.warmup_timesteps = warmup_timesteps
        self.total_warmup_timesteps_so_far = 0
        assert all(key in kwargs.keys() for key in ('policy', 'policy_params', 'model', 'model_params')), \
            "Did not specify a policy or a model."
        self.model = self._make_model(kwargs.pop('model'), kwargs.pop('model_params'))
        self.policy = self._make_policy(kwargs.pop('policy'), kwargs.pop('policy_params'))

    def set_random_seeds(self, seed):
        if seed is not None:
            np.random.seed(seed)
            tf.set_random_seed(seed)

    @property
    def warm(self):
        return self.total_warmup_timesteps_so_far >= self.warmup_timesteps

    def update(self):
        # TODO (yarden): not sure about random data, maybe everything, maybe sample N trajectories.
        observations, actions, next_observations, _, _ = \
            self.replay_buffer.sample_random_rollouts(374)
        observations_with_actions = np.concatenate([
            observations,
            actions], axis=1
        )
        # We're fitting s_(t + 1) - s_(t) to improve numerical stability.
        self.model.fit(observations_with_actions, next_observations)

    def report(self):
        report = dict()
        return report

    # TODO (yarden): In MB we need not only to sample trajectories with the
    #  environment but also score each trajectory with it => assign the environment
    #  to the policy. Or a better way: the policy should get the relevant scoring
    #  function in its constructor.
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
            samples, _ = self.sample_trajectories(
                environment,
                self.policy,
                self.train_interaction_steps,
                self.episode_length
            )
        assert samples is not None, "Didn't sample anything."
        return samples

    def _build(self):
        self.model.build()
        self.policy.build()
        self._sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('logs', self._sess.graph)
        writer.close()
        logger.info("Done building Mbrl agent computational graph.")

    def _load(self):
        raise NotImplementedError

    def _make_policy(self, policy, policy_params):
        eval_policy = eval(standardize_name(policy))
        if eval_policy == RandomMpc:
            return RandomMpc(policy_params['environment'].action_space)
        if policy_params is None:
            return eval((standardize_name(policy)))()
        return eval((standardize_name(policy)))(model=self.model, **policy_params)

    def _make_model(self, model, model_params):
        return TransitionModel(
            sess=self._sess,
            model=model,
            observation_space_dim=self.observation_space_dim,
            action_space_dim=self.actions_space_dim,
            **model_params)


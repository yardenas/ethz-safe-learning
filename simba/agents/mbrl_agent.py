import numpy as np
import tensorflow.compat.v1 as tf
from simba.infrastructure.common import create_tf_session
from simba.agents import BaseAgent
from simba.models.transition_model import TransitionModel
from simba.infrastructure.logging_utils import logger
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
                 policy,
                 policy_parameters,
                 model,
                 model_parameters
                 ):
        self._sess = create_tf_session(tf.test.is_gpu_available())
        super().__init__(
            seed,
            observation_space_dim,
            action_space_dim,
            train_batch_size,
            train_interaction_steps,
            eval_interaction_steps,
            episode_length,
            replay_buffer_size,
            policy,
            policy_parameters,
            model,
            model_parameters)
        self.warmup_policy = None
        self.warmup_timesteps = warmup_timesteps
        self.total_warmup_timesteps_so_far = 0

    def set_random_seeds(self, seed):
        if seed is not None:
            np.random.seed(seed)
            tf.set_random_seed(seed)

    @property
    def warm(self):
        return self.total_warmup_timesteps_so_far >= self.warmup_timesteps

    def update(self):
        # TODO (yarden): not sure about random data, maybe everything, maybe sample N trajectories.
        observations, actions, _, next_observations, _ = \
            self.replay_buffer.sample_random_rollouts(374)
        observations_with_actions = np.hstack((self.replay_buffer.observations,
                                               self.replay_buffer.actions))
        # We're fitting s_(t + 1) - s_(t) to improve numerical stability.
        self.model.fit(observations_with_actions, next_observations - observations)

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
        logger.info("Done building Mbrl agent computational graph.")

    def _load(self):
        raise NotImplementedError

    def _make_policy(self, policy, policy_parameters):
        pass

    def _make_model(self, model, model_parameters):
        return TransitionModel(
            sess=self._sess,
            model=model,
            observation_space_dim=self.observation_space_dim,
            action_space_dim=self.actions_space_dim,
            model_parameters=model_parameters)


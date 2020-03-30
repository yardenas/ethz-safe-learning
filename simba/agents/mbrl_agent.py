import numpy as np
from simba.agents import BaseAgent
from simba.infrastructure.logger import logger


class MBRLAgent(BaseAgent):
    def __init__(self,
                 warmup_policy,
                 warmup_timesteps,
                 policy,
                 model,
                 **agent_kwargs):
        super().__init__(**agent_kwargs)
        self.policy = policy
        self.model = model
        self.warmup_policy = warmup_policy
        self.warmup_timesteps = warmup_timesteps
        self.total_warmup_timesteps_so_far = 0

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
        self.policy.build(self.model)
        logger.info("Done building MBRL agent computational graph.")

    def _load(self):
        raise NotImplementedError


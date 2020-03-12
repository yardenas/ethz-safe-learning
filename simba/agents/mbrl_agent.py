from simba.agents import BaseAgent
import numpy as np


class MBRLAgent(BaseAgent):
    def __init__(self,
                 warmup_policy,
                 warmup_timesteps,
                 **agent_kwargs
                 ):
        super().__init__(**agent_kwargs)
        self.warmup_policy = warmup_policy
        self.warmup_timesteps = warmup_timesteps
        self.total_warmup_timesteps_so_far = 0
        self.data_mean = None
        self.data_std = None

    @property
    def warm(self):
        return self.total_warmup_timesteps_so_far >= self.warmup_timesteps

    def update_model(self):
        # TODO (yarden): depends on model: if it is an ensemble, find out if each model
        #  in the ensemble need random data from the randomly sampled data from the replay buffer.
        #  Furthermore, maybe somehow sample in a prioritized way so that our samples are the ones with states that we
        #  didn't visit a lot (to allow exploration).
        observations, actions, _, next_observations, _ = \
            self.replay_buffer.sample_random_data(self.train_batch_size)
        model_features = np.hstack((self.replay_buffer.observations,
                                    self.replay_buffer.actions))
        self.data_mean = model_features.mean(axis=0)
        self.data_std = model_features.std(axis=0)
        self.model.fit(self._create_fit_feed_dict(
            observations,
            actions,
            None,
            next_observations,
            None))

    def update_policy(self):
        pass

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

    def _create_fit_feed_dict(
            self,
            observations,
            actions,
            rewards,
            next_observations,
            terminals):
        return dict(
            observations_ph=observations,
            actions_ph=actions,
            next_observations_ph=next_observations,
            input_mean_ph=self.data_mean,
            input_std_ph=self.data_std)

    def _create_prediction_feed_dict(
            self,
            observations,
            actions):
        pass

    def _report(self):
        NotImplementedError

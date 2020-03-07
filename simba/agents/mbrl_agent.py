from simba.agents import BaseAgent


class ModelBaseRLAgent(BaseAgent):
    def __init__(self,
                 warmup_policy,
                 warmup_timesteps,
                 **agent_kwargs
                 ):
        super().__init__(**agent_kwargs)
        self.warmup_policy = warmup_policy
        self.warmup_timesteps = warmup_timesteps
        self.total_warmup_timesteps_so_far = 0

    @property
    def warm(self):
        return self.total_warmup_timesteps_so_far >= self.warmup_timesteps

    def update_model(self):
        if self.warm:
            print("Doing some magnificent training.")

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
            self.total_training_timesteps_so_far += timesteps_this_batch
        else:
            samples, _ = self.sample_trajectories(
                environment,
                self.policy,
                self.train_batch_size,
                self.episode_length
            )
        assert samples is not None, "Didn't sample anything."
        return samples

    def _report(self):
        NotImplementedError


import simba.infrastructure.replay_buffer as rb
from simba.infrastructure.logging_utils import logger


class BaseAgent(object):
    """
    A base class for RL agents. An RL agent inherits from this
    class and implements a concrete RL algorithm.
    """
    def __init__(self,
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
                 model_parameters
                 ):
        self.observation_space_dim = observation_space_dim
        self.actions_space_dim = action_space_dim
        self.train_batch_size = train_batch_size
        self.train_interaction_steps = train_interaction_steps
        self.eval_batch_size = eval_interaction_steps
        self.episode_length = episode_length
        self.replay_buffer = rb.ReplayBuffer(replay_buffer_size)
        self.policy = self._make_policy(policy, policy_parameters)
        self.model = self._make_model(model, model_parameters)
        self.set_random_seeds(seed)
        # TODO (yarden): make this better.

    def set_random_seeds(self, seed):
        raise NotImplementedError("Random seeds function must be implemented.")

    def interact(self, environment):
        samples = self._interact(environment)
        self.replay_buffer.store(samples)

    def update(self):
        raise NotImplementedError

    def report(self):
        raise NotImplementedError

    def _interact(self, environment):
        raise NotImplementedError

    def build_graph(self, graph_dir=None):
        if graph_dir is None:
            logger.info("Building computational graph.")
            self._build()
        else:
            logger.info("Loading computational graph from {}".format(graph_dir))
            self._load()

    def _build(self):
        raise NotImplementedError

    def _load(self):
        raise NotImplementedError

    def _make_policy(self, policy, policy_parameters):
        raise NotImplementedError

    def _make_model(self, model, model_parameters):
        raise NotImplementedError

    def sample_trajectories(
            self,
            environment,
            policy,
            batch_size,
            max_trajectory_length):
        timesteps_this_batch = 0
        trajectories = []
        while timesteps_this_batch < batch_size:
            trajectories.append(self.sample_trajectory(
                environment,
                policy,
                max_trajectory_length))
            timesteps_this_batch += rb.path_length(trajectories[-1])
        return trajectories, timesteps_this_batch

    @staticmethod
    def sample_trajectory(
            self,
            environment,
            policy,
            max_trajectory_length):
        observation = environment.reset()
        observations, actions, \
            rewards, next_observations, \
            terminals, image_obs = [], [], [], [], [], []
        steps = 0
        while True:
            observations.append(observation)
            action = policy.get_action(observation)
            actions.append(action)
            observation, reward, done, _ = \
                environment.step(action)
            steps += 1
            next_observations.append(observation)
            rewards.append(reward)
            rollout_done = int((steps == max_trajectory_length)
                               or done)
            terminals.append(rollout_done)
            if rollout_done:
                break
        return rb.path_summary(
            observations,
            actions,
            rewards,
            next_observations,
            terminals
        )

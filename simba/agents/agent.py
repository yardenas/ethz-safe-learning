import tensorflow as tf
from simba.infrastructure.replay_buffer import ReplayBuffer, path_summary, path_length


class BaseAgent(object):
    """
    A base class for RL agents. An RL agent inherits from this
    class an implements a concrete RL algorithm.
    """

    def __init__(self,
                 model,
                 policy,
                 train_batch_size,
                 eval_batch_size,
                 episode_length,
                 replay_buffer_size,
                 **kwargs):
        self.model = model
        self.policy = policy
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.episode_length = episode_length
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.sess = self._make_session()
        self._build_graph()

    def interact(self, environment):
        """
        Interacts with the environment and stores the sampled
        interactions in a replay buffer.
        """
        samples = self._interact(environment)
        self.replay_buffer.store(samples)

    def update_model(self):
        raise NotImplementedError

    def update_policy(self):
        raise NotImplementedError

    def report(self):
        """
        :return: A dictionary with this iteration's report
        """
        report = dict()
        report.update(self._report())

    def say_cheese(self):
        """
        :return: Renderings of evaluation trajectories.
        """
        raise NotImplementedError

    def _interact(self, environment):
        raise NotImplementedError

    def _report(self):
        raise NotImplementedError

    def _build_graph(self):
        print("Building computational graph.")
        self.model.build(self.sess)
        self.policy.build(self.sess)

    def _make_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

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
            timesteps_this_batch += path_length(trajectories[-1])
        return trajectories, timesteps_this_batch

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
        return path_summary(
            observations,
            actions,
            rewards,
            next_observations,
            terminals
        )

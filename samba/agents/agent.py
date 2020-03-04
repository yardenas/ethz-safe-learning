from samba.infrastructure.replay_buffer import ReplayBuffer


class BaseAgent(object):
    """
    A base class for RL agents. An RL agent inherits from this
    class an implements a concrete RL algorithm.
    """
    def __init__(self, **kwargs):
        raise NotImplementedError

    def interact(self, environment):
        """
        Interacts with the environment and stores the sampled
        interactions in a replay buffer.
        :param environment: the environment to interact with.
        :return:
        """
        raise NotImplementedError

    def report(self):
        """
        :return: A dictionary with this iteration's report
        """
        raise NotImplementedError

    def say_cheese(self):
        """
        :return: Renderings of evaluation trajectories.
        """
        raise NotImplementedError

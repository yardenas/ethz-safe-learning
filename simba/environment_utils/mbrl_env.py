import gym


class MbrlEnv(gym.Wrapper):
    def __init__(self, env_name):
        environment = gym.make(env_name)
        super(MbrlEnv, self).__init__(environment)

    def get_reward(self, obs, acs, *args, **kwargs):
        raise NotImplementedError


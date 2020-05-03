
class MbrlEnv(object):
    def __init__(self):
        raise NotImplementedError

    def get_reward(self, obs, acs):
        raise NotImplementedError

    # def is_done(self, obs, acs):
    #     raise NotImplementedError

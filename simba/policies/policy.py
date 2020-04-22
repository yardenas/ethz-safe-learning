class PolicyBase(object):
    def __init__(self):
        pass

    def generate_action(self, state):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

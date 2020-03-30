
class BaseModel(object):
    """
    A base class for RL models. An RL model inherits from this class
    and implements a concrete RL model.
    """
    def __init__(self):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def fit(self, inputs, targets):
        raise NotImplementedError

    def predict(self, inputs):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError






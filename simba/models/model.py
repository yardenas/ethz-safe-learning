
class BaseModel(object):
    """
    A base class for RL models. An RL model inherits from this class
    and implements a concrete RL model.
    """
    def __init__(self,
                 sess):
        self.sess = sess

    def build(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError






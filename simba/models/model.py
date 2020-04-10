class BaseModel(object):
    def __init__(self,
                 inputs_dim,
                 outputs_dim):
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim

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


class TensorFlowBaseModel(BaseModel):
    """
    A base class for RL models. An RL model inherits from this class
    and implements a concrete RL model.
    """
    def __init__(self,
                 sess,
                 inputs_dim,
                 outputs_dim):
        super().__init__(inputs_dim, outputs_dim)
        self._sess = sess

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








class BaseModel(object):
    """
    A base class for RL models. An RL model inherits from this class
    and implements a concrete RL model.
    """
    def __init__(self,
                 sess,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalize_batch,
                 batch_size,
                 epochs,
                 learning_rate
                 ):
        self._sess = sess
        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation
        self.normalize_batch = normalize_batch
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def build(self):
        raise NotImplementedError

    def fit(self, fit_feed_dict):
        raise NotImplementedError

    def predict(self, predition_feed_dict):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError






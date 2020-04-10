from simba.infrastructure.common import standardize_name
from simba.models import TensorFlowBaseModel, MlpEnsemble


class TransitionModel(TensorFlowBaseModel):
    def __init__(self,
                 sess,
                 model,
                 model_parameters,
                 observation_space_dim,
                 action_space_dim):
        super().__init__(sess,
                         observation_space_dim + action_space_dim,
                         observation_space_dim)
        self.model = eval(standardize_name(model))(
            self._sess,
            self.inputs_dim,
            self.outputs_dim,
            **model_parameters)
        self.inputs_mean = None
        self.inputs_stddev = None

    def build(self):
        self.model.build()

    def fit(self, inputs, targets):
        self.inputs_mean = inputs.mean(axis=0)
        self.inputs_stddev = inputs.std(axis=0)
        losses = self.model.fit(
            (inputs - self.inputs_mean) / (self.inputs_stddev + 1e-8),
            targets)

    def predict(self, inputs):
        return self.model.predict(
            (inputs - self.inputs_mean) / (self.inputs_stddev + 1e-8))

    def save(self):
        pass

    def load(self):
        pass



from simba.models import BaseModel


class TransitionModel(BaseModel):
    def __init__(self,
                 model,
                 model_kwargs):
        super().__init__()
        self.model = model(**model_kwargs)
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


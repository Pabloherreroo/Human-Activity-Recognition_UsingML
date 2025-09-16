import random
import numpy as np
from .base_model import BaseModel

class Baseline(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self._model = None
        self.labels = None


    def fit(self, X, y, labels):
        self.labels = labels
        return self

    def predict(self, X):
        prediction = random.choice(self.labels)
        return np.array([prediction] * len(X))

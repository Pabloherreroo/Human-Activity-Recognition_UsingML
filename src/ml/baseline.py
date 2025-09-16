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
        self._model = self
        return self

    def predict(self, X):
        random.seed(42)
        predictions = [random.choice(self.labels) for _ in range(len(X))]
        return np.array(predictions)

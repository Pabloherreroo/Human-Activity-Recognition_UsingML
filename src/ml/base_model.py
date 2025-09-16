import numpy as np
from abc import ABC, abstractmethod # abstract base class
import joblib # save the models

class BaseModel(ABC):
    """
    Abstract Base Class for all machine learning models.
    """
    
    def __init__(self):
        self._model = None 
        print(f"Initialized an instance of {self.__class__.__name__}")

    @abstractmethod
    def fit(self, X, y, labels):
        pass 

    @abstractmethod
    def predict(self, X):
        pass 

    def save(self, file_path: str):
        if self._model is not None:
            joblib.dump(self._model, file_path)
            print(f"Model saved successfully to {file_path}")
        else:
            raise ValueError("Model has not been trained yet. Call .fit() before saving.")

    @classmethod
    def load(cls, file_path: str):
        instance = cls() # Create a new instance of the class
        instance._model = joblib.load(file_path)
        print(f"Model loaded successfully from {file_path}")
        return instance
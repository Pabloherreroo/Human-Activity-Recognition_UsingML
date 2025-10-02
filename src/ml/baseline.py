import random
import numpy as np
from .base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError


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

class RandomForest(BaseModel):
    """
    A simple baseline model using RandomForestClassifier.
    It flattens the windowed input data to make it compatible.
    """
    def __init__(self, **kwargs):
        super().__init__()
        # Store model hyperparameters, e.g., n_estimators=100
        self.kwargs = kwargs

    def fit(self, X, y, labels=None):
        """
        Trains the RandomForest model.
        
        It reshapes the 3D input X (samples, timesteps, features)
        into a 2D array (samples, timesteps * features).
        """
        print("Fitting RandomForest...")
        
        # 1. Reshape the data
        n_samples = X.shape[0]
        # Flatten the (window_size, features) into a single dimension
        X_reshaped = X.reshape(n_samples, -1)
        
        print(f"Original X shape: {X.shape}")
        print(f"Reshaped X shape for RandomForest: {X_reshaped.shape}")

        # 2. Initialize and train the model
        # We create the model instance here, inside fit()
        self._model = RandomForestClassifier(random_state=42, **self.kwargs)
        print("Training model...")
        self._model.fit(X_reshaped, y)
        
        print("Model training complete.")
        return self

    def predict(self, X):
        """
        Makes predictions using the trained model.

        It reshapes the 3D input X in the same way as the fit method.
        """
        if self._model is None:
            raise NotFittedError("Model has not been trained yet. Call .fit() before making predictions.")
        
        # 1. Reshape the data exactly as done in the .fit() method
        n_samples = X.shape[0]
        X_reshaped = X.reshape(n_samples, -1)

        # 2. Make predictions
        predictions = self._model.predict(X_reshaped)
        return predictions

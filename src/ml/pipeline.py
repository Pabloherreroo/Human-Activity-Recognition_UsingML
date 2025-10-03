from ..utils.metrics import get_accuracy
from sklearn.metrics import confusion_matrix
import numpy as np
from datetime import datetime
import os


class Pipeline():
    def __init__(self, data_loader, model):
        self.data_loader = data_loader
        self.model = model
        self.labels = None
        self.best_accuracy = self.load_best_accuracy()

    def load_best_accuracy(self):
        if not os.path.exists("results"):
            os.makedirs("results")
        try:
            with open('results/best_accuracy.txt', 'r') as f:
                return float(f.read())
        except (FileNotFoundError, ValueError):
            return 0.0

    def save_best_accuracy(self):
        with open('results/best_accuracy.txt', 'w') as f:
            f.write(str(self.best_accuracy))

    def run_pipeline(self, model_path=None, test_size=None, force_save=False):
        X_train, X_test, y_train, y_test, labels, _ = self.data_loader.get_data(test_size=test_size)
        self.labels = labels
        if model_path:
            print(f"Loading model from {model_path}")
            self.model = self.model.load(model_path)
        else:
            print("Fitting new model...")
            self.model = self.model.fit(X_train, y_train, labels)
        predictions = self.model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        accuracy = get_accuracy(cm)
        model_path = None
        if accuracy > self.best_accuracy or force_save:
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.save_best_accuracy()
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_path = f"models/{self.model.__class__.__name__}_{date_str}.joblib"
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            self.model.save(model_path)
        return cm, model_path


from .data_loader import DataLoader
from .baseline import Baseline
from .config import DATA_PATH


class Pipeline():
    def __init__(self, data_loader, model):
        self.data_loader = data_loader
        self.model = model

    def load_data(self):
        self.data_loader.load_data()

    def run_pipeline(self):
        X_train, X_test, y_train, y_test, labels, feature_names = self.data_loader.get_data()
        self.model = self.model.fit(X_train, y_train, labels)
        predictions = self.model.predict(X_test)
        return self.correct_predictions(predictions, y_test)

    def correct_predictions(self, predictions, y_test):
        correct = 0
        for pred, label in zip(predictions, y_test):
            if pred == label:
                correct += 1
        return correct, len(predictions)


if __name__ == "__main__":
    data_loader = DataLoader(DATA_PATH)
    model = Baseline()
    pipeline = Pipeline(data_loader, model)
    correct, total = pipeline.run_pipeline()
    print(f"Correct predictions: {correct} out of {total}")
    print(f"Accuracy: {correct / total * 100:.2f}%")

from .data_loader import DataLoader
from .baseline import Baseline
from .config import DATA_PATH


class Pipeline():
    def __init__(self, data_loader, model):
        self.data_loader = data_loader
        self.model = model

    def run_pipeline(self):
        X_train, X_test, y_train, y_test, labels = self.data_loader.get_data()
        self.trained.model = self.model.fit(X_train, y_train, labels)
        predictions = self.trained_model.predict(X_test)
        return predictions

    def correct_predictions(self, predictions, labels):
        pass

if __name__ == "__main__":
    data_loader = DataLoader(DATA_PATH)
    model = Baseline()
    pipeline = Pipeline(data_loader, model)
    predictions = pipeline.run_pipeline()

    print(len(predictions))
    print(predictions[0])
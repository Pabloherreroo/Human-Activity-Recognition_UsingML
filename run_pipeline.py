from src.ml.data_loader import DataLoader
from src.ml.baseline import RandomForest
from src.ml.pipeline import Pipeline
from src.ml.config import CSV_DATA_PATH
from src.utils.metrics import check_best


if __name__ == "__main__":
    model_path = "models/RandomForest_2025-09-21_14-58-35.joblib"
    data_loader = DataLoader(CSV_DATA_PATH)
    model = RandomForest()
    pipeline = Pipeline(data_loader, model)
    confusion_matrix, model_path = pipeline.run_pipeline(model_path)
    check_best(pipeline, confusion_matrix, model_path)

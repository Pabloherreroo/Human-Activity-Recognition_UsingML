from src.ml.data_loader import DataLoader
from src.ml.baseline import Baseline
from src.ml.pipeline import Pipeline
from src.ml.config import DATA_PATH
from src.utils.plot_metrics import plot_confusion_matrix
from src.utils.metrics import check_best


if __name__ == "__main__":
    data_loader = DataLoader(DATA_PATH)
    model = Baseline()
    pipeline = Pipeline(data_loader, model)
    confusion_matrix, model_path = pipeline.run_pipeline()
    check_best(pipeline, confusion_matrix, model_path)

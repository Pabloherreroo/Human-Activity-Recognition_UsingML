import argparse
from src.ml.data_loader import DataLoader
from src.ml.baseline import RandomForest
from src.ml.pipeline import Pipeline
from src.ml.config import CSV_DATA_PATH, TEST_CSV_DATA_PATH
from src.utils.metrics import check_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Human Activity Recognition pipeline')
    parser.add_argument('-t', '--test', action='store_true', help='Use test dataset (TEST_CSV_DATA_PATH) with 100%% test size')
    parser.add_argument('-n', '--new-model', action='store_true', help='Train a new model instead of loading existing one')
    args = parser.parse_args()

    data_path = TEST_CSV_DATA_PATH if args.test else CSV_DATA_PATH
    test_size = 1.0 if args.test else None

    model_path = None if args.new_model else "models/RandomForest_2025-09-21_14-58-35.joblib"
    
    data_loader = DataLoader(data_path)
    model = RandomForest()
    pipeline = Pipeline(data_loader, model)
    confusion_matrix, model_path = pipeline.run_pipeline(model_path, test_size=test_size)
    check_best(pipeline, confusion_matrix, model_path)

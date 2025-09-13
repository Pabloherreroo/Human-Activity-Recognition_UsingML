import argparse
from src.data_processing import run_processing

def train_model():
    # Placeholder for model training
    print("Training model...")
    pass

def evaluate_model():
    # Placeholder for model evaluation
    print("Evaluating model...")
    pass

if __name__ == '__main__':
    # Define paths
    DATA_DIR = 'data'
    OUTPUT_FILE = 'data/merged_data.csv'

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run parts of the data processing pipeline.")
    parser.add_argument('--no-extract', dest='extract', action='store_false', help='Skip the extraction step.')
    parser.add_argument('--no-aggregate', dest='aggregate', action='store_false', help='Skip the aggregation step.')
    parser.add_argument('--no-merge', dest='merge', action='store_false', help='Skip the merging step.')
    parser.set_defaults(extract=True, aggregate=True, merge=True)
    args = parser.parse_args()

    # Run the data processing pipeline based on arguments
    run_processing(
        DATA_DIR,
        OUTPUT_FILE,
        extract=args.extract,
        aggregate=args.aggregate,
        merge=args.merge
    )

    # Train the model
    train_model()

    # Evaluate the model
    evaluate_model()


import argparse
from src.data_processing import run_processing
from src.training import run_training

def train_model():
    # Backward-compat shim: direct to run_training with defaults
    print("Training model...")
    run_training(data_path='data/merged_data.csv')

def evaluate_model():
    # Kept for compatibility; evaluation happens inside run_training
    print("Evaluating model...")

if __name__ == '__main__':
    # Define paths
    DATA_DIR = 'data'
    OUTPUT_FILE = 'data/merged_data.csv'

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run parts of the data processing and training pipeline.")
    parser.add_argument('--mode', type=str, choices=['extract_and_merge', 'full', 'train'], required=True, help='The mode to run: data steps or training.')
    args = parser.parse_args()

    # Set processing flags based on mode
    if args.mode == 'extract_and_merge':
        extract = True
        aggregate = True
        merge = False
    elif args.mode == 'full':
        extract = False
        aggregate = False
        merge = True
    else:
        extract = False
        aggregate = False
        merge = False

    # Run the data processing pipeline based on arguments
    if args.mode in ['extract_and_merge', 'full']:
        run_processing(
            DATA_DIR,
            OUTPUT_FILE,
            extract=extract,
            aggregate=aggregate,
            merge=merge
        )

    # Training/evaluation
    if args.mode in ['full', 'train']:
        run_training(data_path=OUTPUT_FILE)


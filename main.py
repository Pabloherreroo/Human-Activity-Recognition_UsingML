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
    parser.add_argument('--mode', type=str, choices=['extract_and_merge', 'full'], required=True, help='The mode to run the processing in.')
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
    
    # Run the data processing pipeline based on arguments
    run_processing(
        DATA_DIR,
        OUTPUT_FILE,
        extract=extract,
        aggregate=aggregate,
        merge=merge
    )

    # In 'full' mode, also run model training and evaluation
    if args.mode == 'full':
        # Train the model
        train_model()

        # Evaluate the model
        evaluate_model()


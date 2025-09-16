import argparse
from src.data_processing import run_processing
from src.ml.config import DATA_PATH, OUTPUT_FILE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run parts of the data processing pipeline.")
    parser.add_argument('--prepare', action='store_true', help='Run extraction and aggregation steps.')
    parser.add_argument('--merge', action='store_true', help='Run merge step.')
    args = parser.parse_args()
    run_all = not (args.prepare or args.merge)

    run_processing(
        DATA_PATH,
        OUTPUT_FILE,
        extract=args.prepare or run_all,
        aggregate=args.prepare or run_all,
        merge=args.merge or run_all
    )


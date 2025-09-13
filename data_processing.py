import zipfile
import os
import sys
import pandas as pd
from typing import Dict, Tuple, List
from pathlib import Path

def extract_zip(zip_path, extract_to=None):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted all files to: {extract_to}")

def process_directory(directory) -> List[str]:
    extracted_dirs = []
    extract_to = directory
    with os.scandir(directory) as entries:
        files = [entry.name for entry in entries if entry.is_file()]
        for file in files:
            if file.endswith(".zip"):
                extract_to = os.path.join(directory, file[:-4])
                extract_zip(os.path.join(directory, file), extract_to)
                os.remove(os.path.join(directory, file))
                extracted_dirs.append(extract_to)
    return extracted_dirs

# Required sensor files and their output prefixes
SENSOR_FILES: Dict[str, str] = {
    "Gyroscope.csv": "gyro",
    "Accelerometer.csv": "acc",
    "Gravity.csv": "grav",
}

# The order in which sensor columns should appear (x,y,z grouped by sensor)
SENSOR_ORDER = ["Gyroscope.csv", "Accelerometer.csv", "Gravity.csv"]


def read_and_prepare(csv_path: str, prefix: str) -> Tuple[pd.DataFrame, Tuple[str, str]]:
    """Read a sensor CSV and rename its last three columns to {prefix}_{x,y,z}.

    Returns the prepared DataFrame and the names of the first two columns.
    """
    df = pd.read_csv(csv_path)
    if df.shape[1] < 5:
        raise ValueError(f"{csv_path} must have at least 5 columns, found {df.shape[1]}")

    common_cols = (df.columns[0], df.columns[1])

    df_5 = df.iloc[:, :5].copy()

    df_5.rename(columns={
        df_5.columns[2]: f"{prefix}_x",
        df_5.columns[3]: f"{prefix}_y",
        df_5.columns[4]: f"{prefix}_z",
    }, inplace=True)

    return df_5, common_cols


def process_second_level_folder(folder_path: str) -> bool:
    """Process a single second-level folder. Returns True if Aggregated.csv was written."""

    available = {name: os.path.join(folder_path, name) for name in SENSOR_FILES}
    if not all(os.path.isfile(p) for p in available.values()):
        missing = [n for n, p in available.items() if not os.path.isfile(p)]
        print(f"Skipping (missing files) {folder_path}: {missing}")
        return False

    prepared: Dict[str, pd.DataFrame] = {}
    commons: Dict[str, Tuple[str, str]] = {}

    # Read & prepare each sensor CSV
    for fname, prefix in SENSOR_FILES.items():
        fpath = available[fname]
        df, common_cols = read_and_prepare(fpath, prefix)
        prepared[fname] = df
        commons[fname] = common_cols

    # Verify the first two column names are identical across files
    common_pairs = list(commons.values())
    if not all(common_pairs[0] == pair for pair in common_pairs[1:]):
        raise ValueError(
            f"First two columns differ across files in {folder_path}: "
            f"{commons}"
        )
    common_cols = list(common_pairs[0])

    # Select columns to merge from each prepared df
    keep_cols = {
        "Gyroscope.csv": common_cols + ["gyro_x", "gyro_y", "gyro_z"],
        "Accelerometer.csv": common_cols + ["acc_x", "acc_y", "acc_z"],
        "Gravity.csv": common_cols + ["grav_x", "grav_y", "grav_z"],
    }

    base_name = SENSOR_ORDER[0]
    merged = prepared[base_name][keep_cols[base_name]]
    for name in SENSOR_ORDER[1:]:
        merged = merged.merge(
            prepared[name][keep_cols[name]],
            on=common_cols,
            how="inner",
        )

    # Final column order: common, then (gyro x,y,z), (acc x,y,z), (grav x,y,z)
    final_cols = (
        common_cols
        + ["gyro_x", "gyro_y", "gyro_z"]
        + ["acc_x", "acc_y", "acc_z"]
        + ["grav_x", "grav_y", "grav_z"]
    )
    merged = merged[final_cols]

    out_path = os.path.join(folder_path, "Aggregated.csv")
    merged.to_csv(out_path, index=False)
    print(f"Saved: {out_path} (rows={len(merged)})")
    return True


def aggregate_all(parent_folders: List[str]) -> None:
    """Walk each parent folder and process its immediate subfolders only."""
    total_written = 0
    for parent in parent_folders:
        if not os.path.isdir(parent):
            print(f"Parent folder not found: {parent}")
            continue

        with os.scandir(parent) as it:
            for entry in it:
                if entry.is_dir():
                    try:
                        if process_second_level_folder(entry.path):
                            total_written += 1
                    except Exception as e:
                        print(f"ERROR in {entry.path}: {e}")

    print(f"Done. Aggregated.csv written for {total_written} subfolder(s).")

def merge_aggregated_csvs(data_dir, output_file):
    """
    Merge all Aggregated.csv files from data folder subfolders into one combined dataset.
    The sitting-down folder contains both 'sitting_down' and 'standing_up' classes.
    Other folders will use their folder name as the class label.
    """
    
    # Define the data directory
    data_dir = Path(data_dir)
    
    # List to store all dataframes
    all_dataframes = []
    
    # Get all subdirectories in data folder
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(subdirs)} subdirectories in data folder:")
    
    for subdir in subdirs:
        aggregated_csv = subdir / 'Aggregated.csv'
        
        if aggregated_csv.exists():
            print(f"Processing: {subdir.name}/Aggregated.csv")
            
            # Read the CSV file
            df = pd.read_csv(aggregated_csv)
            
            # Check if the CSV already has a 'label' column
            if 'label' in df.columns:
                # For sitting-down folder, keep existing labels (sitting_down and standing_up)
                if subdir.name == 'sitting-down':
                    print(f"  - Keeping existing labels: {df['label'].unique()}")
                else:
                    # For other folders, replace with folder name if different
                    expected_label = subdir.name.lower().replace('-', '_')
                    if df['label'].iloc[0] != expected_label:
                        df['label'] = expected_label
                        print(f"  - Updated label to: {expected_label}")
                    else:
                        print(f"  - Label already correct: {expected_label}")
            else:
                # Add label column based on folder name
                label = subdir.name.lower().replace('-', '_')
                df['label'] = label
                print(f"  - Added label: {label}")
            
            print(f"  - Rows: {len(df)}")
            all_dataframes.append(df)
        else:
            print(f"Warning: {aggregated_csv} not found")
    
    if not all_dataframes:
        print("No Aggregated.csv files found!")
        return
    
    # Combine all dataframes
    print("\nMerging all dataframes...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Sort by time column for chronological order
    merged_df = merged_df.sort_values('time').reset_index(drop=True)
    
    # Display summary
    print(f"\nMerged dataset summary:")
    print(f"Total rows: {len(merged_df)}")
    print(f"Total columns: {len(merged_df.columns)}")
    print(f"Class distribution:")
    print(merged_df['label'].value_counts().sort_index())
    
    # Save the merged dataset
    merged_df.to_csv(output_file, index=False)
    print(f"\nMerged dataset saved as: {output_file}")
    
    return merged_df

def run_processing(
    data_dir: str,
    output_file: str,
    extract: bool = True,
    aggregate: bool = True,
    merge: bool = True,
):
    """Main function to run the complete data processing pipeline."""
    print(f"Starting data processing in: {data_dir}")

    extracted_dirs = []
    if extract:
        # Step 1: Extract zip files
        print("--- Running: Step 1: Extract zip files ---")
        extracted_dirs = process_directory(data_dir)
        print("--- Finished: Step 1 ---")

    if aggregate:
        # Step 2: Aggregate sensor data
        print("--- Running: Step 2: Aggregate sensor data ---")
        # If extraction was skipped, we need to find the directories to aggregate
        if not extracted_dirs:
            extracted_dirs = [
                os.path.join(data_dir, d.name)
                for d in os.scandir(data_dir)
                if d.is_dir()
            ]

        if extracted_dirs:
            aggregate_all(extracted_dirs)
        else:
            print("No directories found to process for aggregation.")
        print("--- Finished: Step 2 ---")

    if merge:
        # Step 3: Merge aggregated CSVs
        print("--- Running: Step 3: Merge aggregated CSVs ---")
        merge_aggregated_csvs(data_dir, output_file)
        print("--- Finished: Step 3 ---")

    print("Data processing pipeline finished.")
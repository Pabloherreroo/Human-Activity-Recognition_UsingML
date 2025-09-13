import pandas as pd
import os
from pathlib import Path

def merge_aggregated_csvs():
    """
    Merge all Aggregated.csv files from data folder subfolders into one combined dataset.
    The sitting-down folder contains both 'sitting_down' and 'standing_up' classes.
    Other folders will use their folder name as the class label.
    """
    
    # Define the data directory
    data_dir = Path('data')
    
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
    output_file = 'merged_aggregated_data.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\nMerged dataset saved as: {output_file}")
    
    return merged_df

if __name__ == "__main__":
    # Change to the script's directory to ensure relative paths work
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("Starting CSV merge process...")
    print(f"Working directory: {os.getcwd()}")
    
    merged_data = merge_aggregated_csvs()
    
    if merged_data is not None:
        print("\nMerge completed successfully!")
    else:
        print("\nMerge failed!")
from .config import CSV_DATA_PATH, PROCESSED_DATA_DIR, WINDOW_SIZE, STEP, TEST_SIZE, ALL_REQUIRED_COLUMNS, FEATURE_COLUMNS, LABEL_COLUMN
import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        # We are not loading data at init anymore
        
    def load_data(self):
        """Loads the dataset from the specified CSV file path."""
        if self.data is not None:
            return

        try:
            abs_path = os.path.abspath(self.file_path)
            print(f"Trying to load file from: {abs_path}")
            
            # Load only the required columns
            self.data = pd.read_csv(abs_path, usecols=ALL_REQUIRED_COLUMNS)
            print("File loaded successfully with required features.")
        except FileNotFoundError:
            print(f"File not found: {abs_path}")
            self.data = pd.DataFrame() # Ensure self.data is not None


    # OPTIMIZED
    def _create_windows(self, df, window_size, step):
        """
        Helper function to create time-series windows from a dataframe.
        :returns: num_windows, window_size, num_features
        """
        # Check if dataframe is empty or too small for windowing
        if len(df) < window_size:
            print(f"Warning: Dataset has {len(df)} rows, which is less than window_size {window_size}")
            return np.array([]), np.array([])
        
        # Pre-compute feature columns to avoid repeated drop operations
        feature_df = df.drop(LABEL_COLUMN, axis=1)
        feature_values = feature_df.values
        label_values = df[LABEL_COLUMN].values
        
        # Calculate number of windows to pre-allocate arrays
        num_windows = max(0, (len(df) - window_size) // step + 1)
        num_features = len(feature_df.columns)
        
        # Pre-allocate arrays for better memory efficiency
        X = np.empty((num_windows, window_size, num_features), dtype=np.float64)
        y = np.empty(num_windows, dtype=object)
        
        # Process windows in chunks to avoid memory issues
        chunk_size = 1000  # Process 1000 windows at a time
        window_idx = 0
        
        for chunk_start in range(0, len(df) - window_size, chunk_size * step):
            chunk_end = min(chunk_start + chunk_size * step, len(df) - window_size)
            
            for i in range(chunk_start, chunk_end, step):
                if window_idx >= num_windows:
                    break
                    
                # Extract window features more efficiently
                X[window_idx] = feature_values[i:i + window_size]
                
                # Get the most common label in the window
                window_labels = label_values[i:i + window_size]
                unique_labels, counts = np.unique(window_labels, return_counts=True)
                y[window_idx] = unique_labels[np.argmax(counts)]
                
                window_idx += 1
                
            if window_idx >= num_windows:
                break
        
        # Trim arrays to actual size
        X = X[:window_idx]
        y = y[:window_idx]
        
        return X, y

    def get_data(self, test_size=TEST_SIZE, window_size=WINDOW_SIZE, step=STEP, save_processed=True, load_from_saved=True):
        """
        Processes the loaded data to create non-leaky training and test sets
        with moving windows.

        Args:
            test_size (float): The proportion of the dataset to allocate to the test split for each activity.
            window_size (int): The number of timesteps in one window (e.g., 20 steps = 2 seconds if 0.1s interval).
            step (int): The number of timesteps to slide the window forward.
            save_processed (bool): Whether to save the processed data to a file.
            load_from_saved (bool): Whether to load the processed data from a file.
        
        Returns:
            tuple: A tuple containing X_train, X_test, y_train, y_test, and a list of unique labels.
        """
        # --- Dynamic Path for Processed Data ---
        filename = f"processed_w{window_size}_s{step}_t{int(test_size*100)}.npz"
        processed_data_path = os.path.join(PROCESSED_DATA_DIR, filename)

        if load_from_saved and os.path.exists(processed_data_path):
            print(f"Loading processed data from {processed_data_path}...")
            with np.load(processed_data_path, allow_pickle=True) as data:
                X_train = data['X_train']
                X_test = data['X_test']
                y_train = data['y_train']
                y_test = data['y_test']
                labels = data['labels']
                feature_names = data['feature_names']
            return X_train, X_test, y_train, y_test, labels, feature_names

        self.load_data()
        if self.data is None or self.data.empty:
            print("Data not loaded. Cannot process.")
            return None, None, None, None, [], []

        labels = self.data[LABEL_COLUMN].unique()
        feature_names = FEATURE_COLUMNS
        
        train_dfs = []
        test_dfs = []

        # --- Step 1: Temporal Split ---
        print("\nPerforming temporal split for each activity...")
        for label in labels:
            activity_df = self.data[self.data[LABEL_COLUMN] == label].copy()
            split_index = int(len(activity_df) * (1 - test_size))
            activity_train = activity_df.iloc[:split_index]
            activity_test = activity_df.iloc[split_index:]
            train_dfs.append(activity_train)
            test_dfs.append(activity_test)

        train_df = pd.concat(train_dfs).reset_index(drop=True)
        test_df = pd.concat(test_dfs).reset_index(drop=True)
        
        print(f"Training Set Shape (before windowing): {train_df.shape}")
        print(f"Test Set Shape (before windowing): {test_df.shape}")

        # --- Step 2: Create Moving Windows ---
        print("\nCreating moving windows for training and test sets...")
        X_train, y_train = self._create_windows(train_df, window_size, step)
        X_test, y_test = self._create_windows(test_df, window_size, step)
        print("Windowing complete")
        print(f"Training Set Shape (after windowing): {X_train.shape}")
        print(f"Test Set Shape (after windowing): {X_test.shape}")

        if save_processed:
            print(f"Saving processed data to {processed_data_path}...")
            np.savez(processed_data_path, 
                     X_train=X_train, X_test=X_test, 
                     y_train=y_train, y_test=y_test, 
                     labels=labels, feature_names=feature_names)

        return X_train, X_test, y_train, y_test, labels, feature_names
            
if __name__ == "__main__":
    try:
        data_loader = DataLoader(CSV_DATA_PATH)
        X_train, X_test, y_train, y_test, labels, feature_names = data_loader.get_data(
            test_size=TEST_SIZE, 
            window_size=WINDOW_SIZE, 
            step=STEP
        )

        if X_train is not None:
            num_windows = len(X_train)
            window_size = len(X_train[0])
            num_features = len(X_train[0][0])
            
            print("\n--- Data Loading and Processing Complete ---")
            print(f"Shape of X_train: {X_train.shape}")
            print(f"X_train contains {num_windows} windows, each with {window_size} steps and {num_features} features.")
            print(f"Feature names: {feature_names}")

    except NameError:
        print("\nERROR: 'CSV_DATA_PATH' is not defined.")
        print("Please create a 'config.py' file in the same directory and add the line:")
        print("CSV_DATA_PATH = 'path/to/your/data.csv'")
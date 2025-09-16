from config import DATA_PATH
import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.load_data()
        
    def load_data(self):
        """Loads the dataset from the specified CSV file path."""
        try:
            abs_path = os.path.abspath(self.file_path)
            print(f"Trying to load file from: {abs_path}")
            
            self.data = pd.read_csv(abs_path)
            print("File loaded successfully.")
        except FileNotFoundError:
            print(f"File not found: {abs_path}")
            self.data = pd.DataFrame() # Ensure self.data is not None

    def _create_windows(self, df, window_size, step):
        """
        Helper function to create time-series windows from a dataframe.
        :returns: num_windows, window_size, num_features
        """
        segments = []
        labels = []

        # sliding window
        for i in range(0, len(df) - window_size, step):
            window_features = df.drop('label', axis=1).values[i: i + window_size]
            
            # The label for the window is the mode (most common) label within that window.
            window_labels = df['label'][i: i + window_size]
            
            label = window_labels.mode()[0]
            
            segments.append(window_features)
            labels.append(label)
            
        X = np.array(segments)
        y = np.array(labels)
        
        return X, y

    def get_data(self, test_size=0.2, window_size=20, step=10):
        """
        Processes the loaded data to create non-leaky training and test sets
        with moving windows.

        Args:
            test_size (float): The proportion of the dataset to allocate to the test split for each activity.
            window_size (int): The number of timesteps in one window (e.g., 20 steps = 2 seconds if 0.1s interval).
            step (int): The number of timesteps to slide the window forward.
        
        Returns:
            tuple: A tuple containing X_train, X_test, y_train, y_test, and a list of unique labels.
        """
        if self.data is None or self.data.empty:
            print("Data not loaded. Cannot process.")
            return None, None, None, None, []

        labels = self.data['label'].unique()
        train_dfs = []
        test_dfs = []

        # --- Step 1: Temporal Split ---
        print("\nPerforming temporal split for each activity...")
        for label in labels:
            activity_df = self.data[self.data['label'] == label].copy()
            split_index = int(len(activity_df) * (1 - test_size))
            activity_train = activity_df.iloc[:split_index]
            activity_test = activity_df.iloc[split_index:]
            train_dfs.append(activity_train)
            test_dfs.append(activity_test)

        train_df = pd.concat(train_dfs).reset_index(drop=True)
        test_df = pd.concat(test_dfs).reset_index(drop=True)
        
        print(f"Combined Training Set Shape (before windowing): {train_df.shape}")
        print(f"Combined Test Set Shape (before windowing): {test_df.shape}")

        # --- Step 2: Create Moving Windows ---
        print("\nCreating moving windows for training and test sets...")
        X_train, y_train = self._create_windows(train_df, window_size, step)
        X_test, y_test = self._create_windows(test_df, window_size, step)
        
        return X_train, X_test, y_train, y_test, labels
            
if __name__ == "__main__":
    try:
        data_loader = DataLoader(DATA_PATH)
        X_train, X_test, y_train, y_test, labels = data_loader.get_data(
            test_size=0.2, 
            window_size=20, 
            step=10
        )

        num_windows = len(X_train)
        window_size = len(X_train[0])
        num_features = len(X_train[0][0])
        
        if X_train is not None:
            print("\n--- Data Loading and Processing Complete ---")
            print(f"Shape of X_train: {X_train.shape}")
            print(f"X_train contains {num_windows} windows, each with {window_size} steps and {num_features} features.")

    except NameError:
        print("\nERROR: 'DATA_PATH' is not defined.")
        print("Please create a 'config.py' file in the same directory and add the line:")
        print("DATA_PATH = 'path/to/your/data.csv'")

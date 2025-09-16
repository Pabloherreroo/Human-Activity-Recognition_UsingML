# Human Activity Recognition Using ML
This project goal is to accurately predict human activity based on mobile sensor data such as gyroscope, accelerometer and gravity. It consists of a full Data Science pipeline from data acquisition until model inference 

## Data Processing

To add a new action type, follow these steps:

1. Create a new folder for the action type inside the `data` directory.
2. Place the zip file containing the sensor data (Gyroscope.csv, Accelerometer.csv, Gravity.csv) inside the newly created folder.
3. Run the `prepare` and `merge` steps to process the new data:
   ```bash
   python preprocess.py --prepare --merge
   ```
4. Use the `csv_trimmer.py` script to label the data and remove any unwanted sections. 

<p align="center">
  <img src="images/labeling.png" width="400">
</p>

5. Run the `merge` option to generate a complete `data/merged_data.csv` with all activity data labeled and grouped by session

> Note: To ensure class balance, make sure the new activity data contains 60 seconds of clean sensor data

## Model Training

To get started with model training, you need to define a model that inherits from the `BaseModel` class inside the `src/ml/base_model.py` file with the following two core methods to **train the model** and **predict unseen data**:
 
```python
    @abstractmethod
    def fit(self, X, y, labels):
        pass 

    @abstractmethod
    def predict(self, X):
        pass 
```

Where `X` is a `np` array with shape (num_windows, window_size, num_features) and `y` is a `np` array with shape (num_windows,)

Once you have defined your model, plug it in the training pipeline (`run_pipeline.py`) using the following command:

```bash
python run_pipeline.py
```

The current **baseline score** is **14.5%** with the following confusion matrix:

<p align="center">
  <img src="results/Baseline_2025-09-16_23-19-15.png" width="400">
</p>

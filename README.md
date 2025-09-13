# Human Activity Recognition Using ML
This project goal is to accurately predict human activity based on mobile sensor data such as gyroscope, accelerometer and gravity. It consists of a full Data Science pipeline from data acquisition until model inference 

## Data Processing

To add a new action type, follow these steps:

1. Create a new folder for the action type inside the `data` directory.
2. Place the zip file containing the sensor data (Gyroscope.csv, Accelerometer.csv, Gravity.csv) inside the newly created folder.
3. Run the `extract_and_merge` mode to process the new data:
   ```bash
   python main.py --mode extract_and_merge
   ```
4. Use the `csv_trimmer.py` script to label the data and remove any unwanted sections. 

<p align="center">
  <img src="images/labeling.png" width="400">
</p>

5. Once the data is labeled and trimmed, run the `full` mode to merge all the data and train the model:
   ```bash
   python main.py --mode full
   ```
> Note: To ensure class balance, make sure the new activity data contains 60 seconds of clean sensor data

---

## Modeling (New)

Goal: simple, standardized skeleton to compare ML models and feature extractors using 1-second windows.

- Windowing: aggregate raw sensors into 1-second windows.
- Features: baseline uses simple stats (mean, std, min, max) per axis for gyro/acc/grav.
- Split: fair group-based split to avoid leakage (contiguous 30s blocks kept together across train/test).
- Models: minimal registry starting with Logistic Regression (L2) + StandardScaler. Easy to add more.
- Outputs: metrics saved to `models/baseline_results.json`.

### Quick start

- Install deps:
  ```powershell
  pip install -r requirements.txt
  ```
- Train/evaluate on merged dataset:
  ```powershell
  python main.py --mode train
  ```

### How to extend

- Add feature extractor: implement a class like `SimpleStatFeatures` in <mcfile name="features.py" path="src/features.py"></mcfile> that implements `extract(df) -> (X, y)`.
- Register models: edit <mcfile name="experiments.py" path="src/experiments.py"></mcfile> in `_get_registry()` to add a scikit-learn pipeline.
- Control split: tweak `DatasetConfig` (test_size, random_state, block seconds).

### Notes

- Dataset path used by default: `data/merged_data.csv`.
- Class distribution used (provided): climbing_stairs=6027, running=5997, sitting_down=6080, standing_up=6199, still=6207, walking=6049.
- Keep it simple: the architecture is intentionally small and modular to scale later without pain.

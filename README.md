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

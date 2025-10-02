CSV_DATA_PATH="data/merged_data.csv"
PROCESSED_DATA_DIR = "data/"
OUTPUT_FILE = 'data/merged_data.csv'

# TESTING MODEL W/NEW DATA
TEST_CSV_DATA_PATH = "test_data/merged_data.csv"
TEST_PROCESSED_DATA_DIR = "test_data/"
TEST_OUTPUT_FILE = 'test_data/merged_data.csv'

WINDOW_SIZE=20
STEP=5
TEST_SIZE=0.2

# Column definitions
SENSOR_COLUMNS = {
    'time': 'seconds_elapsed',
    'gyroscope': ['gyro_x', 'gyro_y', 'gyro_z'],
    'accelerometer': ['acc_x', 'acc_y', 'acc_z'],
    'gravity': ['grav_x', 'grav_y', 'grav_z']
}

LABEL_COLUMN = 'label'

FEATURE_COLUMNS = (
    [SENSOR_COLUMNS['time']] +
    SENSOR_COLUMNS['gyroscope'] +
    SENSOR_COLUMNS['accelerometer'] +
    SENSOR_COLUMNS['gravity']
)


ALL_REQUIRED_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]
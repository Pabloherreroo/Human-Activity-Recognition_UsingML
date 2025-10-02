import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

from src.ml.data_loader import DataLoader
from src.ml.config import CSV_DATA_PATH, WINDOW_SIZE, STEP, TEST_SIZE, SENSOR_COLUMNS


st.set_page_config(
    page_title="HAR Data Visualizer",
    page_icon="🏃",
    layout="wide"
)

# Use Streamlit's caching to load data only once. This is crucial for performance.
# The app will rerun this function only if the code inside it changes.
@st.cache_data
def load_activity_data():
    """
    Loads and processes the activity data using our DataLoader.
    This function is cached to avoid reloading on every user interaction.
    """
    try:
        data_loader = DataLoader(CSV_DATA_PATH)
        # Use the same parameters as before for consistency
        X_train, X_test, y_train, y_test, _, feature_names = data_loader.get_data(
            test_size=TEST_SIZE,
            window_size=WINDOW_SIZE,
            step=STEP
        )
        return X_train, X_test, y_train, y_test, feature_names
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, []

X_train, X_test, y_train, y_test, feature_names = load_activity_data()

if X_train is not None:
    st.sidebar.header("Visualization Controls")

    # 1. Select the dataset (Train or Test)
    dataset_choice = st.sidebar.radio(
        "Choose a dataset to explore:",
        ("Train", "Test"),
        horizontal=True
    )

    # Dynamically select the data based on user's choice
    if dataset_choice == "Train":
        data_X = X_train
        data_y = y_train
    else:
        data_X = X_test
        data_y = y_test

    # 2. Select the instance (window) index
    max_instance_index = data_X.shape[0] - 1
    instance_index = st.sidebar.number_input(
        f"Select a data instance (0 to {max_instance_index}):",
        min_value=0,
        max_value=max_instance_index,
        value=0,
        step=1
    )
    # 3. Select the sensor to plot
    sensor_options = [k for k in SENSOR_COLUMNS.keys() if k != 'time']
    sensor_choice = st.sidebar.radio(
        "Choose a sensor to plot:",
        options=sensor_options,
        horizontal=True
    )

    st.header(f"Instance {instance_index} from `{dataset_choice}` Set")

    # Get the specific data window and its corresponding label
    selected_window = data_X[instance_index]
    selected_label = data_y[instance_index]
    
    # Activity label for the selected instance
    st.metric(label="Activity Label", value=str(selected_label).replace('_', ' ').title())

    # Get the columns and indices for the chosen sensor
    sensor_cols = SENSOR_COLUMNS[sensor_choice]
    feature_names_list = list(feature_names)
    sensor_indices = [feature_names_list.index(col) for col in sensor_cols]

    # Get time values and sensor data
    time_values = selected_window[:, 0]
    sensor_data = selected_window[:, sensor_indices]

    # Create a DataFrame and melt it for plotting
    sensor_df = pd.DataFrame(sensor_data, columns=sensor_cols)
    sensor_df['Time (s)'] = time_values
    melted_df = sensor_df.melt(
        id_vars=['Time (s)'],
        value_vars=sensor_cols,
        var_name='sensor_axis',
        value_name='sensor_value'
    )

    # --- Plotting ---
    # Use Plotly Express for a clean, interactive line chart
    fig = px.line(
        melted_df,
        x="Time (s)",
        y="sensor_value",
        color='sensor_axis',
        title=f"Sensor Readings for {sensor_choice.title()}",
        markers=True,
        labels={"Time (s)": "Time in Seconds", "sensor_value": "Sensor Value"}
    )
    
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Sensor Value",
        font=dict(size=14),
        legend_title_text='Sensor Axis'
    )

    # Display the plot in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)

    # Optionally, show the raw data for the selected window
    with st.expander("Show Raw Data for this Window"):
        st.dataframe(pd.DataFrame(selected_window, columns=feature_names))

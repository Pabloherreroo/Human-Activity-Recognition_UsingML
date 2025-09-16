# src/ml/app.py

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

# Import the DataLoader we built. We assume the 'config.py' and 'data_loader.py'
# are in the same directory or accessible via the Python path.
from data_loader import DataLoader
from config import DATA_PATH

# --- Page Configuration ---
st.set_page_config(
    page_title="HAR Data Visualizer",
    page_icon="🏃",
    layout="wide"
)

# --- Caching ---
# Use Streamlit's caching to load data only once. This is crucial for performance.
# The app will rerun this function only if the code inside it changes.
@st.cache_data
def load_activity_data():
    """
    Loads and processes the activity data using our DataLoader.
    This function is cached to avoid reloading on every user interaction.
    """
    try:
        data_loader = DataLoader(DATA_PATH)
        # Use the same parameters as before for consistency
        X_train, X_test, y_train, y_test, _ = data_loader.get_data(
            test_size=0.2,
            window_size=20,
            step=10
        )
        # For simplicity, let's create generic feature names
        feature_names = [f"Feature_{i+1}" for i in range(X_train.shape[2])]
        return X_train, X_test, y_train, y_test, feature_names
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error("Please ensure 'config.py' points to the correct DATA_PATH.")
        return None, None, None, None, []

# --- Main Application ---
st.title("Human Activity Recognition - Window Visualizer")
st.markdown("""
This app visualizes the processed time-series windows from the HAR dataset.
Use the controls in the sidebar to select a specific data instance and feature to plot.
""")

# Load the data using our cached function
X_train, X_test, y_train, y_test, feature_names = load_activity_data()

# Check if data was loaded successfully before proceeding
if X_train is not None:
    # --- Sidebar for User Controls ---
    st.sidebar.header("Visualization Controls")

    # 1. Widget to select the dataset (Train or Test)
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

    # 2. Widget to select the instance (window) index
    max_instance_index = data_X.shape[0] - 1
    instance_index = st.sidebar.slider(
        f"Select a data instance (0 to {max_instance_index}):",
        0, max_instance_index, 0  # min, max, default_value
    )

    # 3. Widget to select the feature to plot
    feature_choice = st.sidebar.selectbox(
        "Select a feature to plot:",
        feature_names
    )

    # --- Main Panel for Displaying Plots and Info ---
    st.header(f"Visualizing Instance #{instance_index} from the `{dataset_choice}` Set")

    # Get the specific data window and its corresponding label
    selected_window = data_X[instance_index]
    selected_label = data_y[instance_index]
    
    # Display the activity label for the selected instance
    st.metric(label="Activity Label", value=str(selected_label).replace('_', ' ').title())

    # Get the index of the feature chosen by the user
    feature_index = feature_names.index(feature_choice)
    
    # Extract the data for the single feature across all time steps in the window
    feature_data_to_plot = selected_window[:, feature_index]
    
    # Create a simple DataFrame for plotting
    time_steps = np.arange(feature_data_to_plot.shape[0])
    plot_df = pd.DataFrame({
        "Time Step": time_steps,
        "Sensor Value": feature_data_to_plot
    })

    # --- Plotting ---
    # Use Plotly Express for a clean, interactive line chart
    fig = px.line(
        plot_df,
        x="Time Step",
        y="Sensor Value",
        title=f"Sensor Readings for '{feature_choice}'",
        markers=True, # Add dots for each reading
        labels={"Time Step": "Time Step within Window (0.1s intervals)"}
    )
    
    fig.update_layout(
        xaxis_title="Time Step in Window",
        yaxis_title="Sensor Value",
        font=dict(size=14)
    )

    # Display the plot in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)

    # Optionally, show the raw data for the selected window
    with st.expander("Show Raw Data for this Window"):
        st.dataframe(pd.DataFrame(selected_window, columns=feature_names))

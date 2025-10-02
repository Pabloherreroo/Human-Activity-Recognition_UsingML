import streamlit as st
import pandas as pd
import plotly.express as px

from src.ml.config import CSV_DATA_PATH, SENSOR_COLUMNS

st.set_page_config(
    page_title="Raw Sensor Data Visualizer",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_raw_data():
    """
    Loads the merged raw data from the specified path.
    This function is cached to avoid reloading on every interaction.
    """
    try:
        df = pd.read_csv(CSV_DATA_PATH)
        return df
    except FileNotFoundError:
        st.error(f"The data file was not found at: {CSV_DATA_PATH}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

df = load_raw_data()

if df is not None:
    st.sidebar.header("Visualization Controls")

    # 1. Select activities to visualize
    activities = df['label'].unique().tolist()
    selected_activities = st.sidebar.multiselect(
        "Choose activities to visualize:",
        options=activities,
        default=activities[0] if activities else []
    )

    # 2. Select the sensor to plot
    sensor_options = list(SENSOR_COLUMNS.keys())
    # The 'time' column is not a sensor, so we remove it
    sensor_options.remove('time')
    sensor_choice = st.sidebar.radio(
        "Choose a sensor to plot:",
        options=sensor_options,
        horizontal=True
    )

    if not selected_activities:
        st.warning("Please select at least one activity to visualize.")
    else:
        st.header(f"Raw Sensor Data for {sensor_choice.title()}")

        # Filter the DataFrame based on selected activities
        filtered_df = df[df['label'].isin(selected_activities)]

        # Get the columns for the chosen sensor
        sensor_cols = SENSOR_COLUMNS[sensor_choice]

        # Melt the DataFrame to long format for coloring lines differently
        id_vars = [SENSOR_COLUMNS['time'], 'label']
        melted_df = filtered_df.melt(
            id_vars=id_vars,
            value_vars=sensor_cols,
            var_name='sensor_axis',
            value_name='sensor_value'
        )

        # --- Plotting ---
        # Use Plotly Express for a clean, interactive line chart
        fig = px.line(
            melted_df,
            x=SENSOR_COLUMNS['time'],
            y='sensor_value',
            color='sensor_axis',  # Each sensor axis (e.g., x, y, z) gets a unique color
            line_dash='label',  # Differentiate activities with line styles
            title=f"Raw {sensor_choice.title()} Data",
            labels={"sensor_value": "Sensor Value", SENSOR_COLUMNS['time']: "Time (s)"}
        )
        
        fig.update_layout(
            xaxis_title="Time (seconds)",
            yaxis_title="Sensor Value",
            font=dict(size=14),
            legend_title_text='Sensor Axis'
        )

        # Display the plot in the Streamlit app
        st.plotly_chart(fig, use_container_width=True)

        # Optionally, show the raw data in an expander
        with st.expander("Show Raw Data Table"):
            st.dataframe(filtered_df)
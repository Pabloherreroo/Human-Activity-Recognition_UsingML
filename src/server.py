import dash
from dash.dependencies import Output, Input
from dash import dcc, html, dcc
from datetime import datetime, timedelta
import json
import plotly.graph_objs as go
from collections import deque
from flask import Flask, request

# New imports for ML
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from typing import Optional, List

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

MAX_DATA_POINTS = 1000
UPDATE_FREQ_MS = 100

# Deques for plotting
time = deque(maxlen=MAX_DATA_POINTS)
accel_x = deque(maxlen=MAX_DATA_POINTS)
accel_y = deque(maxlen=MAX_DATA_POINTS)
accel_z = deque(maxlen=MAX_DATA_POINTS)

# Raw samples buffer for real-time features (timestamp, x, y, z)
raw_samples = deque(maxlen=MAX_DATA_POINTS)

# Globals for model
_model: Optional[Pipeline] = None
_model_feature_names: Optional[List[str]] = None
_model_classes: Optional[np.ndarray] = None

# Global for current prediction
_current_prediction: str = "No prediction yet"


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not np.issubdtype(df["time"].dtype, np.datetime64):
        dt = pd.to_datetime(df["time"], unit="ns", errors="coerce")
        if dt.isna().all():
            dt = pd.to_datetime(df["time"], unit="ms", errors="coerce")
        df = df.copy()
        df["time"] = dt
    return df.set_index("time").sort_index()


def _compute_acc_features(df_idx: pd.DataFrame) -> pd.DataFrame:
    """Compute simple stats (mean, std, min, max) for acc_x/y/z.
    df_idx must be indexed by datetime and contain columns acc_x, acc_y, acc_z.
    Returns a single-row DataFrame with consistent feature names.
    """
    cols = ["acc_x", "acc_y", "acc_z"]
    agg = {}
    for c in cols:
        s = df_idx[c]
        agg[f"{c}_mean"] = s.mean()
        agg[f"{c}_std"] = s.std()
        agg[f"{c}_min"] = s.min()
        agg[f"{c}_max"] = s.max()
    return pd.DataFrame([agg])


def _train_realtime_model(data_path: str = "data/merged_data.csv") -> tuple[Pipeline, List[str], np.ndarray]:
    """Train a lightweight model using only accelerometer features with 1s windows.
    Trains on all data to maximize generalization for live inference.
    Returns: (pipeline, feature_names, classes)
    """
    df = pd.read_csv(data_path)
    needed = {"time", "label", "acc_x", "acc_y", "acc_z"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Cannot train realtime model, missing columns: {missing}")

    df_idx = _ensure_datetime_index(df)

    # Build 1s window features and majority label
    agg_map = {c: ["mean", "std", "min", "max"] for c in ["acc_x", "acc_y", "acc_z"]}
    X = df_idx[["acc_x", "acc_y", "acc_z"]].groupby(pd.Grouper(freq="1s")).agg(agg_map)
    X.columns = [f"{c}_{stat}" for c, stat in X.columns]
    y = df_idx["label"].groupby(pd.Grouper(freq="1s")).agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)

    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask].astype(str)

    # Fill NaNs with per-feature medians
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    pipe.fit(X, y)
    feature_names = list(X.columns)
    classes = pipe.named_steps["clf"].classes_
    return pipe, feature_names, classes


def _predict_from_recent_window() -> None:
    global _model, _model_feature_names, _model_classes, _current_prediction
    if _model is None or _model_feature_names is None:
        return
    if len(raw_samples) == 0:
        return

    now = raw_samples[-1][0]
    if not isinstance(now, datetime):
        return
    window_start = now - timedelta(seconds=1)

    # Collect last 1s data
    xs = [s for s in raw_samples if s[0] >= window_start]
    if len(xs) < 5:  # require minimal samples
        return

    df = pd.DataFrame({
        "time": [t for (t, x, y, z) in xs],
        "acc_x": [x for (t, x, y, z) in xs],
        "acc_y": [y for (t, x, y, z) in xs],
        "acc_z": [z for (t, x, y, z) in xs],
    })
    df_idx = df.set_index("time").sort_index()

    X_window = _compute_acc_features(df_idx)
    # Align feature order with training
    X_window = X_window.reindex(columns=_model_feature_names)
    # Fill any missing with zeros (should be none)
    X_window = X_window.fillna(0)

    if hasattr(_model, "predict_proba"):
        probs = _model.predict_proba(X_window)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = _model_classes[pred_idx]
        pred_prob = float(probs[pred_idx])
        _current_prediction = f"{pred_label} (p={pred_prob:.3f})"
        print(f"Predicted activity: {_current_prediction}")
    else:
        pred_label = _model.predict(X_window)[0]
        _current_prediction = str(pred_label)
        print(f"Predicted activity: {_current_prediction}")


app.layout = html.Div(
	[
		dcc.Markdown(
			children="""
			# Live Sensor Readings
			Streamed from Sensor Logger: tszheichoi.com/sensorlogger
		"""
		),
		dcc.Graph(id="live_graph"),
		dcc.Interval(id="counter", interval=UPDATE_FREQ_MS),
	]
)


@app.callback(Output("live_graph", "figure"), Input("counter", "n_intervals"))
def update_graph(_counter):
	global _current_prediction
	data = [
		go.Scatter(x=list(time), y=list(d), name=name)
		for d, name in zip([accel_x, accel_y, accel_z], ["X", "Y", "Z"])
	]

	graph = {
		"data": data,
		"layout": go.Layout(
			{
				"title": f"Live Accelerometer Data - Current Activity: {_current_prediction}",
				"xaxis": {"type": "date"},
				"yaxis": {"title": "Acceleration ms<sup>-2</sup>"},
			}
		),
	}
	if (
		len(time) > 0
	):  #  cannot adjust plot ranges until there is at least one data point
		graph["layout"]["xaxis"]["range"] = [min(time), max(time)]
		graph["layout"]["yaxis"]["range"] = [
			min(accel_x + accel_y + accel_z),
			max(accel_x + accel_y + accel_z),
		]

	return graph


@server.route("/data", methods=["POST"])
def data():  # listens to the data streamed from the sensor logger
	if str(request.method) == "POST":
		# print(f'received data: {request.data}')
		data = json.loads(request.data)
		for d in data['payload']:
			if (
				d.get("name", None) == "accelerometer"
			):  #  modify to access different sensors
				ts = datetime.fromtimestamp(d["time"] / 1000000000)
				if len(time) == 0 or ts > time[-1]:
					time.append(ts)
					# modify the following based on which sensor is accessed, log the raw json for guidance
					accel_x.append(d["values"]["x"])
					accel_y.append(d["values"]["y"])
					accel_z.append(d["values"]["z"])
					# push raw sample for prediction
					raw_samples.append((ts, d["values"]["x"], d["values"]["y"], d["values"]["z"]))
					_predict_from_recent_window()
	return "success"


if __name__ == "__main__":
	# Train the realtime model at startup
	try:
		_model, _model_feature_names, _model_classes = _train_realtime_model("data/merged_data.csv")
		print(f"Realtime model trained. Classes: {_model_classes.tolist()}")
	except Exception as e:
		print(f"Failed to train realtime model: {e}")

	app.run(port=8000, host="0.0.0.0")
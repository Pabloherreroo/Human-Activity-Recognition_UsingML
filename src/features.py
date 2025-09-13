from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


EXPECTED_SENSOR_COLS = [
    "gyro_x", "gyro_y", "gyro_z",
    "acc_x", "acc_y", "acc_z",
    "grav_x", "grav_y", "grav_z",
]


class BaseFeatureExtractor:
    """Interface for feature extractors.

    Implement extract(df) -> (X, y) that returns a feature matrix X (DataFrame)
    built from 1-second windows and the corresponding labels y (Series).
    """

    def extract(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        raise NotImplementedError


@dataclass
class SimpleStatFeatures(BaseFeatureExtractor):
    """Compute simple statistics per 1-second window for each axis.

    Stats: mean, std, min, max
    """

    time_col: str = "time"
    label_col: str = "label"
    freq: str = "1s"  # 1 second

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if not np.issubdtype(df[self.time_col].dtype, np.datetime64):
            # Try to parse as ns epoch first (dataset uses ns)
            dt = pd.to_datetime(df[self.time_col], unit="ns", errors="coerce")
            if dt.isna().all():
                # Fallback to ms if needed
                dt = pd.to_datetime(df[self.time_col], unit="ms", errors="coerce")
            df = df.copy()
            df[self.time_col] = dt
        return df.set_index(self.time_col).sort_index()

    def _majority_label(self, s: pd.Series) -> str:
        if s.empty:
            return np.nan
        mode = s.mode()
        return mode.iloc[0] if not mode.empty else np.nan

    def extract(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        missing = [c for c in EXPECTED_SENSOR_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected sensor columns: {missing}")
        if self.label_col not in df.columns:
            raise ValueError(f"Missing label column: {self.label_col}")

        df_idx = self._ensure_datetime_index(df)

        # Aggregate features per 1-second window
        agg_map = {c: ["mean", "std", "min", "max"] for c in EXPECTED_SENSOR_COLS}
        X = df_idx[EXPECTED_SENSOR_COLS].groupby(pd.Grouper(freq=self.freq)).agg(agg_map)

        # Flatten multiindex columns
        X.columns = [f"{c}_{stat}" for c, stat in X.columns]

        # Compute window labels by majority vote
        y = df_idx[self.label_col].groupby(pd.Grouper(freq=self.freq)).apply(self._majority_label)

        # Drop windows without label
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]

        # Some windows may yield NaNs (e.g., std on single sample); fill with per-feature median
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        return X, y.astype(str)
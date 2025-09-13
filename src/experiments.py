from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

from .features import SimpleStatFeatures


@dataclass
class DatasetConfig:
    data_path: str
    test_size: float = 0.2
    random_state: int = 42
    group_by_time_block_seconds: int = 30  # keep contiguous blocks together
    enforce_all_classes_in_test: bool = True


class ExperimentRunner:
    def __init__(self, ds_cfg: DatasetConfig):
        self.ds_cfg = ds_cfg

    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self.ds_cfg.data_path)
        # Ensure expected columns exist; features extractor will validate too
        return df

    def _build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        extractor = SimpleStatFeatures()
        X, y = extractor.extract(df)

        # Build grouping to avoid leakage: contiguous blocks by time
        time_index = X.index
        ns = time_index.view('i8')  # nanoseconds since epoch
        secs = (ns // 1_000_000_000).astype(np.int64)
        blocks = (secs - secs.min()) // self.ds_cfg.group_by_time_block_seconds
        groups = pd.Series(blocks, index=time_index)
        return X, y, groups

    def _get_registry(self) -> Dict[str, Pipeline]:
        models: Dict[str, Pipeline] = {
            "logreg_l2": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200)),
            ]),
        }
        return models

    def _grouped_split_with_all_classes(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series, all_classes):
        rng = np.random.RandomState(self.ds_cfg.random_state)
        unique_groups = groups.unique()
        n_test = max(1, int(len(unique_groups) * self.ds_cfg.test_size))

        is_test = None
        for _ in range(50):
            rng.shuffle(unique_groups)
            test_groups = set(unique_groups[:n_test])
            mask = groups.isin(test_groups)
            y_test_subset = y[mask]
            if set(y_test_subset.unique()) >= set(all_classes):
                is_test = mask
                break
        if is_test is None:
            test_groups = set(unique_groups[:n_test])
            is_test = groups.isin(test_groups)
        return is_test

    def run(self) -> Dict[str, dict]:
        df = self._load()
        X, y, groups = self._build_features(df)
        all_classes = sorted(y.unique())

        is_test = self._grouped_split_with_all_classes(X, y, groups, all_classes)

        X_train, X_test = X[~is_test], X[is_test]
        y_train, y_test = y[~is_test], y[is_test]

        if self.ds_cfg.enforce_all_classes_in_test and set(y_test.unique()) != set(all_classes):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.ds_cfg.test_size, random_state=self.ds_cfg.random_state, stratify=y
            )

        results = {}
        for name, model in self._get_registry().items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(
                y_test, y_pred,
                labels=all_classes,
                target_names=all_classes,
                output_dict=True,
                zero_division=0,
            )
            cm = confusion_matrix(y_test, y_pred, labels=all_classes)

            results[name] = {
                "accuracy": float(acc),
                "report": report,
                "confusion_matrix": cm.tolist(),
                "classes": all_classes,
            }
        return results
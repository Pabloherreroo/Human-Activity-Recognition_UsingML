from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .experiments import DatasetConfig, ExperimentRunner


def run_training(data_path: str = 'data/merged_data.csv', out_dir: Optional[str] = None) -> None:
    out_dir = out_dir or 'models'
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ds_cfg = DatasetConfig(data_path=data_path)
    runner = ExperimentRunner(ds_cfg)
    results = runner.run()

    # Save metrics json
    metrics_path = Path(out_dir) / 'baseline_results.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved baseline results to {metrics_path}")
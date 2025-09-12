"""
Aggregate sensor CSVs per second-level folder.

- Looks under the parent folders: `climbing-stairs`, `walking`, `sitting-down`, `standing-up`.
- For **each immediate subfolder** (the "second folders") that contains
  `Gyroscope.csv`, `Accelerometer.csv`, and `Gravity.csv`, it:
    * verifies the first two columns have the same names across the three files,
    * renames the last three columns (x, y, z) to prefixed names per sensor,
    * merges on the first two common columns (inner join),
    * orders columns as: [common1, common2, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, grav_x, grav_y, grav_z],
    * writes `Aggregated.csv` **inside that subfolder**.

Notes:
- Only immediate subfolders are processed; deeper descendants are ignored by design.
- If a subfolder is missing any of the three required CSVs, it is skipped with a message.
"""

import os
import sys
import pandas as pd
from typing import Dict, Tuple

PARENT_FOLDERS = [
    "climbing-stairs",
    "walking",
    "sitting-down",
    "standing-up",
    "running",
    "Still"
]

# Required sensor files and their output prefixes
SENSOR_FILES: Dict[str, str] = {
    "Gyroscope.csv": "gyro",
    "Accelerometer.csv": "acc",
    "Gravity.csv": "grav",
}

# The order in which sensor columns should appear (x,y,z grouped by sensor)
SENSOR_ORDER = ["Gyroscope.csv", "Accelerometer.csv", "Gravity.csv"]


def read_and_prepare(csv_path: str, prefix: str) -> Tuple[pd.DataFrame, Tuple[str, str]]:
    """Read a sensor CSV and rename its last three columns to {prefix}_{x,y,z}.

    Returns the prepared DataFrame and the names of the first two columns.
    """
    df = pd.read_csv(csv_path)
    if df.shape[1] < 5:
        raise ValueError(f"{csv_path} must have at least 5 columns, found {df.shape[1]}")

    common_cols = (df.columns[0], df.columns[1])

    df_5 = df.iloc[:, :5].copy()

    df_5.rename(columns={
        df_5.columns[2]: f"{prefix}_x",
        df_5.columns[3]: f"{prefix}_y",
        df_5.columns[4]: f"{prefix}_z",
    }, inplace=True)

    return df_5, common_cols


def process_second_level_folder(folder_path: str) -> bool:
    """Process a single second-level folder. Returns True if Aggregated.csv was written."""

    available = {name: os.path.join(folder_path, name) for name in SENSOR_FILES}
    if not all(os.path.isfile(p) for p in available.values()):
        missing = [n for n, p in available.items() if not os.path.isfile(p)]
        print(f"Skipping (missing files) {folder_path}: {missing}")
        return False

    prepared: Dict[str, pd.DataFrame] = {}
    commons: Dict[str, Tuple[str, str]] = {}

    # Read & prepare each sensor CSV
    for fname, prefix in SENSOR_FILES.items():
        fpath = available[fname]
        df, common_cols = read_and_prepare(fpath, prefix)
        prepared[fname] = df
        commons[fname] = common_cols

    # Verify the first two column names are identical across files
    common_pairs = list(commons.values())
    if not all(common_pairs[0] == pair for pair in common_pairs[1:]):
        raise ValueError(
            f"First two columns differ across files in {folder_path}: "
            f"{commons}"
        )
    common_cols = list(common_pairs[0])

    # Select columns to merge from each prepared df
    keep_cols = {
        "Gyroscope.csv": common_cols + ["gyro_x", "gyro_y", "gyro_z"],
        "Accelerometer.csv": common_cols + ["acc_x", "acc_y", "acc_z"],
        "Gravity.csv": common_cols + ["grav_x", "grav_y", "grav_z"],
    }

    base_name = SENSOR_ORDER[0]
    merged = prepared[base_name][keep_cols[base_name]]
    for name in SENSOR_ORDER[1:]:
        merged = merged.merge(
            prepared[name][keep_cols[name]],
            on=common_cols,
            how="inner",
        )

    # Final column order: common, then (gyro x,y,z), (acc x,y,z), (grav x,y,z)
    final_cols = (
        common_cols
        + ["gyro_x", "gyro_y", "gyro_z"]
        + ["acc_x", "acc_y", "acc_z"]
        + ["grav_x", "grav_y", "grav_z"]
    )
    merged = merged[final_cols]

    out_path = os.path.join(folder_path, "Aggregated.csv")
    merged.to_csv(out_path, index=False)
    print(f"Saved: {out_path} (rows={len(merged)})")
    return True


def aggregate_all() -> None:
    """Walk each parent folder and process its immediate subfolders only."""
    total_written = 0
    for parent in PARENT_FOLDERS:
        if not os.path.isdir(parent):
            print(f"Parent folder not found: {parent}")
            continue

        with os.scandir(parent) as it:
            for entry in it:
                if entry.is_dir():
                    try:
                        if process_second_level_folder(entry.path):
                            total_written += 1
                    except Exception as e:
                        print(f"ERROR in {entry.path}: {e}")

    print(f"Done. Aggregated.csv written for {total_written} subfolder(s).")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else None
    if root:
        os.chdir(root)
    aggregate_all()

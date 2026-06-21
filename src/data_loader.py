"""
src/data_loader.py
------------------
Responsible for reading the raw CSV, performing the two-pass column drop,
and removing rows with nulls in mandatory fields.

Notebook cells faithfully reproduced:
  - pd.read_csv
  - drop ['closed_datetime', 'description', 'action_taken_timestamp',
          'data_sent_to_scita', 'data_sent_to_scita_timestamp']
  - dropna(subset=['junction_name', 'police_station', 'created_by_id'])
  - drop ['vehicle_number', 'offence_code', 'device_id', 'created_by_id',
          'center_code', 'police_station', 'updated_vehicle_number',
          'validation_status', 'validation_timestamp']
"""

import logging
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


def load_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw parking-violation CSV and return a cleaned DataFrame.

    Steps
    -----
    1. Read ``data/data_grid.csv``.
    2. Drop irrelevant operational columns (first pass).
    3. Drop rows that are missing values in mandatory columns.
    4. Drop remaining admin / identifier columns (second pass).
    5. Merge ``updated_vehicle_type`` into ``vehicle_type`` and remove it.

    Parameters
    ----------
    path:
        Absolute or relative path to the raw CSV file.
        Defaults to ``config.RAW_DATA_PATH``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for pre-processing.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at *path*.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw data not found at: {path}")

    logger.info("Reading raw data from: %s", path)
    df: pd.DataFrame = pd.read_csv(path)
    logger.info("Raw shape: %s", df.shape)

    # ------------------------------------------------------------------
    # Step 1 – First-pass column drop
    # ------------------------------------------------------------------
    first_drop = [c for c in config.INITIAL_COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=first_drop)
    logger.info("After initial column drop: %s", df.shape)

    # ------------------------------------------------------------------
    # Step 2 – Drop rows with nulls in mandatory columns
    #           (only 5 rows in the original dataset, negligible loss)
    # ------------------------------------------------------------------
    required = [c for c in config.REQUIRED_NON_NULL_COLUMNS if c in df.columns]
    df = df.dropna(subset=required)
    logger.info("After dropping mandatory-null rows: %s", df.shape)

    # ------------------------------------------------------------------
    # Step 3 – Second-pass column drop
    # ------------------------------------------------------------------
    second_drop = [c for c in config.SECONDARY_COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=second_drop)
    logger.info("After secondary column drop: %s", df.shape)

    # ------------------------------------------------------------------
    # Step 4 – Merge updated_vehicle_type into vehicle_type
    #           (validation officers may correct the original entry)
    # ------------------------------------------------------------------
    if "updated_vehicle_type" in df.columns:
        import numpy as np

        df["vehicle_type"] = np.where(
            df["updated_vehicle_type"].notna()
            & (df["updated_vehicle_type"] != df["vehicle_type"]),
            df["updated_vehicle_type"],
            df["vehicle_type"],
        )
        df = df.drop(columns=["updated_vehicle_type"])
        logger.info("Merged 'updated_vehicle_type' into 'vehicle_type'.")

    logger.info(
        "Final columns after loading: %s", df.columns.tolist()
    )
    return df.reset_index(drop=True)

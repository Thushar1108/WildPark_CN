"""
src/preprocessing.py
--------------------
Handles all datetime conversions and the derivation of ``time_to_resolve``.

Notebook cells faithfully reproduced:
  - pd.to_datetime for created_datetime / modified_datetime
  - time_to_resolve = modified_datetime - created_datetime
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw string columns into typed values and derive ``time_to_resolve``.

    Steps
    -----
    1. Parse ``created_datetime`` and ``modified_datetime`` as timezone-aware
       datetimes.
    2. Compute ``time_to_resolve`` as the timedelta between the two timestamps.
    3. Normalise ``vehicle_type`` to lowercase (required before feature
       engineering keyword matching).

    Parameters
    ----------
    df:
        DataFrame produced by :func:`src.data_loader.load_data`.

    Returns
    -------
    pd.DataFrame
        DataFrame with typed datetime columns, ``time_to_resolve`` added,
        and ``vehicle_type`` lowercased.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Step 1 – Parse datetimes
    # ------------------------------------------------------------------
    logger.info("Parsing datetime columns …")
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], utc=True)
    df["modified_datetime"] = pd.to_datetime(df["modified_datetime"], utc=True)

    # ------------------------------------------------------------------
    # Step 2 – Derive resolution duration
    # ------------------------------------------------------------------
    df["time_to_resolve"] = df["modified_datetime"] - df["created_datetime"]
    logger.info(
        "time_to_resolve range: %s → %s",
        df["time_to_resolve"].min(),
        df["time_to_resolve"].max(),
    )

    # ------------------------------------------------------------------
    # Step 3 – Normalise vehicle_type for downstream keyword matching
    # ------------------------------------------------------------------
    df["vehicle_type"] = df["vehicle_type"].str.lower().fillna("unknown")

    logger.info("Preprocessing complete. Shape: %s", df.shape)
    return df

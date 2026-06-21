"""
src/feature_engineering.py
--------------------------
Executes the 7 feature-engineering steps that transform pre-processed data
into model-ready features with an ``impact_score`` for each violation record.

Each step is implemented as a pure function that accepts a DataFrame and
returns the augmented DataFrame, making unit-testing straightforward.

All weight constants and thresholds are imported from ``config`` so they
can be adjusted centrally without touching this module.
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ===========================================================================
# Step 1 – Temporal Extraction
# ===========================================================================

def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract hour-of-day, day-of-week, and peak-hour flag from
    ``created_datetime``.

    The ``is_peak_hour`` flag uses a *vectorised overlap* check (from the
    notebook) rather than a simple hour lookup, because a blockage that
    starts before a peak window but ends inside it should still be counted
    as a peak-hour event.  A record is flagged ``True`` when ANY of the
    following hold:

    * total duration ≥ ``config.LONG_DURATION_HRS`` (always-peak)
    * total duration > ``config.SIGNIFICANT_DURATION_HRS`` (very likely
      to span a peak)
    * the [created, modified] window overlaps morning peak
      ``config.PEAK_WINDOW_1``
    * the [created, modified] window overlaps evening peak
      ``config.PEAK_WINDOW_2``

    Parameters
    ----------
    df:
        DataFrame with ``created_datetime`` and ``modified_datetime`` as
        timezone-aware datetime columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new columns:
        ``hour_of_day``, ``day_of_week``, ``is_peak_hour``.
    """
    df = df.copy()

    df["hour_of_day"] = df["created_datetime"].dt.hour
    df["day_of_week"] = df["created_datetime"].dt.dayofweek

    # --- vectorised peak-overlap logic (reproduced from notebook cell 18) ---
    start_hour: pd.Series = (
        df["created_datetime"].dt.hour + df["created_datetime"].dt.minute / 60
    )
    end_hour: pd.Series = (
        (df["modified_datetime"] - df["created_datetime"].dt.normalize())
        .dt.total_seconds()
        / 3600
    )
    duration_hrs: pd.Series = (
        (df["modified_datetime"] - df["created_datetime"]).dt.total_seconds() / 3600
    )

    long_duration: pd.Series = duration_hrs >= config.LONG_DURATION_HRS
    significant_duration: pd.Series = duration_hrs > config.SIGNIFICANT_DURATION_HRS

    p1_start, p1_end = config.PEAK_WINDOW_1
    overlap_p1: pd.Series = (start_hour < p1_end) & (end_hour > p1_start)

    p2_start, p2_end = config.PEAK_WINDOW_2
    overlap_p2: pd.Series = (start_hour < p2_end) & (end_hour > p2_start)

    df["is_peak_hour"] = long_duration | overlap_p1 | overlap_p2 | significant_duration

    logger.info(
        "Temporal features added. Peak-hour records: %d / %d",
        df["is_peak_hour"].sum(),
        len(df),
    )
    return df


# ===========================================================================
# Step 2 – Duration Binning
# ===========================================================================

def _add_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ``time_to_resolve`` to numeric minutes and derive
    ``duration_category`` and ``duration_weight``.

    Bin boundaries (from ``config.DURATION_BINS``):

    * **Transient** – 0 to 15 minutes  → weight 0.5
    * **Moderate**  – 15 to 60 minutes → weight 1.0
    * **Severe**    – > 60 minutes     → weight 2.0

    NaN durations are filled with 0 (treated as Transient).

    Parameters
    ----------
    df:
        DataFrame with ``time_to_resolve`` as a timedelta column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new columns:
        ``time_to_resolve_mins``, ``duration_category``, ``duration_weight``.
    """
    df = df.copy()

    df["time_to_resolve_mins"] = (
        df["time_to_resolve"].dt.total_seconds() / 60
    ).fillna(0.0)

    df["duration_category"] = pd.cut(
        df["time_to_resolve_mins"],
        bins=config.DURATION_BINS,
        labels=config.DURATION_LABELS,
    )

    df["duration_weight"] = df["duration_category"].map(config.DURATION_WEIGHT_MAP)

    logger.info(
        "Duration categories:\n%s",
        df["duration_category"].value_counts().to_string(),
    )
    return df


# ===========================================================================
# Step 3 – Vehicle Weights
# ===========================================================================

def _add_vehicle_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map ``vehicle_type`` to ``vehicle_weight`` using regex keyword matching.

    Weight hierarchy (highest priority first):

    * Heavy vehicles (truck, lorry, bus, …) → 3.0
    * Passenger cars (car, suv, jeep, van, …) → 1.5
    * Autos / tempos (auto, rickshaw, …) → 1.0
    * Two-wheelers (motorcycle, scooter, …) → 0.5
    * Default → ``config.VEHICLE_DEFAULT_WEIGHT`` (1.0)

    ``vehicle_type`` must already be lowercased (done in preprocessing).

    Parameters
    ----------
    df:
        DataFrame with a lowercase ``vehicle_type`` column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new column: ``vehicle_weight``.
    """
    df = df.copy()

    conditions: list[pd.Series] = [
        df["vehicle_type"].str.contains(pattern, regex=True, na=False)
        for pattern, _ in config.VEHICLE_WEIGHT_PATTERNS
    ]
    weights: list[float] = [w for _, w in config.VEHICLE_WEIGHT_PATTERNS]

    df["vehicle_weight"] = np.select(
        conditions, weights, default=config.VEHICLE_DEFAULT_WEIGHT
    )

    logger.info(
        "Vehicle weight distribution:\n%s",
        df["vehicle_weight"].value_counts().to_string(),
    )
    return df


# ===========================================================================
# Step 4 – Violation Weights
# ===========================================================================

def _get_violation_weight(text: str) -> float:
    """
    Assign a violation severity weight from a raw ``violation_type`` string.

    The string is lowercased and checked for keywords in priority order
    (higher-severity rules are checked first):

    1. Pedestrian / crossing safety → 3.0
    2. Bus stop / public-transit disruption → 2.5
    3. Wrong parking / hydrant / fire hazard → 2.0
    4. All others → 1.0

    Parameters
    ----------
    text:
        Raw value from the ``violation_type`` column (may contain a JSON-like
        list of violation names as a string).

    Returns
    -------
    float
        Violation severity weight.
    """
    normalised: str = str(text).lower()
    for keywords, weight in config.VIOLATION_WEIGHT_RULES:
        if any(kw in normalised for kw in keywords):
            return weight
    return config.VIOLATION_DEFAULT_WEIGHT


def _add_violation_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply :func:`_get_violation_weight` row-wise to ``violation_type``.

    Parameters
    ----------
    df:
        DataFrame with a ``violation_type`` column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new column: ``violation_weight``.
    """
    df = df.copy()
    df["violation_weight"] = df["violation_type"].apply(_get_violation_weight)

    logger.info(
        "Violation weight distribution:\n%s",
        df["violation_weight"].value_counts().to_string(),
    )
    return df


# ===========================================================================
# Step 5 – Spatial Binning
# ===========================================================================

def _add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round coordinates to 4 decimal places to create ~11 m spatial grid cells.

    Each cell is identified by ``grid_id`` (``"lat_lon"`` string), and
    ``grid_violation_count`` records how many incidents share that cell.

    Parameters
    ----------
    df:
        DataFrame with ``latitude`` and ``longitude`` columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new columns: ``grid_id``, ``grid_violation_count``.
        Temporary rounding columns are not retained.
    """
    df = df.copy()

    lat_round: pd.Series = df["latitude"].round(config.COORD_DECIMAL_PLACES)
    lon_round: pd.Series = df["longitude"].round(config.COORD_DECIMAL_PLACES)

    df["grid_id"] = lat_round.astype(str) + "_" + lon_round.astype(str)

    grid_counts: dict[str, int] = df["grid_id"].value_counts().to_dict()
    df["grid_violation_count"] = df["grid_id"].map(grid_counts)

    logger.info("Unique grid cells: %d", df["grid_id"].nunique())
    return df


# ===========================================================================
# Step 6 – Road-Size Weights
# ===========================================================================

def _get_road_size_weight(location: object, junction_name: object) -> float:
    """
    Assign a road-size congestion weight from combined location text.

    The ``location`` and ``junction_name`` fields are concatenated, lowercased,
    and matched against keyword tiers in priority order:

    1. Narrow / complex roads (lane, narrow, market, gali, cross, …) → 3.0
    2. High-capacity roads (highway, flyover, expressway, …) → 0.5
    3. Standard roads (road, street, main, avenue) → 1.5
    4. Default → ``config.ROAD_SIZE_DEFAULT_WEIGHT`` (1.0)

    Parameters
    ----------
    location:
        Value from the ``location`` column (may be NaN).
    junction_name:
        Value from the ``junction_name`` column (may be NaN).

    Returns
    -------
    float
        Road-size weight.
    """
    loc: str = str(location).lower() if pd.notnull(location) else ""
    junc: str = str(junction_name).lower() if pd.notnull(junction_name) else ""
    text: str = loc + " " + junc

    for keywords, weight in config.ROAD_SIZE_WEIGHT_RULES:
        if any(kw in text for kw in keywords):
            return weight
    return config.ROAD_SIZE_DEFAULT_WEIGHT


def _add_road_size_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply :func:`_get_road_size_weight` row-wise to ``location`` and
    ``junction_name``.

    Parameters
    ----------
    df:
        DataFrame with ``location`` and ``junction_name`` columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new column: ``road_size_weight``.
    """
    df = df.copy()
    df["road_size_weight"] = df.apply(
        lambda row: _get_road_size_weight(row["location"], row["junction_name"]),
        axis=1,
    )

    logger.info(
        "Road size weight distribution:\n%s",
        df["road_size_weight"].value_counts().to_string(),
    )
    return df


# ===========================================================================
# Step 7 – Composite Impact Score
# ===========================================================================

def _add_impact_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the composite ``impact_score`` and sort records descending.

    Formula
    -------
    ::

        impact_score = (
            vehicle_weight
            × violation_weight
            × duration_weight
            × road_size_weight
            × peak_hour_multiplier   # 2.0 if is_peak_hour else 1.0
        )

    All intermediate weight columns are cast to ``float64`` before
    multiplication to avoid errors from Categorical dtypes.

    Parameters
    ----------
    df:
        DataFrame with all five component columns present.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``impact_score`` added, sorted in descending
        order of impact, and the index reset.
    """
    df = df.copy()

    weight_cols: list[str] = [
        "vehicle_weight",
        "violation_weight",
        "duration_weight",
        "road_size_weight",
    ]
    for col in weight_cols:
        df[col] = df[col].astype(float)

    peak_multiplier: np.ndarray = np.where(
        df["is_peak_hour"],
        config.PEAK_HOUR_MULTIPLIER,
        config.NON_PEAK_MULTIPLIER,
    )

    df["impact_score"] = (
        df["vehicle_weight"]
        * df["violation_weight"]
        * df["duration_weight"]
        * df["road_size_weight"]
        * peak_multiplier
    )

    df = df.sort_values(by="impact_score", ascending=False).reset_index(drop=True)

    logger.info(
        "Impact score – min: %.3f | max: %.3f | mean: %.3f",
        df["impact_score"].min(),
        df["impact_score"].max(),
        df["impact_score"].mean(),
    )
    return df


# ===========================================================================
# Public API
# ===========================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all 7 feature-engineering steps in sequence.

    This is the single entry-point called by ``main.py``.

    Parameters
    ----------
    df:
        Pre-processed DataFrame produced by :func:`src.preprocessing.preprocess`.

    Returns
    -------
    pd.DataFrame
        Fully enriched DataFrame with all engineered features and
        ``impact_score``, sorted descending.
    """
    logger.info("=== Feature Engineering: START ===")

    df = _add_temporal_features(df)     # Step 1
    df = _add_duration_features(df)     # Step 2
    df = _add_vehicle_weight(df)        # Step 3
    df = _add_violation_weight(df)      # Step 4
    df = _add_spatial_features(df)      # Step 5
    df = _add_road_size_weight(df)      # Step 6
    df = _add_impact_score(df)          # Step 7

    logger.info("=== Feature Engineering: DONE  === Shape: %s", df.shape)
    return df

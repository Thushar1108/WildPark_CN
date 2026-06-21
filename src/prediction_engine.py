"""
src/prediction_engine.py
------------------------
Prediction engine for the Gridlock AI-driven illegal-parking
intelligence system.

Given the processed historical DataFrame and a target future datetime,
this module returns a DataFrame of predicted hotspots ranked by
predicted average impact score.
"""

from __future__ import annotations

import warnings
from datetime import datetime, time, timedelta, timezone

import pandas as pd


def generate_predictions(
    df: pd.DataFrame,
    target_datetime: datetime,
    start_time: time,
    end_time: time
) -> pd.DataFrame:
    """Predict parking violation hotspots for the given target datetime and time window.

    Parameters
    ----------
    df:
        The fully processed historical DataFrame produced by the pipeline
        (must contain: ``day_of_week``, ``hour_of_day``, ``grid_id``,
        ``location``, ``junction_name``, ``latitude``, ``longitude``,
        ``impact_score``, ``created_datetime``).
    target_datetime:
        The future (or hypothetical) datetime for which to generate
        predictions. Only its day-of-week is used.
    start_time:
        The start of the time window.
    end_time:
        The end of the time window.

    Returns
    -------
    pd.DataFrame
        Aggregated hotspot predictions sorted by ``predicted_avg_impact``
        descending. Columns:

        - ``grid_id``
        - ``location``
        - ``latitude``, ``longitude``
        - ``record_count``          – raw historical records used
        - ``predicted_violation_count`` – sum of time-decay weights
        - ``predicted_avg_impact``  – weighted-average impact score
        - ``trend_direction``       – 'Increasing' | 'Stable' | 'Decreasing'
        - ``confidence_level``      – 'High' | 'Medium' | 'Low'

        Returns an empty DataFrame (with the same columns) when no
        historical data exists for the requested day-of-week / hour.
    """

    _RESULT_COLUMNS = [
        "grid_id",
        "location",
        "latitude",
        "longitude",
        "record_count",
        "predicted_violation_count",
        "predicted_avg_impact",
        "trend_direction",
        "confidence_level",
    ]

    # ── 1. Guard clause – required columns ──────────────────────────────────
    _required = {
        "day_of_week", "hour_of_day", "grid_id", "location",
        "junction_name", "latitude", "longitude",
        "impact_score", "created_datetime",
    }
    missing = _required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing}"
        )

    # ── 2. Target time extraction ────────────────────────────────────────────
    target_dow = target_datetime.weekday()   # 0 = Monday … 6 = Sunday
    start_hour = start_time.hour
    end_hour = end_time.hour

    # ── 3. Historical data filtering ─────────────────────────────────────────
    mask = (
        (df["day_of_week"] == target_dow) & 
        (df["hour_of_day"] >= start_hour) & 
        (df["hour_of_day"] <= end_hour)
    )
    filtered = df.loc[mask].copy()

    if filtered.empty:
        warnings.warn(
            f"No historical data found for day_of_week={target_dow}, "
            f"hours {start_hour}-{end_hour}. Returning empty predictions.",
            stacklevel=2,
        )
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    # ── 4. Parse created_datetime (handle tz-aware strings) ─────────────────
    if not pd.api.types.is_datetime64_any_dtype(filtered["created_datetime"]):
        filtered["created_datetime"] = pd.to_datetime(
            filtered["created_datetime"], utc=True, errors="coerce"
        )
    else:
        # Ensure tz-aware (UTC)
        if filtered["created_datetime"].dt.tz is None:
            filtered["created_datetime"] = filtered[
                "created_datetime"
            ].dt.tz_localize("UTC")
        else:
            filtered["created_datetime"] = filtered[
                "created_datetime"
            ].dt.tz_convert("UTC")

    # ── 5. Establish time boundaries ─────────────────────────────────────────
    max_date = filtered["created_datetime"].max()
    cutoff_recent = max_date - timedelta(days=7)
    cutoff_baseline_start = max_date - timedelta(days=28)   # 7 + 21 days back

    # ── 6. Time-decay weighting ───────────────────────────────────────────────
    filtered["time_decay_weight"] = filtered["created_datetime"].apply(
        lambda dt: 2.0 if dt > cutoff_recent else 1.0
    )

    # ── 7. Trend detection per grid ──────────────────────────────────────────
    # Recent period: last 7 days
    # Baseline period: the 21 days preceding the recent window (days -28 to -7)
    recent_mask = filtered["created_datetime"] > cutoff_recent
    baseline_mask = (
        (filtered["created_datetime"] > cutoff_baseline_start) &
        (filtered["created_datetime"] <= cutoff_recent)
    )

    recent_avg = (
        filtered.loc[recent_mask]
        .groupby("grid_id")["impact_score"]
        .mean()
        .rename("recent_avg")
    )
    baseline_avg = (
        filtered.loc[baseline_mask]
        .groupby("grid_id")["impact_score"]
        .mean()
        .rename("baseline_avg")
    )

    trend_df = pd.concat([recent_avg, baseline_avg], axis=1)

    def _classify_trend(row: pd.Series) -> str:
        r, b = row.get("recent_avg"), row.get("baseline_avg")
        if pd.isna(r) or pd.isna(b):
            return "Stable"
        diff = r - b
        if diff > 0.05 * b:           # >5 % above baseline → Increasing
            return "Increasing"
        elif diff < -0.05 * b:        # >5 % below baseline → Decreasing
            return "Decreasing"
        return "Stable"

    trend_df["trend_direction"] = trend_df.apply(_classify_trend, axis=1)

    # ── 8. Aggregation ───────────────────────────────────────────────────────
    def _weighted_avg(x: pd.DataFrame) -> float:
        scores = x["impact_score"]
        weights = x["time_decay_weight"]
        denom = weights.sum()
        return float((scores * weights).sum() / denom) if denom else float("nan")

    def _representative_location(x: pd.Series) -> str:
        """Return first non-missing, non-null location string."""
        for val in x:
            cleaned = str(val).strip()
            if cleaned and cleaned.lower() not in ("nan", "none", "null", ""):
                return cleaned
        return ""

    agg = filtered.groupby("grid_id").apply(
        lambda g: pd.Series(
            {
                "location": _representative_location(
                    g["location"].fillna(g["junction_name"])
                ),
                "latitude": g["latitude"].mean(),
                "longitude": g["longitude"].mean(),
                "record_count": len(g),
                "predicted_violation_count": g["time_decay_weight"].sum(),
                "predicted_avg_impact": _weighted_avg(g),
            }
        ),
        include_groups=False,
    ).reset_index()

    # ── 9. Merge trend direction ─────────────────────────────────────────────
    agg = agg.merge(
        trend_df["trend_direction"].reset_index(),
        on="grid_id",
        how="left",
    )
    agg["trend_direction"] = agg["trend_direction"].fillna("Stable")

    # ── 10. Confidence scoring ───────────────────────────────────────────────
    def _confidence(count: int) -> str:
        if count >= 10:
            return "High"
        elif count >= 3:
            return "Medium"
        return "Low"

    agg["confidence_level"] = agg["record_count"].astype(int).apply(_confidence)
    agg["record_count"] = agg["record_count"].astype(int)
    agg["predicted_violation_count"] = agg["predicted_violation_count"].round(1)
    agg["predicted_avg_impact"] = agg["predicted_avg_impact"].round(4)

    # ── 11. Sort and return ──────────────────────────────────────────────────
    return (
        agg[_RESULT_COLUMNS]
        .sort_values("predicted_avg_impact", ascending=False)
        .reset_index(drop=True)
    )

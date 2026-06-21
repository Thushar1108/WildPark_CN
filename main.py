"""
main.py
-------
Pipeline orchestrator for the Gridlock illegal-parking intelligence system.

Run with:
    python main.py
    python main.py --log-level DEBUG   # verbose output
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

import config
from src import load_data, preprocess, engineer_features


def _configure_logging(level: str) -> None:
    """Set up root logger with a consistent format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _print_summary(df: pd.DataFrame) -> None:
    """Print a brief summary of the final processed DataFrame."""
    logger = logging.getLogger(__name__)
    logger.info("─" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("─" * 60)
    logger.info("Final shape          : %s", df.shape)
    logger.info(
        "Impact score range   : %.3f → %.3f",
        df["impact_score"].min(),
        df["impact_score"].max(),
    )
    logger.info(
        "Top-5 highest-impact records:\n%s",
        df[["id", "vehicle_type", "impact_score"]].head(5).to_string(index=False),
    )
    logger.info(
        "Duration category breakdown:\n%s",
        df["duration_category"].value_counts().to_string(),
    )
    logger.info(
        "Unique grid cells    : %d", df["grid_id"].nunique()
    )
    logger.info("─" * 60)


def run_pipeline() -> pd.DataFrame:
    """
    Execute the end-to-end data pipeline.

    Returns
    -------
    pd.DataFrame
        The fully processed and feature-engineered DataFrame.
    """
    logger = logging.getLogger(__name__)
    t0: float = time.perf_counter()

    # ── Stage 1: Load ──────────────────────────────────────────────────
    logger.info("STAGE 1/3 — Loading data …")
    df: pd.DataFrame = load_data(config.RAW_DATA_PATH)

    # ── Stage 2: Pre-process ───────────────────────────────────────────
    logger.info("STAGE 2/3 — Pre-processing …")
    df = preprocess(df)

    # ── Stage 3: Feature Engineering ──────────────────────────────────
    logger.info("STAGE 3/3 — Feature engineering …")
    df = engineer_features(df)

    # ── Save output ────────────────────────────────────────────────────
    output_path: Path = config.PROCESSED_DATA_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Processed data saved → %s", output_path)

    elapsed: float = time.perf_counter() - t0
    logger.info("Pipeline completed in %.2f s", elapsed)

    _print_summary(df)
    return df


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(
        description="Gridlock: AI-driven illegal-parking intelligence pipeline"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO)",
    )
    args = parser.parse_args()
    _configure_logging(args.log_level)

    run_pipeline()


if __name__ == "__main__":
    main()

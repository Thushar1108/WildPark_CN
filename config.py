"""
config.py
---------
Central configuration for the Gridlock illegal-parking intelligence pipeline.
All file paths, column lists, and weight mappings are defined here so that
no magic values are scattered across the source modules.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the project (one level above this file)
PROJECT_ROOT: Path = Path(__file__).parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_PATH: Path = DATA_DIR / "data_grid.csv"
PROCESSED_DATA_PATH: Path = DATA_DIR / "processed_data.csv"

# ---------------------------------------------------------------------------
# Data-loading: columns to drop immediately after reading the CSV
# ---------------------------------------------------------------------------

# First-pass drop – columns that are entirely irrelevant to the ML task
INITIAL_COLUMNS_TO_DROP: list[str] = [
    "closed_datetime",
    "description",
    "action_taken_timestamp",
    "data_sent_to_scita",
    "data_sent_to_scita_timestamp",
]

# Second-pass drop – identifiers / admin columns not needed for modelling
SECONDARY_COLUMNS_TO_DROP: list[str] = [
    "vehicle_number",
    "offence_code",
    "device_id",
    "created_by_id",
    "center_code",
    "police_station",
    "updated_vehicle_number",
    "validation_status",
    "validation_timestamp",
]

# Rows with nulls in these columns are dropped outright (too few to impute)
REQUIRED_NON_NULL_COLUMNS: list[str] = [
    "junction_name",
    "police_station",
    "created_by_id",
]

# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------

# Peak-hour windows used for the vectorised peak-overlap check.
# A blockage is flagged is_peak_hour=True if its time window overlaps either
# of these hour ranges OR if its total duration exceeds SIGNIFICANT_DURATION_HRS.
PEAK_WINDOW_1: tuple[int, int] = (8, 12)   # Morning peak  (08:00 – 12:00)
PEAK_WINDOW_2: tuple[int, int] = (17, 22)  # Evening peak  (17:00 – 22:00)
LONG_DURATION_HRS: float = 24.0            # Always peak if duration ≥ 24 h
SIGNIFICANT_DURATION_HRS: float = 12.0    # Also peak if duration > 12 h

# ---------------------------------------------------------------------------
# Duration binning
# ---------------------------------------------------------------------------

DURATION_BINS: list[float] = [-float("inf"), 15.0, 60.0, float("inf")]
DURATION_LABELS: list[str] = ["Transient", "Moderate", "Severe"]

DURATION_WEIGHT_MAP: dict[str, float] = {
    "Transient": 0.5,
    "Moderate":  1.0,
    "Severe":    2.0,
}

# ---------------------------------------------------------------------------
# Vehicle-type weights  (keyword → weight via np.select conditions)
# ---------------------------------------------------------------------------

# Each tuple: (regex pattern, weight)
VEHICLE_WEIGHT_PATTERNS: list[tuple[str, float]] = [
    (r"truck|lorry|bus|heavy|commercial|tractor", 3.0),  # Heavy vehicles
    (r"car|suv|jeep|van|taxi cab",                1.5),  # Passenger cars
    (r"auto|rickshaw|e-rickshaw|tempo",           1.0),  # Autos / tempos
    (r"motorcycle|scooter|bike|two-wheeler",      0.5),  # Two-wheelers
]
VEHICLE_DEFAULT_WEIGHT: float = 1.0

# ---------------------------------------------------------------------------
# Violation-type weights  (keyword priority list)
# ---------------------------------------------------------------------------

# Ordered list of (keywords, weight); first match wins.
VIOLATION_WEIGHT_RULES: list[tuple[list[str], float]] = [
    (["pedestrian", "zebra", "crossing"], 3.0),          # Pedestrian safety
    (["bus", "bus bay"],                  2.5),           # Public-transit disruption
    (["wrong parking", "hydrant", "fire"], 2.0),          # Lane-capacity / hazard
]
VIOLATION_DEFAULT_WEIGHT: float = 1.0

# ---------------------------------------------------------------------------
# Spatial binning
# ---------------------------------------------------------------------------

COORD_DECIMAL_PLACES: int = 4   # ~11 m grid resolution

# ---------------------------------------------------------------------------
# Road-size weights  (keyword priority list)
# ---------------------------------------------------------------------------

# Ordered list of (keywords, weight); first match wins.
ROAD_SIZE_WEIGHT_RULES: list[tuple[list[str], float]] = [
    (["lane", "narrow", "market", "gali", "cross", "intersection", "choke"], 3.0),
    (["highway", "flyover", "expressway", "bypass", "national highway"],      0.5),
    (["road", "street", "main", "avenue"],                                    1.5),
]
ROAD_SIZE_DEFAULT_WEIGHT: float = 1.0

# ---------------------------------------------------------------------------
# Impact score
# ---------------------------------------------------------------------------

PEAK_HOUR_MULTIPLIER: float = 2.0
NON_PEAK_MULTIPLIER: float = 1.0


# ---------------------------------------------------------------------------
# Patrol Route Optimizer
# ---------------------------------------------------------------------------

import os

# MapMyIndia (Mappls) REST API key.
# Prefer the environment variable; this is a hard-coded fallback for dev only.
# Production: export MAPPLS_API_KEY=your_key  (never commit the real key)
MAPPLS_API_KEY: str = os.environ.get("MAPPLS_API_KEY", "YOUR_KEY_HERE")

# Number of patrol units available per shift.
NUM_PATROL_UNITS: int = 5

# How many top-ranked grid cells to consider for patrol allocation.
# Should be >= NUM_PATROL_UNITS so every unit gets at least one stop.
TOP_N_HOTSPOTS: int = 20

# Output path for patrol route JSON
PATROL_ROUTES_PATH: Path = DATA_DIR / "patrol_routes.json"
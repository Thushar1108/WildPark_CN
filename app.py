"""
app.py
------
Streamlit dashboard for the Gridlock AI-driven illegal-parking
intelligence system.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import ast
import re
from datetime import date, datetime, time, timedelta

import folium
from pathlib import Path
import json
import requests
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from src.prediction_engine import generate_predictions

# ── Page config (must be the very first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="Gridlock - Parking Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS overrides ────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Hide default Streamlit UI elements ── */
    header { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ── Global Font & Main Background ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"]  { 
        font-family: 'Inter', sans-serif; 
        color: #262730 !important;
    }
    
    .stApp {
        background-color: #ffffff !important;
    }

    /* ── Main panel text overrides to prevent white/invisible labels in dark theme mode ── */
    .main *, .block-container * {
        color: #262730;
    }
    .main label, .main p, .main span, .main h1, .main h2, .main h3,
    .block-container label, .block-container p, .block-container span, .block-container h1, .block-container h2, .block-container h3 {
        color: #262730 !important;
    }

    /* ── Date, Time, and Selectbox inputs styling in main panel: #659287 and white text ── */
    [data-testid="stDateInput"] div[data-baseweb="input"],
    [data-testid="stDateInput"] input,
    [data-testid="stTimeInput"] div[data-baseweb="input"],
    [data-testid="stTimeInput"] input,
    [data-testid="stTimeInput"] [data-baseweb="select"] > div,
    [data-testid="stTimeInput"] [data-baseweb="select"] span,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] select {
        background-color: #659287 !important;
        border-color: #537a70 !important;
        color: #ffffff !important;
    }

    /* ── Alert Boxes (Info, Error, Success) styling: Light Green (#B1D3B9) ── */
    [data-testid="stAlert"] {
        background-color: #B1D3B9 !important;
        color: #262730 !important;
        border: 1px solid #99c2a6 !important;
    }
    [data-testid="stAlert"] * {
        color: #262730 !important;
    }

    /* ── Sidebar: Medium Green (#88BDA4) ── */
    [data-testid="stSidebar"] {
        background-color: #88BDA4 !important;
        background-image: none !important;
        border-right: 1px solid #659287;
    }
    [data-testid="stSidebar"] * { 
        color: #ffffff !important; 
    }
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
        font-weight: 500;
    }

    /* ── Filter Input / Dropdowns: Dark Green (#659287) with white text ── */
    [data-testid="stSidebar"] .stMultiSelect div[role="button"],
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #659287 !important;
        border-color: #537a70 !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stMultiSelect span[data-baseweb="tag"] {
        background-color: #537a70 !important;
        color: #ffffff !important;
        border: 1px solid #45655c !important;
    }
    [data-testid="stSidebar"] .stMultiSelect span[data-baseweb="tag"] span {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stSlider div {
        color: #ffffff !important;
    }
    /* Sliders track styling */
    [data-testid="stSidebar"] .stSlider [data-disabled="false"] {
        background-color: #659287 !important;
    }

    /* ── KPI Cards: Light Green (#B1D3B9) with dark text ── */
    .kpi-card {
        background: #B1D3B9 !important;
        border: 1px solid #99c2a6 !important;
        border-radius: 12px !important;
        padding: 1.2rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
        height: 130px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .kpi-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
        border-color: #659287 !important;
        transform: translateY(-2px) !important;
    }
    .kpi-title {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #262730 !important;
        text-align: center !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
        transform: translateY(15px) !important;
    }
    .kpi-card:hover .kpi-title {
        font-size: 0.8rem !important;
        color: #4b5563 !important;
        transform: translateY(-8px) !important;
    }
    .kpi-content {
        opacity: 0 !important;
        max-height: 0 !important;
        transform: translateY(15px) !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-align: center !important;
        visibility: hidden !important;
    }
    .kpi-card:hover .kpi-content {
        opacity: 1 !important;
        max-height: 90px !important;
        transform: translateY(-2px) !important;
        visibility: visible !important;
    }
    .kpi-value {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1e3a27 !important;
        margin: 0 !important;
        line-height: 1.15 !important;
    }
    .kpi-sub {
        font-size: 0.72rem !important;
        color: #4b5563 !important;
        margin-top: 0.15rem !important;
    }

    /* ── Top 10 Locations Interactive Cards in 5-in-a-row Grid ── */
    .locations-grid {
        display: grid !important;
        grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
        gap: 1rem !important;
        width: 100% !important;
        margin-top: 1rem !important;
        margin-bottom: 1.5rem !important;
    }
    .location-card {
        background-color: #B1D3B9 !important;
        border: 1px solid #99c2a6 !important;
        border-radius: 12px !important;
        padding: 1.1rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        height: 195px !important;
        position: relative !important;
    }
    .location-card:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 12px 20px -8px rgba(0, 0, 0, 0.15) !important;
        border-color: #659287 !important;
        z-index: 10 !important;
    }
    .location-rank {
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        color: #659287 !important;
        margin-bottom: 0.25rem !important;
    }
    .location-address {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #262730 !important;
        margin-bottom: 0.5rem !important;
        line-height: 1.3 !important;
        display: -webkit-box !important;
        -webkit-line-clamp: 3 !important;
        -webkit-box-orient: vertical !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    .location-stats {
        display: flex !important;
        flex-direction: column !important;
        gap: 0.2rem !important;
        font-size: 0.75rem !important;
        color: #4b5563 !important;
        border-top: 1px solid rgba(0, 0, 0, 0.08) !important;
        padding-top: 0.5rem !important;
        margin-top: auto !important;
    }
    .location-stat-item strong {
        color: #1e3a27 !important;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: #1e3a27;
        border-bottom: 2px solid #b1d3b9;
        padding-bottom: 0.35rem;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
    }

    /* ── Warning / empty-state ── */
    .empty-state {
        background: #B1D3B9;
        border: 1px dashed #659287;
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        color: #262730;
        font-size: 1rem;
    }
    /* ── Patrol Route Cards ── */
    .patrol-unit-card {
        background: #B1D3B9 !important;
        border: 1px solid #99c2a6 !important;
        border-radius: 12px !important;
        padding: 1.1rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05) !important;
    }
    .patrol-unit-header {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #1e3a27 !important;
        margin-bottom: 0.5rem !important;
        border-bottom: 1px solid rgba(0,0,0,0.08) !important;
        padding-bottom: 0.4rem !important;
    }
    .patrol-stop-row {
        display: flex !important;
        justify-content: space-between !important;
        align-items: flex-start !important;
        padding: 0.35rem 0 !important;
        border-bottom: 1px solid rgba(0,0,0,0.05) !important;
        font-size: 0.78rem !important;
        color: #262730 !important;
        gap: 0.5rem !important;
    }
    .patrol-stop-num {
        font-weight: 700 !important;
        color: #659287 !important;
        min-width: 1.5rem !important;
    }
    .patrol-stop-loc {
        flex: 1 !important;
        color: #262730 !important;
        line-height: 1.3 !important;
    }
    .patrol-stop-score {
        font-weight: 600 !important;
        color: #1e3a27 !important;
        white-space: nowrap !important;
    }
    .patrol-meta {
        font-size: 0.75rem !important;
        color: #45655c !important;
        margin-top: 0.5rem !important;
        display: flex !important;
        gap: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────

if Path("data/processed_data.csv").exists():
    DATA_PATH = "data/processed_data.csv"
else:
    DATA_PATH = "https://drive.google.com/uc?id=18J21u1qfILeEZNnCPt6WNtQ1bgX6t-oj"

DAY_NAME_MAP = {
    0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu",
    4: "Fri", 5: "Sat", 6: "Sun",
}
WEEKDAY_INDICES = [0, 1, 2, 3, 4]
WEEKEND_INDICES = [5, 6]


@st.cache_data(show_spinner="Loading processed dataset ...")
def load_processed_data(path: str) -> pd.DataFrame:
    """Read processed_data.csv and return a typed DataFrame."""
    df = pd.read_csv(path)

    # Ensure correct dtypes after CSV round-trip
    for col in ("hour_of_day", "day_of_week", "grid_violation_count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in ("impact_score", "latitude", "longitude",
                "vehicle_weight", "violation_weight",
                "duration_weight", "road_size_weight"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_peak_hour"] = df["is_peak_hour"].astype(bool)
    df["is_peak_hour"] = df["is_peak_hour"].astype(bool)
    df["duration_category"] = df["duration_category"].astype(str)
    df["vehicle_type"] = df["vehicle_type"].astype(str).str.strip()
    df["junction_name"] = df["junction_name"].astype(str).str.strip()
    df["grid_id"] = df["grid_id"].astype(str)

    return df


def _extract_violation_labels(raw: str) -> list[str]:
    """Parse the JSON-like violation_type list string into individual labels."""
    try:
        items = ast.literal_eval(raw)
        return [str(i).strip().title() for i in items if str(i).strip()]
    except Exception:
        # Fallback: strip brackets/quotes and split on comma
        cleaned = re.sub(r'[\[\]"]', "", raw)
        return [p.strip().title() for p in cleaned.split(",") if p.strip()]


@st.cache_data(show_spinner="Extracting violation labels ...")
def build_violation_options(df: pd.DataFrame) -> list[str]:
    """Return sorted unique individual violation type labels."""
    labels: set[str] = set()
    for raw in df["violation_type"].dropna():
        for label in _extract_violation_labels(str(raw)):
            labels.add(label)
    return sorted(labels)


# ────────────────────────────────────────────────────────────────────────────
# Sidebar: filters
# ────────────────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render the sidebar filter panel and return the filtered DataFrame."""

    with st.sidebar:
        st.markdown(
            "<div style='font-size:1.15rem;font-weight:700;color:#ffffff;"
            "margin-bottom:1rem;'>Filters</div>",
            unsafe_allow_html=True,
        )

        # ── Hour of day ─────────────────────────────────────────────────
        st.markdown("**Hour of Day**")
        hour_range = st.slider(
            label="hour_of_day",
            min_value=0, max_value=23,
            value=(0, 23),
            label_visibility="collapsed",
        )

        # ── Day type ────────────────────────────────────────────────────
        st.markdown("**Day Type**")
        day_type_options = ["Weekdays (Mon-Fri)", "Weekends (Sat-Sun)"]
        selected_day_types = st.multiselect(
            label="day_type",
            options=day_type_options,
            default=day_type_options,
            label_visibility="collapsed",
        )
        allowed_days: list[int] = []
        if "Weekdays (Mon-Fri)" in selected_day_types:
            allowed_days += WEEKDAY_INDICES
        if "Weekends (Sat-Sun)" in selected_day_types:
            allowed_days += WEEKEND_INDICES
        if not allowed_days:
            allowed_days = list(range(7))

        # ── Impact level ────────────────────────────────────────────────
        st.markdown("**Impact Level**")
        q25 = float(df["impact_score"].quantile(0.25))
        q75 = float(df["impact_score"].quantile(0.75))
        impact_levels = st.multiselect(
            label="impact_level",
            options=["Low (<= 25th pct)", "Medium (25-75th pct)", "High (> 75th pct)"],
            default=["Low (<= 25th pct)", "Medium (25-75th pct)", "High (> 75th pct)"],
            label_visibility="collapsed",
        )

        # ── Duration category ───────────────────────────────────────────
        st.markdown("**Duration Category**")
        dur_cats = sorted(df["duration_category"].dropna().unique().tolist())
        selected_dur = st.multiselect(
            label="duration_category",
            options=dur_cats,
            default=dur_cats,
            label_visibility="collapsed",
        )
        if not selected_dur:
            selected_dur = dur_cats

        # ── Vehicle type ────────────────────────────────────────────────
        st.markdown("**Vehicle Type**")
        all_vehicles = sorted(df["vehicle_type"].dropna().unique().tolist())
        selected_vehicles = st.multiselect(
            label="vehicle_type",
            options=all_vehicles,
            default=[],
            placeholder="All vehicles",
            label_visibility="collapsed",
        )
        if not selected_vehicles:
            selected_vehicles = all_vehicles

        # ── Violation type ──────────────────────────────────────────────
        st.markdown("**Violation Type**")
        all_violations = build_violation_options(df)
        selected_violations = st.multiselect(
            label="violation_type",
            options=all_violations,
            default=[],
            placeholder="All violations",
            label_visibility="collapsed",
        )

        st.divider()
        st.caption(f"Total records loaded: **{len(df):,}**")

    # ── Apply filters ────────────────────────────────────────────────────────
    fdf = df.copy()

    # Hour of day
    fdf = fdf[fdf["hour_of_day"].between(hour_range[0], hour_range[1])]

    # Day type
    fdf = fdf[fdf["day_of_week"].isin(allowed_days)]

    # Impact level
    if impact_levels:
        masks = []
        if "Low (<= 25th pct)" in impact_levels:
            masks.append(fdf["impact_score"] <= q25)
        if "Medium (25-75th pct)" in impact_levels:
            masks.append(
                (fdf["impact_score"] > q25) & (fdf["impact_score"] <= q75)
            )
        if "High (> 75th pct)" in impact_levels:
            masks.append(fdf["impact_score"] > q75)
        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m
        fdf = fdf[combined]

    # Duration
    fdf = fdf[fdf["duration_category"].isin(selected_dur)]

    # Vehicle type
    fdf = fdf[fdf["vehicle_type"].isin(selected_vehicles)]

    # Violation type (substring match against the raw string)
    if selected_violations:
        pattern = "|".join(re.escape(v.upper()) for v in selected_violations)
        fdf = fdf[fdf["violation_type"].str.upper().str.contains(pattern, na=False)]

    return fdf.reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────
# KPI helpers
# ────────────────────────────────────────────────────────────────────────────

def _kpi_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-title">{label}</div>'
        f'<div class="kpi-content">'
        f'<div class="kpi-value">{value}</div>'
        f'{sub_html}'
        f'</div>'
        f'</div>'
    )


def _most_common_violation(df: pd.DataFrame) -> str:
    """Return the single most-frequent individual violation keyword."""
    from collections import Counter
    counter: Counter = Counter()
    for raw in df["violation_type"].dropna():
        for label in _extract_violation_labels(str(raw)):
            counter[label] += 1
    if not counter:
        return "-"
    label, count = counter.most_common(1)[0]
    # Truncate long labels for the card
    if len(label) > 22:
        label = label[:20] + "..."
    return label


# ────────────────────────────────────────────────────────────────────────────
# Map
# ────────────────────────────────────────────────────────────────────────────

# Hard cap to avoid browser slowdown when no filter is applied
HEATMAP_MAX_POINTS = 40_000


@st.cache_data(show_spinner=False)
def _build_heatmap_data(
    _df_hash: int,
    lats: tuple,
    lons: tuple,
    scores: tuple,
) -> list[list[float]]:
    """Convert coordinate + score arrays to the format HeatMap expects."""
    return [[lat, lon, score] for lat, lon, score in zip(lats, lons, scores)]


def render_map(map_data: pd.DataFrame, weight_col: str = "impact_score") -> None:
    """Render the Folium heatmap inside a Streamlit component.

    Parameters
    ----------
    map_data:
        DataFrame with at least ``latitude``, ``longitude``, and the
        column named by *weight_col*.
    weight_col:
        Column to use as the heatmap intensity weight.
        Use ``'impact_score'`` for historical mode and
        ``'predicted_avg_impact'`` for predictive mode.
    """
    required_cols = ["latitude", "longitude", weight_col]
    map_df = map_data[required_cols].dropna()

    if map_df.empty:
        st.warning("No data points to display on the map for the current selection.")
        return

    # Sample if too large
    if len(map_df) > HEATMAP_MAX_POINTS:
        map_df = map_df.sample(n=HEATMAP_MAX_POINTS, random_state=42)

    center_lat = float(map_df["latitude"].mean())
    center_lon = float(map_df["longitude"].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="CartoDB Positron",
        control_scale=True,
    )

    heat_data = [
        [row["latitude"], row["longitude"], row[weight_col]]
        for _, row in map_df.iterrows()
    ]

    HeatMap(
        heat_data,
        min_opacity=0.4,
        max_zoom=18,
        radius=15,
        blur=5,
        gradient={
            0.0: "blue",
            0.5: "lime",
            0.7: "yellow",
            1.0: "red",
        },
    ).add_to(m)

    st_folium(m, width="100%", height=500, returned_objects=[])


# ────────────────────────────────────────────────────────────────────────────
# Hotspot Location Cards
# ────────────────────────────────────────────────────────────────────────────

def render_hotspot_cards(
    data: pd.DataFrame,
    predictive_mode: bool = False,
) -> None:
    """Show the Top 10 locations as interactive HTML/CSS cards in a 5x2 grid.

    In historical mode *data* is the filtered raw DataFrame.
    In predictive mode *data* is the output of generate_predictions().
    """

    # ── Confidence / trend helpers (predictive mode only) ────────────────────
    _CONFIDENCE_COLORS = {
        "High":   ("#166534", "#dcfce7"),   # dark-green text, light-green bg
        "Medium": ("#854d0e", "#fef9c3"),   # amber text, yellow bg
        "Low":    ("#991b1b", "#fee2e2"),   # red text, light-red bg
    }
    _TREND_ICONS = {
        "Increasing": "&#9650;",   # ▲
        "Stable":     "&#8212;",   # —
        "Decreasing": "&#9660;",   # ▼
    }
    _TREND_COLORS = {
        "Increasing": "#15803d",
        "Stable":     "#64748b",
        "Decreasing": "#b91c1c",
    }

    def _badge(text: str, fg: str, bg: str) -> str:
        return (
            f'<span style="display:inline-block;padding:0.15rem 0.5rem;'
            f'border-radius:999px;font-size:0.68rem;font-weight:700;'
            f'color:{fg};background:{bg};margin-right:0.3rem;">'
            f'{text}</span>'
        )

    # ── Historical mode: aggregate raw df ───────────────────────────────────
    if not predictive_mode:
        tmp = data.copy()
        loc_s = tmp["location"].astype(str).str.strip()
        jun_s = tmp["junction_name"].astype(str).str.strip()
        _BAD = {"", "nan", "none", "null", "no junction"}
        missing = loc_s.str.lower().isin(_BAD) | loc_s.isna()
        tmp["display_location"] = loc_s.where(~missing, jun_s)
        dl_s = tmp["display_location"].astype(str).str.strip()
        still_missing = dl_s.str.lower().isin(_BAD) | dl_s.isna()
        tmp["display_location"] = dl_s.where(
            ~still_missing, "Grid ID: " + tmp["grid_id"]
        )
        grouped = (
            tmp.groupby("display_location")
            .agg(
                total_violations=("id", "count"),
                avg_impact=("impact_score", "mean"),
                max_impact=("impact_score", "max"),
            )
            .reset_index()
            .sort_values("avg_impact", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        grid_items = []
        for i, row in grouped.iterrows():
            rank = i + 1
            addr = row["display_location"]
            stats_html = (
                f'<div class="location-stat-item">Violations: <strong>{row["total_violations"]}</strong></div>'
                f'<div class="location-stat-item">Avg Impact: <strong>{row["avg_impact"]:.2f}</strong></div>'
                f'<div class="location-stat-item">Max Impact: <strong>{row["max_impact"]:.2f}</strong></div>'
            )
            grid_items.append(f"""<div class="location-card">
<div>
<div class="location-rank">Rank #{rank}</div>
<div class="location-address" title="{addr}">{addr}</div>
</div>
<div class="location-stats">{stats_html}</div>
</div>""")

    # ── Predictive mode: use engine output directly ──────────────────────────
    else:
        top10 = data.head(10).reset_index(drop=True)
        grid_items = []
        for i, row in top10.iterrows():
            rank = i + 1
            addr = str(row.get("location", "")).strip() or f'Grid ID: {row["grid_id"]}'
            confidence = str(row.get("confidence_level", "Low"))
            trend      = str(row.get("trend_direction", "Stable"))
            pred_viol  = row.get("predicted_violation_count", 0)
            pred_imp   = row.get("predicted_avg_impact", 0.0)

            conf_fg, conf_bg = _CONFIDENCE_COLORS.get(confidence, ("#374151", "#f3f4f6"))
            trend_icon  = _TREND_ICONS.get(trend, "&#8212;")
            trend_color = _TREND_COLORS.get(trend, "#64748b")

            conf_badge  = _badge(confidence, conf_fg, conf_bg)
            trend_badge = (
                f'<span style="color:{trend_color};font-weight:700;font-size:0.8rem;">'
                f'{trend_icon} {trend}</span>'
            )

            stats_html = (
                f'<div class="location-stat-item">Predicted Violations: <strong>{pred_viol:.0f}</strong></div>'
                f'<div class="location-stat-item">Avg Impact: <strong>{pred_imp:.2f}</strong></div>'
                f'<div style="margin-top:0.35rem;">{conf_badge}{trend_badge}</div>'
            )
            grid_items.append(f"""<div class="location-card">
<div>
<div class="location-rank">Rank #{rank}</div>
<div class="location-address" title="{addr}">{addr}</div>
</div>
<div class="location-stats">{stats_html}</div>
</div>""")

    full_grid_html = f'<div class="locations-grid">{"" .join(grid_items)}</div>'
    st.markdown(full_grid_html, unsafe_allow_html=True)


import json
import requests
from pathlib import Path

if Path("data/patrol_routes.json").exists():
    PATROL_ROUTES_PATH = "data/patrol_routes.json"
    with open(PATROL_ROUTES_PATH, "r") as f:
        patrol_data = json.load(f)
else:
    RAW_JSON_URL = "https://raw.githubusercontent.com/Thushar1108/WildPark_CN/main/data/patrol_routes.json"
    
    try:
        response = requests.get(RAW_JSON_URL, timeout=10)
        if response.status_code == 200:
            patrol_data = response.json()
        else:
            raise ValueError(f"Status code {response.status_code}")
    except Exception as e:
        # 🚨 Emergency hardcoded fallback so your app NEVER crashes during the demo
        print(f"Failed to fetch remote routes ({e}), loading local emergency fallback.")
        patrol_data = [
          {
            "unit_id": 0,
            "num_stops": 2,
            "total_distance_km": 8.87,
            "estimated_duration_mins": 16.7,
            "route_geometry": "",
            "waypoints": [],
            "api_resource_used": "fallback"
          }
        ]


_UNIT_COLORS = [
    "#e63946", "#2196f3", "#ff9800", "#9c27b0",
    "#009688", "#795548", "#607d8b", "#e91e63",
]


@st.cache_data(show_spinner="Loading patrol routes ...")
def load_patrol_routes(path: str) -> list[dict]:
    """Load patrol_routes.json produced by patrol_router.py."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []


def _decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """
    Decode a Google/Mappls encoded polyline string into (lat, lon) pairs.
    Uses the `polyline` package if available; falls back to empty list.
    """
    if not encoded:
        return []
    try:
        import polyline as pl
        return pl.decode(encoded)
    except ImportError:
        return []


def render_patrol_routes(routes: list[dict]) -> None:
    """
    Render the patrol route section: a Folium map with per-unit
    colour-coded routes + numbered stop markers, followed by
    per-unit waypoint cards.
    """
    if not routes:
        st.markdown(
            '<div class="empty-state">'
            "No patrol routes found. Run <code>python main.py</code> with a valid "
            "MAPPLS_API_KEY to generate routes."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Build Folium map ─────────────────────────────────────────────────────
    # Collect all waypoint coords to centre the map
    all_lats, all_lons = [], []
    for r in routes:
        for wp in r.get("waypoints", []):
            all_lats.append(wp["latitude"])
            all_lons.append(wp["longitude"])

    center_lat = sum(all_lats) / len(all_lats) if all_lats else 12.9716
    center_lon = sum(all_lons) / len(all_lons) if all_lons else 77.5946

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="CartoDB Positron",
        control_scale=True,
    )

    for r in routes:
        unit_id    = r["unit_id"]
        color      = _UNIT_COLORS[unit_id % len(_UNIT_COLORS)]
        waypoints  = r.get("waypoints", [])
        geometry   = r.get("route_geometry")

        # Draw route line — decoded polyline if available, straight lines fallback
        coords = _decode_polyline(geometry)
        if coords:
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=4,
                opacity=0.85,
                tooltip=f"Unit {unit_id}",
            ).add_to(m)
        elif len(waypoints) >= 2:
            # Straight-line fallback between waypoints
            line_coords = [[wp["latitude"], wp["longitude"]] for wp in waypoints]
            folium.PolyLine(
                locations=line_coords,
                color=color,
                weight=3,
                opacity=0.6,
                dash_array="8 4",
                tooltip=f"Unit {unit_id} (no route geometry)",
            ).add_to(m)

        # Numbered stop markers
        for wp in waypoints:
            folium.Marker(
                location=[wp["latitude"], wp["longitude"]],
                tooltip=(
                    f"Unit {unit_id} · Stop {wp['stop_order']}<br>"
                    f"{wp['location']}<br>"
                    f"Impact: {wp['total_impact_score']:.2f} | "
                    f"Violations: {wp['violation_count']}"
                ),
                icon=folium.DivIcon(
                    html=(
                        f'<div style="background:{color};color:#fff;'
                        f'border-radius:50%;width:24px;height:24px;'
                        f'display:flex;align-items:center;justify-content:center;'
                        f'font-size:11px;font-weight:700;border:2px solid #fff;'
                        f'box-shadow:0 1px 4px rgba(0,0,0,0.3);">'
                        f'{wp["stop_order"]}</div>'
                    ),
                    icon_size=(24, 24),
                    icon_anchor=(12, 12),
                ),
            ).add_to(m)

    st_folium(m, width="100%", height=500, returned_objects=[])

    # ── Per-unit waypoint cards ──────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.85rem;color:#45655c;"
        "margin-top:1rem;margin-bottom:0.75rem;'>"
        "Each unit's stops are ordered by impact score. "
        "Distance and ETA are traffic-aware estimates from Mappls."
        "</div>",
        unsafe_allow_html=True,
    )

    # Lay out units in two columns
    cols = st.columns(2)
    for i, r in enumerate(routes):
        unit_id  = r["unit_id"]
        color    = _UNIT_COLORS[unit_id % len(_UNIT_COLORS)]
        dist     = f"{r['total_distance_km']:.1f} km" if r["total_distance_km"] else "N/A"
        eta      = f"{r['estimated_duration_mins']:.0f} min" if r["estimated_duration_mins"] else "N/A"
        resource = r.get("api_resource_used", "fallback")
        stops    = r.get("waypoints", [])

        stop_rows_html = "".join(
            f'<div class="patrol-stop-row">'
            f'<span class="patrol-stop-num">{wp["stop_order"]}.</span>'
            f'<span class="patrol-stop-loc">{wp["location"]}</span>'
            f'<span class="patrol-stop-score">⚡ {wp["total_impact_score"]:.2f}</span>'
            f'</div>'
            for wp in stops
        )

        traffic_note = (
            "🟢 Traffic-aware" if resource == "route_eta"
            else "🟡 Basic route" if resource == "route"
            else "⚪ No route (API unavailable)"
        )

        card_html = (
            f'<div class="patrol-unit-card">'
            f'<div class="patrol-unit-header">'
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'border-radius:50%;background:{color};margin-right:0.5rem;'
            f'vertical-align:middle;"></span>'
            f'Patrol Unit {unit_id} &nbsp;·&nbsp; {len(stops)} stops'
            f'</div>'
            f'{stop_rows_html}'
            f'<div class="patrol-meta">'
            f'<span>📍 {dist}</span>'
            f'<span>⏱ {eta}</span>'
            f'<span>{traffic_note}</span>'
            f'</div>'
            f'</div>'
        )

        with cols[i % 2]:
            st.markdown(card_html, unsafe_allow_html=True)
# ────────────────────────────────────────────────────────────────────────────
# Main layout
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load data ────────────────────────────────────────────────────────────
    try:
        df = load_processed_data(DATA_PATH)
    except FileNotFoundError:
        st.error(
            "processed_data.csv not found. "
            "Run python main.py first to generate the processed dataset."
        )
        st.stop()

    # ── Page header: Large font style with an underline like Vaishno Devi Traders ──
    st.markdown(
        """
        <div style="padding-top: 1.5rem; padding-bottom: 0.75rem; border-bottom: 2px solid #b1d3b9; margin-bottom: 2rem;">
            <h1 style="font-size: 2.8rem; font-weight: 800; color: #1e3a27; margin: 0; padding: 0; font-family: 'Inter', sans-serif; letter-spacing: -0.02em;">
                Gridlock Parking Intelligence
            </h1>
            <p style="font-size: 1.15rem; color: #45655c; margin: 0.3rem 0 0 0; padding: 0; font-family: 'Inter', sans-serif;">
                AI-driven illegal parking hotspot analysis &middot; Bengaluru Traffic Authority
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar filters ──────────────────────────────────────────────────────
    fdf = render_sidebar(df)

    # ── Empty-state guard ────────────────────────────────────────────────────
    if fdf.empty:
        st.markdown(
            '<div class="empty-state">'
            "No records match the current filter combination. "
            "Please adjust your filters."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Section 1: KPI row ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Key Performance Indicators</div>',
                unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)

    total = len(fdf)
    avg_impact = fdf["impact_score"].mean()
    most_violation = _most_common_violation(fdf)
    most_vehicle = fdf["vehicle_type"].value_counts().idxmax() if total else "-"
    if len(most_vehicle) > 20:
        most_vehicle = most_vehicle[:18] + "..."

    pct_of_total = total / len(df) * 100

    with k1:
        st.markdown(
            _kpi_card(
                "Total Filtered Violations",
                f"{total:,}",
                f"{pct_of_total:.1f}% of full dataset",
            ),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            _kpi_card(
                "Average Impact Score",
                f"{avg_impact:.2f}",
                f"Max: {fdf['impact_score'].max():.1f}",
            ),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            _kpi_card(
                "Top Violation Type",
                most_violation,
                "Most frequent category",
            ),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            _kpi_card(
                "Top Vehicle Type",
                most_vehicle.title(),
                f"{fdf['vehicle_type'].value_counts().iloc[0]:,} occurrences",
            ),
            unsafe_allow_html=True,
        )

    # ── Section 2: Predictive Mode Controls + Hotspot Heatmap ──────────────

    # --- PREDICTIVE MODE CONTROLS ---
    st.markdown("### Data View Mode")
    ctl1, ctl2 = st.columns([1, 3])

    with ctl1:
        st.markdown("""
        <style>
        label:has(input[aria-label="Predictive Forecast"]) > div {
            background-color: #b1d3b9 !important;
            border: 2px solid #659287 !important;
        }
        label:has(input[aria-label="Predictive Forecast"]:checked) > div {
            background-color: #659287 !important;
            border: 2px solid #537a70 !important;
        }
        label:has(input[aria-label="Predictive Forecast"]) p {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            color: #1e3a27 !important;
            background-color: transparent !important;
        }
        label:has(input[aria-label="Predictive Forecast"]) span {
            background-color: transparent !important;
        }
        label:has(input[aria-label="Predictive Forecast"]) {
            background-color: transparent !important;
        }
        [data-testid="stMarkdownContainer"] {
            background-color: transparent !important;
        }
        </style>
        """, unsafe_allow_html=True)
        predictive_mode = st.toggle("Predictive Forecast")

    target_datetime: datetime | None = None
    start_time: time | None = None
    end_time: time | None = None
    map_data = fdf          # default: historical data
    weight_col = "impact_score"

    if predictive_mode:
        latest_data_date = pd.to_datetime(fdf["created_datetime"]).max().date()
        selectable_dates = [latest_data_date + timedelta(days=i) for i in range(1, 8)]
        date_options = {d.strftime("%A, %b %d, %Y"): d for d in selectable_dates}

        with ctl2:
            date_col, start_col, end_col = st.columns(3)
            with date_col:
                selected_date_str = st.selectbox(
                    "Forecast Date",
                    options=list(date_options.keys()),
                    help=f"Select a forecast date relative to the latest available data ({latest_data_date})"
                )
                selected_date = date_options[selected_date_str]
                st.caption(f"📅 Valid range: {selectable_dates[0].strftime('%b %d')} to {selectable_dates[-1].strftime('%b %d, %Y')}")
            
            start_time = start_col.time_input("Start Time", value=time(18, 0))
            end_time = end_col.time_input("End Time", value=time(21, 0))
            
        target_datetime = datetime.combine(selected_date, start_time)
        weight_col = "predicted_avg_impact"
        
        start_dt = datetime.combine(selected_date, start_time)
        end_dt = datetime.combine(selected_date, end_time)
        time_gap_hours = (end_dt - start_dt).total_seconds() / 3600.0

        if end_dt <= start_dt:
            st.error("End time must be after start time.")
            map_data = pd.DataFrame()
            target_datetime = None
        elif not (1 <= time_gap_hours <= 4):
            st.error("Time window must be between 1 and 4 hours.")
            map_data = pd.DataFrame()
            target_datetime = None
        else:
            map_data = generate_predictions(fdf, target_datetime, start_time, end_time)
            if not map_data.empty:
                st.info(
                    f"Showing predicted hotspots for "
                    f"**{selected_date.strftime('%A')}** between "
                    f"**{start_time.strftime('%H:%M')}** and **{end_time.strftime('%H:%M')}**"
                )
    else:
        st.success("Showing historical violation data")

    st.markdown("---")
    # --- END PREDICTIVE MODE CONTROLS ---

    # ── Dynamic section title & caption ──────────────────────────────────────
    if predictive_mode and target_datetime is not None and start_time is not None and end_time is not None:
        map_title = "PREDICTED HOTSPOT HEATMAP"
        caption = (
            f"Forecast based on historical patterns for "
            f"<strong>{target_datetime.strftime('%A')}</strong> "
            f"between <strong>{start_time.strftime('%H:%M')}</strong> and <strong>{end_time.strftime('%H:%M')}</strong>"
        )
    else:
        map_title = "HISTORICAL HOTSPOT HEATMAP"
        caption = "Density weighted by calculated impact score"

    st.markdown(
        f'<div class="section-header">{map_title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<span style='font-size:0.85rem;color:#45655c;display:block;margin-bottom:0.75rem;'>"
        f"{caption}</span>",
        unsafe_allow_html=True,
    )


    # ── Empty-state guard: skip map + cards if no data ────────────────────────
    if map_data.empty:
        st.warning(
            "No historical data found for this specific day and time. "
            "Please try a different time or disable predictive mode."
        )
    else:
        render_map(map_data, weight_col=weight_col)

        # ── Section 3: Top Locations Cards ───────────────────────────────────
        top10_title = (
            "TOP 10 PREDICTED HOTSPOTS" if predictive_mode
            else "TOP 10 LOCATIONS BY IMPACT SCORE"
        )
        st.markdown(
            f'<div class="section-header">{top10_title}</div>',
            unsafe_allow_html=True,
        )
        render_hotspot_cards(map_data, predictive_mode=predictive_mode)

        # ── Section 4: Patrol Routes ─────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">PATROL ROUTE OPTIMIZER</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span style='font-size:0.85rem;color:#45655c;display:block;"
        "margin-bottom:0.75rem;'>"
        "Traffic-aware patrol routes generated by Mappls Routing API · "
        "Stops ranked by composite impact score"
        "</span>",
        unsafe_allow_html=True,
    )
    patrol_routes = load_patrol_routes(PATROL_ROUTES_PATH)
    render_patrol_routes(patrol_routes)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='text-align:center;color:#45655c;font-size:0.75rem;"
        "margin-top:3rem;padding-top:1rem;border-top:1px solid #b1d3b9;'>"
        "Gridlock Parking Intelligence · Built for Bengaluru Traffic Authority"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

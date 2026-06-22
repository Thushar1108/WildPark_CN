"""
src/patrol_router.py
--------------------
Patrol Route Optimizer for the Gridlock illegal-parking intelligence system.

Given the processed DataFrame, this module:
  1. Aggregates violations to grid-cell level and ranks by total impact_score.
  2. Selects the top-N hotspot cells (N = config.TOP_N_HOTSPOTS).
  3. Distributes them across patrol units via greedy round-robin
     (highest score → unit 0, second → unit 1, … wraps around).
  4. For each unit, calls the Mappls Route ETA API (traffic-aware) to get
     an ordered, driveable route through its assigned waypoints.
  5. Returns a list of per-unit route dicts and saves to
     data/patrol_routes.json.

Mappls REST Route API used:
  GET https://route.mappls.com/route/driving/{coordinates}
      ?resource=route_eta        (traffic-aware ETA)
      &geometries=polyline
      &overview=full
      &access_token={key}

  coordinates: semicolon-separated "longitude,latitude" pairs (lng first).
  Response fields used: routes[0].distance, routes[0].duration,
                        routes[0].geometry, waypoints[].name
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mappls API constants
# ---------------------------------------------------------------------------

_ROUTE_BASE = "https://route.mappls.com/route/driving/{coords}"
_RESOURCE    = "route_eta"       # traffic-aware; falls back to "route" if unavailable
_GEOMETRIES  = "polyline"
_OVERVIEW    = "full"
_TIMEOUT_S   = 15                # per-request timeout


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _aggregate_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse individual violation rows to grid-cell level.

    Returns a DataFrame with one row per grid_id, sorted by
    total_impact_score descending. Only the top config.TOP_N_HOTSPOTS
    rows are returned.
    """
    required = {"grid_id", "latitude", "longitude", "impact_score", "location"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"patrol_router: missing columns in DataFrame: {missing}")

    agg = (
        df.groupby("grid_id", sort=False)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            total_impact_score=("impact_score", "sum"),
            violation_count=("impact_score", "count"),
            location=("location", lambda s: next(
                (v for v in s if pd.notna(v) and str(v).strip() not in ("", "nan")),
                str(s.iloc[0]),
            )),
        )
        .reset_index()
        .sort_values("total_impact_score", ascending=False)
        .head(config.TOP_N_HOTSPOTS)
        .reset_index(drop=True)
    )

    logger.info(
        "Hotspot aggregation: %d grid cells selected (top %d of %d unique cells)",
        len(agg),
        config.TOP_N_HOTSPOTS,
        df["grid_id"].nunique(),
    )
    return agg


def _round_robin_assign(hotspots: pd.DataFrame, n_units: int) -> dict[int, pd.DataFrame]:
    """
    Assign hotspot rows to patrol units via greedy round-robin.

    Row 0 (highest score) → unit 0
    Row 1                  → unit 1
    …
    Row n_units            → unit 0  (wraps)

    Each unit therefore gets a geographically spread set of high-priority
    zones rather than all units clustering at the same top zone.

    Returns
    -------
    dict[int, pd.DataFrame]
        Maps unit_id (0-indexed) to its assigned hotspot rows,
        ordered by impact_score descending.
    """
    assignments: dict[int, list[int]] = {u: [] for u in range(n_units)}
    for rank, idx in enumerate(hotspots.index):
        assignments[rank % n_units].append(idx)

    return {
        unit: hotspots.loc[indices].reset_index(drop=True)
        for unit, indices in assignments.items()
        if indices
    }


def _build_coord_string(waypoints: pd.DataFrame) -> str:
    """
    Build the semicolon-separated coordinate string required by the
    Mappls Route API.  Format per point: "longitude,latitude".
    """
    return ";".join(
        f"{row.longitude:.6f},{row.latitude:.6f}"
        for row in waypoints.itertuples()
    )


def _call_mappls_route(
    coord_string: str,
    api_key: str,
    unit_id: int,
) -> dict[str, Any]:
    """
    Call the Mappls Driving Route API for a single patrol unit's waypoints.

    Falls back to resource=``route`` (no traffic) if ``route_eta`` returns
    a non-200 status, so the pipeline never hard-fails due to a traffic
    data outage.

    Parameters
    ----------
    coord_string:
        Semicolon-separated "lng,lat" pairs.
    api_key:
        Mappls REST API access token.
    unit_id:
        Used for logging only.

    Returns
    -------
    dict
        Parsed JSON response, or an error dict on failure.
    """
    for resource in (_RESOURCE, "route"):          # retry with basic route on failure
        url = _ROUTE_BASE.format(coords=coord_string)
        params = {
            "resource":    resource,
            "geometries":  _GEOMETRIES,
            "overview":    _OVERVIEW,
            "access_token": api_key,
        }
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT_S)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("routes"):
                    logger.debug(
                        "Unit %d: Mappls route OK (resource=%s)", unit_id, resource
                    )
                    return data
                logger.warning(
                    "Unit %d: Mappls returned 200 but no routes (resource=%s). "
                    "Body: %s",
                    unit_id, resource, resp.text[:200],
                )
            else:
                logger.warning(
                    "Unit %d: Mappls HTTP %d (resource=%s)",
                    unit_id, resp.status_code, resource,
                )
        except requests.exceptions.Timeout:
            logger.warning("Unit %d: Mappls request timed out (resource=%s)", unit_id, resource)
        except requests.exceptions.RequestException as exc:
            logger.warning("Unit %d: Mappls request error – %s", unit_id, exc)

        time.sleep(0.5)   # brief back-off before fallback attempt

    return {"error": "route_unavailable"}


def _parse_route_response(
    api_response: dict[str, Any],
    waypoints_df: pd.DataFrame,
    unit_id: int,
) -> dict[str, Any]:
    """
    Extract distance, duration, geometry, and waypoint order from the
    Mappls API response and merge back with local hotspot metadata.
    """
    if "error" in api_response:
        logger.warning("Unit %d: using fallback (no API route).", unit_id)
        return _fallback_route(waypoints_df, unit_id)

    route = api_response["routes"][0]

    # Mappls returns distance in metres, duration in seconds
    distance_km   = round(route.get("distance", 0) / 1000, 2)
    duration_mins = round(route.get("duration", 0) / 60, 1)
    geometry      = route.get("geometry", "")

    # Map snapped waypoint names from API back to our hotspot records
    api_waypoints  = api_response.get("waypoints", [])
    waypoint_list  = []
    for i, row in waypoints_df.iterrows():
        snapped_name = (
            api_waypoints[i]["name"]
            if i < len(api_waypoints) and api_waypoints[i].get("name")
            else row.location
        )
        waypoint_list.append({
            "stop_order":        i + 1,
            "grid_id":           row.grid_id,
            "location":          row.location,
            "snapped_road_name": snapped_name,
            "latitude":          round(row.latitude, 6),
            "longitude":         round(row.longitude, 6),
            "total_impact_score": round(row.total_impact_score, 4),
            "violation_count":   int(row.violation_count),
        })

    return {
        "unit_id":              unit_id,
        "num_stops":            len(waypoint_list),
        "total_distance_km":    distance_km,
        "estimated_duration_mins": duration_mins,
        "route_geometry":       geometry,   # encoded polyline string
        "waypoints":            waypoint_list,
        "api_resource_used":    api_response.get("_resource_used", "route"),
    }


def _fallback_route(waypoints_df: pd.DataFrame, unit_id: int) -> dict[str, Any]:
    """
    Graceful fallback when the Mappls API is unavailable.
    Returns waypoints in impact-score order without routing metadata.
    """
    waypoint_list = [
        {
            "stop_order":         i + 1,
            "grid_id":            row.grid_id,
            "location":           row.location,
            "snapped_road_name":  row.location,
            "latitude":           round(row.latitude, 6),
            "longitude":          round(row.longitude, 6),
            "total_impact_score": round(row.total_impact_score, 4),
            "violation_count":    int(row.violation_count),
        }
        for i, row in waypoints_df.iterrows()
    ]
    return {
        "unit_id":                 unit_id,
        "num_stops":               len(waypoint_list),
        "total_distance_km":       None,
        "estimated_duration_mins": None,
        "route_geometry":          None,
        "waypoints":               waypoint_list,
        "api_resource_used":       "fallback",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_patrol_routing(
    df: pd.DataFrame,
    output_path: Path = Path("data/patrol_routes.json"),
) -> list[dict[str, Any]]:
    """
    End-to-end patrol route optimiser.

    Steps
    -----
    1. Read API key from env var ``MAPPLS_API_KEY``; fall back to
       ``config.MAPPLS_API_KEY``.  Raises ``EnvironmentError`` if neither
       is set — caller in main.py should catch this and skip Stage 4.
    2. Aggregate violations to grid-cell level; take top-N hotspots.
    3. Assign hotspots to patrol units via greedy round-robin.
    4. Call Mappls Route ETA API per unit.
    5. Save results as JSON and return the list of route dicts.

    Parameters
    ----------
    df:
        Fully processed DataFrame from the pipeline.
    output_path:
        Where to write ``patrol_routes.json``.

    Returns
    -------
    list[dict]
        One dict per patrol unit (see ``_parse_route_response`` for schema).
    """
    # ── 0. API key resolution ────────────────────────────────────────────────
    api_key = os.environ.get("MAPPLS_API_KEY") or getattr(config, "MAPPLS_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "MAPPLS_API_KEY not set. Export it as an environment variable or "
            "add it to config.py as MAPPLS_API_KEY = 'your_key'."
        )

    n_units = getattr(config, "NUM_PATROL_UNITS", 5)
    logger.info("Patrol routing: %d units, top-%d hotspots", n_units, config.TOP_N_HOTSPOTS)

    # ── 1. Aggregate & select hotspots ───────────────────────────────────────
    hotspots = _aggregate_hotspots(df)

    # ── 2. Round-robin assignment ────────────────────────────────────────────
    unit_assignments = _round_robin_assign(hotspots, n_units)

    # ── 3. Route each unit ───────────────────────────────────────────────────
    routes: list[dict[str, Any]] = []
    for unit_id, waypoints_df in unit_assignments.items():
        if len(waypoints_df) < 2:
            # Single-stop unit — no routing needed
            routes.append(_fallback_route(waypoints_df, unit_id))
            continue

        coord_str    = _build_coord_string(waypoints_df)
        api_response = _call_mappls_route(coord_str, api_key, unit_id)
        route_dict   = _parse_route_response(api_response, waypoints_df, unit_id)
        routes.append(route_dict)

        logger.info(
            "Unit %d: %d stops | %.1f km | %.0f min",
            unit_id,
            route_dict["num_stops"],
            route_dict["total_distance_km"] or 0,
            route_dict["estimated_duration_mins"] or 0,
        )
        time.sleep(0.3)   # gentle rate-limiting between units

    # ── 4. Persist ───────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(routes, fh, indent=2, ensure_ascii=False)

    logger.info("Patrol routes saved → %s", output_path)
    return routes

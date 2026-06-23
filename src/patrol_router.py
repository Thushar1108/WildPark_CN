"""
src/patrol_router.py
--------------------
Patrol Route Optimizer for the Gridlock illegal-parking intelligence system.

Given the processed DataFrame, this module:
  1. Aggregates violations to grid-cell level and ranks by total impact_score.
  2. Selects the top-N hotspot cells (N = config.TOP_N_HOTSPOTS).
  3. Distributes them across patrol units via greedy round-robin.
  4. Generates a temporary OAuth2 token via Client ID and Client Secret.
  5. For each unit, calls the Mappls Route ETA API (traffic-aware) with the Bearer token 
     to get an ordered, driveable route through its assigned waypoints.
  6. Returns a list of per-unit route dicts and saves to data/patrol_routes.json.
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

_TOKEN_URL   = "https://outpost.mappls.com/api/security/oauth/token"
_ROUTE_BASE  = "https://apis.mappls.com/advancedmaps/v1/{key}/route_eta/driving/{coords}"
_RESOURCE    = "route_eta"       # traffic-aware; falls back to "route" if unavailable
_GEOMETRIES  = "polyline"
_OVERVIEW    = "full"
_TIMEOUT_S   = 15                # per-request timeout


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_mappls_access_token() -> str | None:
    """
    Fetches the short-lived OAuth2 access token using Client ID and Client Secret.
    """
    # Try Streamlit secrets first, then environment variables, then config fallbacks
    try:
        import streamlit as st
        client_id = st.secrets.get("MAPPLS_CLIENT_ID")
        client_secret = st.secrets.get("MAPPLS_CLIENT_SECRET")
    except Exception:
        client_id = None
        client_secret = None

    client_id = client_id or os.environ.get("MAPPLS_CLIENT_ID") or getattr(config, "MAPPLS_CLIENT_ID", "")
    client_secret = client_secret or os.environ.get("MAPPLS_CLIENT_SECRET") or getattr(config, "MAPPLS_CLIENT_SECRET", "")
    
    if not client_id or not client_secret:
        logger.error("OAuth2 Failed: MAPPLS_CLIENT_ID or MAPPLS_CLIENT_SECRET is missing.")
        return None

    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    try:
        resp = requests.post(_TOKEN_URL, data=payload, headers=headers, timeout=_TIMEOUT_S)
        print("=== OAUTH DEBUG ===")
        print(f"Status Code: {resp.status_code}")
        print(f"Raw Token Response: {resp.text}")
        if resp.status_code == 200:
            data = resp.json()
            token_type = data.get("token_type", "Bearer")
            access_token = data.get("access_token")
            if access_token:
                logger.info("Successfully authenticated with Mappls OAuth2 server.")
                return f"{token_type} {access_token}"
        logger.warning(f"Failed to fetch Mappls OAuth2 token. HTTP {resp.status_code}: {resp.text}")
    except requests.exceptions.RequestException as exc:
        logger.error(f"Network error while fetching Mappls authorization token: {exc}")
    
    return None


def _aggregate_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse individual violation rows to grid-cell level.
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
    Mappls Route API. Format per point: "longitude,latitude".
    """
    return ";".join(
        f"{row.longitude:.6f},{row.latitude:.6f}"
        for row in waypoints.itertuples()
    )


def _call_mappls_route(
    coord_string: str,
    access_token: str,
    unit_id: int,
) -> dict[str, Any]:
    """
    Call the Mappls Driving Route API for a single patrol unit's waypoints.
    """
    try:
        import streamlit as st
        api_key = st.secrets.get("MAPPLS_API_KEY")
    except Exception:
        api_key = None

    api_key = api_key or os.environ.get("MAPPLS_API_KEY") or getattr(config, "MAPPLS_API_KEY", "")

    for mode in ("route_eta", "route"):
        # Explicit f-string construction avoids the formatting KeyError
        url = f"https://apis.mappls.com/advancedmaps/v1/{api_key}/{mode}/driving/{coord_string}"
        params = {
            "geometries": _GEOMETRIES,
            "overview": _OVERVIEW,
        }
        headers = {
            "Authorization": access_token,
            "Content-Type": "application/json"
        }
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=_TIMEOUT_S)
            
            print("=== ROUTING API DEBUG ===")
            print(f"URL: {resp.url}")
            print(f"Status: {resp.status_code}")
            print(f"Response: {resp.text[:200]}")
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get("routes"):
                    logger.debug("Unit %d: Mappls route OK (mode=%s)", unit_id, mode)
                    data["_resource_used"] = mode
                    return data
            else:
                logger.warning("Unit %d: Mappls HTTP %d | Body: %s", unit_id, resp.status_code, resp.text[:200])
                
        except requests.exceptions.Timeout:
            logger.warning("Unit %d: Mappls request timed out", unit_id)
        except requests.exceptions.RequestException as exc:
            logger.warning("Unit %d: Request error – %s", unit_id, exc)

        time.sleep(0.5)

    return {"error": "route_unavailable"}

def _parse_route_response(
    api_response: dict[str, Any],
    waypoints_df: pd.DataFrame,
    unit_id: int,
) -> dict[str, Any]:
    """
    Extract metadata from response fields and bundle it into your dashboard schema.
    """
    if "error" in api_response:
        logger.warning("Unit %d: using fallback (no API route).", unit_id)
        return _fallback_route(waypoints_df, unit_id)

    route = api_response["routes"][0]

    distance_km   = round(route.get("distance", 0) / 1000, 2)
    duration_mins = round(route.get("duration", 0) / 60, 1)
    geometry      = route.get("geometry", "")

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
        "unit_id":               unit_id,
        "num_stops":             len(waypoint_list),
        "total_distance_km":     distance_km,
        "estimated_duration_mins": duration_mins,
        "route_geometry":        geometry,
        "waypoints":             waypoint_list,
        "api_resource_used":     api_response.get("_resource_used", "route"),
    }


def _fallback_route(waypoints_df: pd.DataFrame, unit_id: int) -> dict[str, Any]:
    """
    Graceful fallback layout when the Mappls API is down or tokens expire.
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
    End-to-end patrol route optimiser leveraging modern OAuth2 tokens.
    """
    # ── 0. OAuth2 Token Generation ──────────────────────────────────────────
    access_token = _get_mappls_access_token()
    if not access_token:
        raise EnvironmentError(
            "Could not authorize Mappls connection. Please verify MAPPLS_CLIENT_ID "
            "and MAPPLS_CLIENT_SECRET are correctly initialized."
        )

    n_units = getattr(config, "NUM_PATROL_UNITS", 5)
    logger.info("Patrol routing execution: %d units tracking top-%d hotspots", n_units, config.TOP_N_HOTSPOTS)

    # ── 1. Aggregate & select hotspots ───────────────────────────────────────
    hotspots = _aggregate_hotspots(df)

    # ── 2. Round-robin assignment ────────────────────────────────────────────
    unit_assignments = _round_robin_assign(hotspots, n_units)

    # ── 3. Route each unit ───────────────────────────────────────────────────
    routes: list[dict[str, Any]] = []
    for unit_id, waypoints_df in unit_assignments.items():
        if len(waypoints_df) < 2:
            routes.append(_fallback_route(waypoints_df, unit_id))
            continue

        coord_str    = _build_coord_string(waypoints_df)
        api_response = _call_mappls_route(coord_str, access_token, unit_id)
        route_dict   = _parse_route_response(api_response, waypoints_df, unit_id)
        routes.append(route_dict)

        logger.info(
            "Unit %d: %d stops | %.1f km | %.0f min",
            unit_id,
            route_dict["num_stops"],
            route_dict["total_distance_km"] or 0,
            route_dict["estimated_duration_mins"] or 0,
        )
        time.sleep(0.3)

    # ── 4. Persist to disk ───────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(routes, fh, indent=2, ensure_ascii=False)

    logger.info("Patrol routes securely generated and saved → %s", output_path)
    return routes
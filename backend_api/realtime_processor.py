"""
Real-Time Processor Module

Aggregates ingested events into 1-minute buckets and applies MAD anomaly detection.
Maintains an in-memory scored DataFrame that mirrors the batch scored_1min.csv schema.

Usage:
------
    from backend_api.realtime_processor import rebuild_scores, get_scored_df, set_scored_df

    # Rebuild scores from events
    events = telemetry_store.get_all_events()
    scored_df = rebuild_scores(events, lookback_minutes=120)
    set_scored_df(scored_df)

    # Get current scored data
    df = get_scored_df()
"""

import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from ingest.detector_mad import add_mad_anomalies


# =============================================================================
# CONFIGURATION
# =============================================================================

# Expected columns in scored DataFrame (matches batch scored_1min.csv)
SCORED_COLUMNS = [
    "bucket_ts",
    "service",
    "endpoint",
    "req_count",
    "error_count",
    "error_rate",
    "avg_latency_ms",
    "p95_latency_ms",
    "mad_z",
    "is_anomaly",
    "severity"
]


# =============================================================================
# THREAD-SAFE IN-MEMORY STORE
# =============================================================================

_scored_df: pd.DataFrame = pd.DataFrame(columns=SCORED_COLUMNS)
_lock = threading.Lock()


# =============================================================================
# PUBLIC API
# =============================================================================

def get_scored_df() -> pd.DataFrame:
    """
    Get the current scored DataFrame.

    Returns
    -------
    pd.DataFrame
        Scored DataFrame with anomaly detection results.
        Returns empty DataFrame with correct columns if no data.

    Thread Safety
    -------------
    This function is thread-safe.
    """
    with _lock:
        return _scored_df.copy()


def set_scored_df(df: pd.DataFrame) -> None:
    """
    Set the scored DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Scored DataFrame to store.

    Thread Safety
    -------------
    This function is thread-safe.
    """
    global _scored_df
    with _lock:
        _scored_df = df.copy()


def rebuild_scores(
    events: List[Dict[str, Any]],
    lookback_minutes: int = 120
) -> pd.DataFrame:
    """
    Rebuild scored DataFrame from raw events.

    This function:
    1. Parses events into a DataFrame
    2. Filters to the lookback window
    3. Aggregates into 1-minute buckets per (service, endpoint)
    4. Applies MAD anomaly detection
    5. Returns DataFrame matching batch scored_1min.csv schema

    Parameters
    ----------
    events : List[Dict[str, Any]]
        List of raw log events with keys:
        timestamp, service, endpoint, status_code, latency_ms, trace_id
    lookback_minutes : int, optional
        Number of minutes to look back from max timestamp (default: 120).

    Returns
    -------
    pd.DataFrame
        Scored DataFrame with columns:
        bucket_ts, service, endpoint, req_count, error_count, error_rate,
        avg_latency_ms, p95_latency_ms, mad_z, is_anomaly, severity

    Notes
    -----
    If events is empty, returns empty DataFrame with correct columns.
    """
    # Handle empty events
    if not events:
        return _create_empty_scored_df()

    # Convert events to DataFrame
    df = pd.DataFrame(events)

    # Parse timestamps (handle various ISO formats)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

    # Drop rows with invalid timestamps
    df = df.dropna(subset=['timestamp'])

    if len(df) == 0:
        return _create_empty_scored_df()

    # Convert to timezone-naive for consistent processing
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    # Filter to lookback window
    max_ts = df['timestamp'].max()
    min_ts = max_ts - timedelta(minutes=lookback_minutes)
    df = df[df['timestamp'] >= min_ts]

    if len(df) == 0:
        return _create_empty_scored_df()

    # Build per-minute features
    features_df = _build_per_minute_features(df)

    if len(features_df) == 0:
        return _create_empty_scored_df()

    # Apply MAD anomaly detection
    scored_df = _apply_mad_detection(features_df)

    return scored_df


# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================

def _create_empty_scored_df() -> pd.DataFrame:
    """Create an empty DataFrame with the correct scored columns."""
    return pd.DataFrame(columns=SCORED_COLUMNS)


def _build_per_minute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-minute aggregated features from raw events.

    Parameters
    ----------
    df : pd.DataFrame
        Raw events DataFrame with columns:
        timestamp, service, endpoint, status_code, latency_ms

    Returns
    -------
    pd.DataFrame
        Aggregated features with columns:
        bucket_ts, service, endpoint, req_count, error_count, error_rate,
        avg_latency_ms, p95_latency_ms
    """
    # Create 1-minute buckets
    df = df.copy()
    df['bucket_ts'] = df['timestamp'].dt.floor('1min')

    # Create error flag (status_code >= 500 is an error)
    df['is_error'] = (df['status_code'] >= 500).astype(int)

    # Group by bucket_ts, service, endpoint
    grouped = df.groupby(['bucket_ts', 'service', 'endpoint'])

    # Compute aggregated metrics
    features = grouped.agg(
        req_count=('timestamp', 'count'),
        error_count=('is_error', 'sum'),
        avg_latency_ms=('latency_ms', 'mean'),
        p95_latency_ms=('latency_ms', lambda x: np.percentile(x, 95) if len(x) > 0 else 0)
    ).reset_index()

    # Calculate error rate
    features['error_rate'] = features['error_count'] / features['req_count']

    # Sort by bucket_ts ascending for consistent ordering
    features = features.sort_values('bucket_ts').reset_index(drop=True)

    return features


def _apply_mad_detection(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply MAD-based anomaly detection to features.

    Parameters
    ----------
    features_df : pd.DataFrame
        Features DataFrame with per-minute metrics.
        bucket_ts should be datetime.

    Returns
    -------
    pd.DataFrame
        DataFrame with added anomaly detection columns:
        mad_z, is_anomaly, severity
    """
    # Ensure bucket_ts is datetime
    if not pd.api.types.is_datetime64_any_dtype(features_df['bucket_ts']):
        features_df = features_df.copy()
        features_df['bucket_ts'] = pd.to_datetime(features_df['bucket_ts'])

    # Apply MAD anomaly detection using existing function
    # Use relaxed parameters for real-time (less historical data)
    scored_df = add_mad_anomalies(
        df=features_df,
        group_cols=("service", "endpoint"),
        ts_col="bucket_ts",
        metric_col="p95_latency_ms",
        threshold=3.5,
        min_points=10,      # Relaxed from 20 for real-time
        warmup_points=5,    # Relaxed from 15 for real-time
        min_req_count=10    # Relaxed from 20 for real-time
    )

    # Sort by bucket_ts ascending for consistent ordering
    scored_df = scored_df.sort_values('bucket_ts').reset_index(drop=True)

    # Convert bucket_ts to ISO string to match batch format
    scored_df['bucket_ts'] = scored_df['bucket_ts'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    # Ensure column order matches expected schema
    scored_df = scored_df[SCORED_COLUMNS]

    return scored_df


def get_realtime_stats() -> Dict[str, Any]:
    """
    Get statistics about the current real-time scored data.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - row_count: Number of rows in scored DataFrame
        - anomaly_count: Number of anomalies detected
        - services: List of unique services
        - oldest_ts: Oldest bucket timestamp
        - newest_ts: Newest bucket timestamp
    """
    with _lock:
        df = _scored_df

        if len(df) == 0:
            return {
                "row_count": 0,
                "anomaly_count": 0,
                "services": [],
                "oldest_ts": None,
                "newest_ts": None
            }

        return {
            "row_count": len(df),
            "anomaly_count": int(df['is_anomaly'].sum()) if 'is_anomaly' in df.columns else 0,
            "services": sorted(df['service'].unique().tolist()) if 'service' in df.columns else [],
            "oldest_ts": df['bucket_ts'].min() if 'bucket_ts' in df.columns else None,
            "newest_ts": df['bucket_ts'].max() if 'bucket_ts' in df.columns else None
        }

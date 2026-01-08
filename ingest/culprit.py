"""
Culprit Identification Module

Purpose:
--------
This module provides functions to identify the most likely problematic 
microservice/endpoint combinations from anomaly detection results.

Key Responsibilities:
--------------------
- Analyze recent anomalies within a configurable time window
- Score and rank service/endpoint combinations by severity
- Provide actionable culprit identification for incident response

Dependencies:
-------------
    pandas, numpy (standard data science stack)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


# Severity ordering for comparison (higher = more severe)
SEVERITY_ORDER = {"Info": 0, "Warning": 1, "Critical": 2}


def find_top_culprits(
    df_scored: pd.DataFrame,
    minutes: int = 15,
    now_ts: Optional[datetime] = None,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Identify the top culprit service/endpoint combinations from recent anomalies.
    
    This function analyzes the scored anomaly data within a recent time window
    and ranks service/endpoint combinations by their anomaly severity to identify
    the most likely root causes of issues.
    
    Parameters
    ----------
    df_scored : pd.DataFrame
        Scored per-minute dataframe with anomaly detection results.
        Expected columns: bucket_ts, service, endpoint, is_anomaly, mad_z,
        p95_latency_ms, severity, and optionally error_rate.
    minutes : int, optional
        Number of minutes to look back from now_ts (default: 15).
    now_ts : datetime, optional
        Reference timestamp for the lookback window. If None, uses the
        maximum bucket_ts in the dataframe (default: None).
    top_k : int, optional
        Number of top culprits to return (default: 3).
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - window_minutes: int - The lookback window used
        - window_start: str - ISO timestamp of window start
        - window_end: str - ISO timestamp of window end
        - total_anomalies: int - Total anomaly rows in window
        - culprit: dict or None - Top culprit details, or None if no anomalies
        - top: list - Top k culprits ranked by severity
    
    Ranking Logic
    -------------
    Culprits are ranked by (in order of priority):
    1. peak_abs_mad (descending) - Highest absolute MAD z-score
    2. peak_severity_rank (descending) - Critical > Warning > Info
    3. peak_error_rate (descending) - Highest error rate (if available)
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv("data/scored_1min.csv")
    >>> result = find_top_culprits(df, minutes=15, top_k=3)
    >>> if result["culprit"]:
    ...     print(f"Top culprit: {result['culprit']['service']}:{result['culprit']['endpoint']}")
    """
    # Make a copy to avoid modifying the original
    df = df_scored.copy()
    
    # Ensure bucket_ts is datetime
    if df['bucket_ts'].dtype == 'object':
        df['bucket_ts'] = pd.to_datetime(df['bucket_ts'])
    
    # Determine reference timestamp (now_ts)
    if now_ts is None:
        now_ts = df['bucket_ts'].max()
    elif isinstance(now_ts, str):
        now_ts = pd.to_datetime(now_ts)
    
    # Calculate window boundaries
    window_start = now_ts - timedelta(minutes=minutes)
    window_end = now_ts
    
    # Filter to the time window
    mask_time = (df['bucket_ts'] >= window_start) & (df['bucket_ts'] <= window_end)
    df_window = df[mask_time]
    
    # Filter to anomalies only
    df_anomalies = df_window[df_window['is_anomaly'] == True]
    
    total_anomalies = len(df_anomalies)
    
    # If no anomalies, return empty result
    if total_anomalies == 0:
        return {
            "window_minutes": minutes,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "total_anomalies": 0,
            "culprit": None,
            "top": []
        }
    
    # Check if error_rate column exists
    has_error_rate = 'error_rate' in df_anomalies.columns
    
    # Aggregate metrics per (service, endpoint)
    culprit_stats: List[Dict[str, Any]] = []
    
    for (service, endpoint), group in df_anomalies.groupby(['service', 'endpoint']):
        # Calculate peak metrics
        peak_abs_mad = float(np.abs(group['mad_z']).max())
        peak_p95 = float(group['p95_latency_ms'].max())
        peak_error = float(group['error_rate'].max()) if has_error_rate else 0.0
        
        # Determine peak severity
        severity_ranks = group['severity'].map(SEVERITY_ORDER)
        peak_severity_rank = int(severity_ranks.max())
        peak_severity = [k for k, v in SEVERITY_ORDER.items() if v == peak_severity_rank][0]
        
        # Count anomaly points
        points = len(group)
        
        # Calculate additional context
        avg_mad = float(np.abs(group['mad_z']).mean())
        first_ts = group['bucket_ts'].min().isoformat()
        last_ts = group['bucket_ts'].max().isoformat()
        
        culprit_stats.append({
            "service": service,
            "endpoint": endpoint,
            "points": points,
            "peak_abs_mad": round(peak_abs_mad, 3),
            "avg_abs_mad": round(avg_mad, 3),
            "peak_p95_latency_ms": round(peak_p95, 2),
            "peak_error_rate": round(peak_error, 4),
            "peak_severity": peak_severity,
            "peak_severity_rank": peak_severity_rank,
            "first_anomaly_ts": first_ts,
            "last_anomaly_ts": last_ts
        })
    
    # Sort by ranking criteria: peak_abs_mad desc, peak_severity_rank desc, peak_error_rate desc
    culprit_stats.sort(
        key=lambda x: (x['peak_abs_mad'], x['peak_severity_rank'], x['peak_error_rate']),
        reverse=True
    )
    
    # Select top_k
    top_culprits = culprit_stats[:top_k]
    
    # Remove internal ranking field from output
    for item in top_culprits:
        del item['peak_severity_rank']
    
    # Build result
    result = {
        "window_minutes": minutes,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "total_anomalies": total_anomalies,
        "culprit": top_culprits[0] if top_culprits else None,
        "top": top_culprits
    }
    
    return result


def format_culprit_report(culprit_result: Dict[str, Any]) -> str:
    """
    Format culprit analysis result as a human-readable report.
    
    Parameters
    ----------
    culprit_result : Dict[str, Any]
        Result from find_top_culprits()
    
    Returns
    -------
    str
        Formatted report string
    """
    lines = [
        "=" * 60,
        "CULPRIT ANALYSIS REPORT",
        "=" * 60,
        f"Window: {culprit_result['window_minutes']} minutes",
        f"  From: {culprit_result['window_start']}",
        f"  To:   {culprit_result['window_end']}",
        f"Total Anomalies: {culprit_result['total_anomalies']}",
        ""
    ]
    
    if culprit_result['culprit'] is None:
        lines.append("No anomalies detected in window.")
    else:
        lines.append("TOP CULPRIT:")
        c = culprit_result['culprit']
        lines.append(f"  Service:  {c['service']}")
        lines.append(f"  Endpoint: {c['endpoint']}")
        lines.append(f"  Severity: {c['peak_severity']}")
        lines.append(f"  Peak MAD z-score: {c['peak_abs_mad']:.2f}")
        lines.append(f"  Peak p95 Latency: {c['peak_p95_latency_ms']:.1f}ms")
        lines.append(f"  Peak Error Rate:  {c['peak_error_rate']:.2%}")
        lines.append(f"  Anomaly Points:   {c['points']}")
        lines.append("")
        
        if len(culprit_result['top']) > 1:
            lines.append("OTHER SUSPECTS:")
            for i, suspect in enumerate(culprit_result['top'][1:], start=2):
                lines.append(f"  #{i}: {suspect['service']}:{suspect['endpoint']} "
                           f"(MAD={suspect['peak_abs_mad']:.2f}, {suspect['peak_severity']})")
    
    lines.append("=" * 60)
    return "\n".join(lines)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    import os
    
    SCORED_FILE = "data/scored_1min.csv"
    
    if not os.path.exists(SCORED_FILE):
        print(f"Error: {SCORED_FILE} not found.")
        print("Please run the pipeline first:")
        print("  python -m ingest.generate_logs")
        print("  python -m ingest.feature_build")
    else:
        # Load scored data
        df = pd.read_csv(SCORED_FILE)
        
        # Find culprits in the last 15 minutes of data
        result = find_top_culprits(df, minutes=15, top_k=3)
        
        # Print formatted report
        print(format_culprit_report(result))
        
        # Also print raw result for debugging
        print("\nRaw Result:")
        import json
        print(json.dumps(result, indent=2, default=str))

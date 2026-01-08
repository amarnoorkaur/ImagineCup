"""
Microservice Anomaly Dashboard

A Streamlit dashboard for visualizing microservice metrics and detected anomalies
using MAD (Median Absolute Deviation) based detection without requiring training.

Supports both:
- Live mode: Polls FastAPI backend for real-time data
- Batch mode: Reads from CSV file (original behavior)

Run Instructions:
-----------------
From the repository root, run:
    streamlit run dashboard/app.py

Dependencies:
-------------
    pip install streamlit pandas requests
"""

import os
import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORED_DATA_PATH = "data/scored_1min.csv"
DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_REFRESH_INTERVAL = 10  # seconds

# Severity ordering for comparison
SEVERITY_ORDER = {"Info": 0, "Warning": 1, "Critical": 2}


# =============================================================================
# API HELPER FUNCTIONS
# =============================================================================

def fetch_stats(api_url: str) -> Optional[Dict[str, Any]]:
    """Fetch overall stats from API."""
    try:
        response = requests.get(f"{api_url}/stats", timeout=5)
        if response.ok:
            return response.json()
        return None
    except Exception:
        return None


def fetch_metrics(api_url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Fetch metrics from API.
    
    Parameters
    ----------
    api_url : str
        Base API URL
    params : Dict[str, Any]
        Query parameters (service, endpoint, limit)
    
    Returns
    -------
    Optional[Dict[str, Any]]
        API response or None on error
    """
    try:
        # Filter out None values
        query_params = {k: v for k, v in params.items() if v is not None}
        response = requests.get(f"{api_url}/metrics", params=query_params, timeout=10)
        if response.ok:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Failed to fetch metrics: {e}")
        return None


def fetch_anomalies(api_url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Fetch anomalies from API.
    
    Parameters
    ----------
    api_url : str
        Base API URL
    params : Dict[str, Any]
        Query parameters (service, endpoint, severity, limit)
    
    Returns
    -------
    Optional[Dict[str, Any]]
        API response or None on error
    """
    try:
        query_params = {k: v for k, v in params.items() if v is not None}
        response = requests.get(f"{api_url}/anomalies", params=query_params, timeout=10)
        if response.ok:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Failed to fetch anomalies: {e}")
        return None


def fetch_incidents(api_url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Fetch incidents from API.
    
    Parameters
    ----------
    api_url : str
        Base API URL
    params : Dict[str, Any]
        Query parameters (limit)
    
    Returns
    -------
    Optional[Dict[str, Any]]
        API response or None on error
    """
    try:
        query_params = {k: v for k, v in params.items() if v is not None}
        response = requests.get(f"{api_url}/incidents", params=query_params, timeout=10)
        if response.ok:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Failed to fetch incidents: {e}")
        return None


def fetch_culprit(api_url: str, minutes: int = 15, top_k: int = 3) -> Optional[Dict[str, Any]]:
    """Fetch top culprits from API."""
    try:
        response = requests.get(
            f"{api_url}/culprit",
            params={"minutes": minutes, "top_k": top_k},
            timeout=10
        )
        if response.ok:
            return response.json()
        return None
    except Exception:
        return None


def fetch_explain_incident(
    api_url: str,
    service: str,
    endpoint: str,
    start_ts: str,
    end_ts: str
) -> Optional[Dict[str, Any]]:
    """Fetch incident explanation from API."""
    try:
        response = requests.get(
            f"{api_url}/explain_incident",
            params={
                "service": service,
                "endpoint": endpoint,
                "start_ts": start_ts,
                "end_ts": end_ts
            },
            timeout=10
        )
        if response.ok:
            return response.json()
        return None
    except Exception:
        return None


def check_api_health(api_url: str) -> bool:
    """Check if API is reachable."""
    try:
        response = requests.get(f"{api_url}/health", timeout=3)
        return response.ok
    except Exception:
        return False


# =============================================================================
# DATA LOADING (CSV - Batch Mode)
# =============================================================================

def load_data_from_csv() -> Optional[pd.DataFrame]:
    """
    Load scored data from CSV file.
    
    Returns
    -------
    Optional[pd.DataFrame]
        Scored data, or None if file not found
    """
    if not os.path.exists(SCORED_DATA_PATH):
        return None
    
    df = pd.read_csv(SCORED_DATA_PATH)
    df['bucket_ts'] = pd.to_datetime(df['bucket_ts'])
    
    return df


def load_data_from_api(api_url: str, limit: int = 10000) -> Optional[pd.DataFrame]:
    """
    Load scored data from API.
    
    Parameters
    ----------
    api_url : str
        Base API URL
    limit : int
        Maximum rows to fetch
    
    Returns
    -------
    Optional[pd.DataFrame]
        Scored data, or None on error
    """
    result = fetch_metrics(api_url, {"limit": limit})
    if result is None or len(result.get("data", [])) == 0:
        return None
    
    df = pd.DataFrame(result["data"])
    df['bucket_ts'] = pd.to_datetime(df['bucket_ts'])
    
    return df


def generate_local_explanation(incident: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a templated explanation for an incident (fallback when API unavailable).
    
    Parameters
    ----------
    incident : Dict[str, Any]
        Incident data with keys: service, endpoint, start_ts, end_ts,
        max_p95_latency_ms, max_error_rate, peak_severity, count_points
    
    Returns
    -------
    Dict[str, Any]
        Explanation dictionary
    """
    service = incident.get('service', 'unknown')
    endpoint = incident.get('endpoint', 'unknown')
    peak_severity = incident.get('peak_severity', 'Info')
    max_latency = float(incident.get('max_p95_latency_ms', 0))
    max_error_rate = float(incident.get('max_error_rate', 0))
    count_points = incident.get('count_points', 0)
    
    # Determine confidence
    if count_points >= 5 and max_latency > 500:
        confidence = "High"
    elif count_points >= 3:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # Generate summary
    if peak_severity == "Critical":
        summary = f"Critical incident on {service}:{endpoint}. Peak latency {max_latency:.0f}ms with elevated errors."
    elif peak_severity == "Warning":
        summary = f"Warning-level anomaly on {service}:{endpoint}. Elevated latency of {max_latency:.0f}ms detected."
    else:
        summary = f"Minor deviation on {service}:{endpoint}. Metrics slightly elevated."
    
    # Why flagged
    why_flagged = [f"P95 latency spiked to {max_latency:.0f}ms"]
    if max_error_rate >= 0.05:
        why_flagged.append(f"Error rate elevated to {max_error_rate:.1%}")
    why_flagged.append(f"Anomaly persisted for {count_points} data points")
    
    # Impact
    if peak_severity == "Critical":
        impact = ["Users experiencing failed requests", "SLO may be breached", "Revenue impact possible"]
    elif peak_severity == "Warning":
        impact = ["Degraded performance for users", "Service functional but slow"]
    else:
        impact = ["Minimal user impact"]
    
    # Likely causes
    if max_error_rate >= 0.05 and max_latency > 500:
        likely_causes = ["Downstream service failure", "Database connection issues", "External API timeout"]
    elif max_latency > 1000:
        likely_causes = ["Resource contention", "Memory pressure", "Network latency spike"]
    else:
        likely_causes = ["Increased traffic", "Cache miss", "GC pause"]
    
    # Actions
    if peak_severity == "Critical":
        actions = ["Check service logs immediately", "Verify downstream dependencies", "Consider rollback if recent deploy"]
    else:
        actions = ["Monitor for recurrence", "Review recent changes", "Check resource utilization"]
    
    return {
        "service": service,
        "endpoint": endpoint,
        "start_ts": incident.get('start_ts', ''),
        "end_ts": incident.get('end_ts', ''),
        "summary": summary,
        "why_flagged": why_flagged,
        "impact": impact,
        "likely_causes": likely_causes,
        "recommended_actions": actions,
        "confidence": confidence,
        "metrics": {
            "peak_p95_latency_ms": max_latency,
            "peak_error_rate": max_error_rate,
            "count_points": count_points,
            "peak_severity": peak_severity
        }
    }


def compute_incidents(df: pd.DataFrame) -> List[Dict]:
    """
    Compute incident windows from anomalies.
    
    Groups consecutive anomalous data points by service and endpoint
    into incident windows.
    
    Parameters
    ----------
    df : pd.DataFrame
        Scored data with anomaly flags
    
    Returns
    -------
    List[Dict]
        List of incident dictionaries
    """
    # Filter only anomalies
    anomalies = df[df['is_anomaly'] == True].copy()
    
    if len(anomalies) == 0:
        return []
    
    # Sort by service, endpoint, and timestamp
    anomalies = anomalies.sort_values(['service', 'endpoint', 'bucket_ts'])
    
    incidents = []
    
    # Group by service and endpoint
    for (service, endpoint), group in anomalies.groupby(['service', 'endpoint']):
        group = group.sort_values('bucket_ts').reset_index(drop=True)
        
        # Detect consecutive minute windows (incidents)
        current_incident = {
            'service': service,
            'endpoint': endpoint,
            'start_ts': group.iloc[0]['bucket_ts'],
            'end_ts': group.iloc[0]['bucket_ts'],
            'max_p95_latency_ms': group.iloc[0]['p95_latency_ms'],
            'max_error_rate': group.iloc[0]['error_rate'],
            'peak_severity': group.iloc[0]['severity'],
            'count_points': 1
        }
        
        for i in range(1, len(group)):
            row = group.iloc[i]
            prev_ts = group.iloc[i-1]['bucket_ts']
            curr_ts = row['bucket_ts']
            
            # Check if this is consecutive (within 2 minutes to allow small gaps)
            time_diff = (curr_ts - prev_ts).total_seconds() / 60
            
            if time_diff <= 2:
                # Continue current incident
                current_incident['end_ts'] = curr_ts
                current_incident['max_p95_latency_ms'] = max(
                    current_incident['max_p95_latency_ms'],
                    row['p95_latency_ms']
                )
                current_incident['max_error_rate'] = max(
                    current_incident['max_error_rate'],
                    row['error_rate']
                )
                # Update severity (Critical > Warning > Info)
                if row['severity'] == 'Critical':
                    current_incident['peak_severity'] = 'Critical'
                elif row['severity'] == 'Warning' and current_incident['peak_severity'] != 'Critical':
                    current_incident['peak_severity'] = 'Warning'
                
                current_incident['count_points'] += 1
            else:
                # Save current incident and start a new one
                incidents.append(current_incident.copy())
                
                # Start new incident
                current_incident = {
                    'service': service,
                    'endpoint': endpoint,
                    'start_ts': curr_ts,
                    'end_ts': curr_ts,
                    'max_p95_latency_ms': row['p95_latency_ms'],
                    'max_error_rate': row['error_rate'],
                    'peak_severity': row['severity'],
                    'count_points': 1
                }
        
        # Don't forget to save the last incident
        incidents.append(current_incident.copy())
    
    # Sort incidents by start time descending (most recent first)
    incidents.sort(key=lambda x: x['start_ts'], reverse=True)
    
    return incidents


def find_top_culprits_local(
    df: pd.DataFrame,
    minutes: int = 15,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Local implementation of culprit finding (no API call).
    
    Parameters
    ----------
    df : pd.DataFrame
        Scored dataframe
    minutes : int
        Lookback window in minutes
    top_k : int
        Number of top culprits to return
    
    Returns
    -------
    Dict[str, Any]
        Culprit analysis result
    """
    # Get time window
    now_ts = df['bucket_ts'].max()
    window_start = now_ts - timedelta(minutes=minutes)
    
    # Filter to window and anomalies only
    mask = (df['bucket_ts'] >= window_start) & (df['bucket_ts'] <= now_ts) & (df['is_anomaly'] == True)
    anomalies = df[mask]
    
    total_anomalies = len(anomalies)
    
    if total_anomalies == 0:
        return {
            "window_minutes": minutes,
            "window_start": window_start.isoformat(),
            "window_end": now_ts.isoformat(),
            "total_anomalies": 0,
            "culprit": None,
            "top": []
        }
    
    # Aggregate per (service, endpoint)
    culprit_stats = []
    has_error_rate = 'error_rate' in anomalies.columns
    
    for (service, endpoint), group in anomalies.groupby(['service', 'endpoint']):
        peak_abs_mad = float(np.abs(group['mad_z']).max())
        peak_p95 = float(group['p95_latency_ms'].max())
        peak_error = float(group['error_rate'].max()) if has_error_rate else 0.0
        
        severity_ranks = group['severity'].map(SEVERITY_ORDER)
        peak_severity_rank = int(severity_ranks.max())
        peak_severity = [k for k, v in SEVERITY_ORDER.items() if v == peak_severity_rank][0]
        
        culprit_stats.append({
            "service": service,
            "endpoint": endpoint,
            "points": len(group),
            "peak_abs_mad": round(peak_abs_mad, 3),
            "peak_p95_latency_ms": round(peak_p95, 2),
            "peak_error_rate": round(peak_error, 4),
            "peak_severity": peak_severity,
            "peak_severity_rank": peak_severity_rank
        })
    
    # Sort by ranking criteria
    culprit_stats.sort(
        key=lambda x: (x['peak_abs_mad'], x['peak_severity_rank'], x['peak_error_rate']),
        reverse=True
    )
    
    # Remove internal rank from output
    for item in culprit_stats:
        del item['peak_severity_rank']
    
    top_culprits = culprit_stats[:top_k]
    
    return {
        "window_minutes": minutes,
        "window_start": window_start.isoformat(),
        "window_end": now_ts.isoformat(),
        "total_anomalies": total_anomalies,
        "culprit": top_culprits[0] if top_culprits else None,
        "top": top_culprits
    }


def generate_incident_explanation(
    df: pd.DataFrame,
    service: str,
    endpoint: str,
    start_ts: datetime,
    end_ts: datetime
) -> Dict[str, Any]:
    """
    Generate a local incident explanation (same logic as API).
    
    Parameters
    ----------
    df : pd.DataFrame
        Scored dataframe
    service : str
        Service name
    endpoint : str
        Endpoint path
    start_ts : datetime
        Incident start time
    end_ts : datetime
        Incident end time
    
    Returns
    -------
    Dict[str, Any]
        Incident explanation
    """
    # Filter data
    mask = (
        (df['service'] == service) &
        (df['endpoint'] == endpoint) &
        (df['bucket_ts'] >= start_ts) &
        (df['bucket_ts'] <= end_ts)
    )
    filtered = df[mask]
    
    if len(filtered) == 0:
        return {"error": "No data found for this incident"}
    
    # Compute metrics
    has_error_rate = 'error_rate' in filtered.columns
    
    peak_p95_latency_ms = float(filtered['p95_latency_ms'].max())
    peak_abs_mad_z = float(np.abs(filtered['mad_z']).max())
    peak_error_rate = float(filtered['error_rate'].max()) if has_error_rate else 0.0
    anomaly_points = int(filtered['is_anomaly'].sum())
    count_points = len(filtered)
    
    severity_ranks = filtered['severity'].map(SEVERITY_ORDER)
    peak_severity_rank = int(severity_ranks.max())
    peak_severity = [k for k, v in SEVERITY_ORDER.items() if v == peak_severity_rank][0]
    
    metrics = {
        "peak_p95_latency_ms": round(peak_p95_latency_ms, 2),
        "peak_abs_mad_z": round(peak_abs_mad_z, 3),
        "peak_error_rate": round(peak_error_rate, 4),
        "anomaly_points": anomaly_points,
        "count_points": count_points,
        "peak_severity": peak_severity
    }
    
    # Confidence
    if anomaly_points >= 5 and peak_abs_mad_z > 5.0:
        confidence = "High"
    elif anomaly_points >= 3 and peak_abs_mad_z > 3.5:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    error_elevated = peak_error_rate >= 0.05
    
    # Summary
    if peak_severity == "Critical":
        summary = f"Critical incident on {service}:{endpoint}. Peak latency {peak_p95_latency_ms:.0f}ms (MAD z: {peak_abs_mad_z:.1f})."
    elif peak_severity == "Warning":
        summary = f"Warning-level anomaly on {service}:{endpoint}. Elevated latency of {peak_p95_latency_ms:.0f}ms."
    else:
        summary = f"Minor deviation on {service}:{endpoint}. Metrics within acceptable bounds."
    
    # Why flagged
    why_flagged = [
        f"P95 latency spiked to {peak_p95_latency_ms:.0f}ms",
        f"MAD z-score of {peak_abs_mad_z:.1f} exceeded threshold"
    ]
    if error_elevated:
        why_flagged.append(f"Error rate elevated to {peak_error_rate:.1%}")
    
    # Impact
    impact = []
    if peak_severity == "Critical":
        impact = ["Users experiencing failed requests", "SLO may be breached"]
    elif peak_severity == "Warning":
        impact = ["Degraded performance", "Service functional but slow"]
    else:
        impact = ["Minimal user impact"]
    
    # Likely causes
    if error_elevated and peak_p95_latency_ms > 500:
        likely_causes = ["Downstream service failure", "Database issues", "External API failure"]
    elif peak_p95_latency_ms > 1000:
        likely_causes = ["Resource contention", "Query performance issue", "Network latency"]
    else:
        likely_causes = ["Increased traffic", "GC pauses", "Cache miss"]
    
    # Recommended actions
    recommended_actions = []
    if peak_severity == "Critical":
        recommended_actions = ["Check service logs immediately", "Verify dependencies", "Enable circuit breaker"]
    else:
        recommended_actions = ["Monitor for recurrence", "Review recent deployments", "Check resource utilization"]
    
    return {
        "service": service,
        "endpoint": endpoint,
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
        "summary": summary,
        "why_flagged": why_flagged,
        "impact": impact,
        "likely_causes": likely_causes,
        "recommended_actions": recommended_actions,
        "confidence": confidence,
        "metrics": metrics
    }


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Microservice Anomaly Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Title
    st.title("📊 Microservice Anomaly Dashboard (MAD, no training)")
    
    # =========================================================================
    # SIDEBAR: MODE CONFIGURATION
    # =========================================================================
    st.sidebar.header("⚙️ Mode Configuration")
    
    # Live mode toggle
    live_mode = st.sidebar.toggle("🔴 Live Mode", value=False, help="Poll API for real-time data")
    
    # API URL (only shown in live mode)
    if live_mode:
        api_url = st.sidebar.text_input(
            "API Base URL",
            value=DEFAULT_API_URL,
            help="FastAPI backend URL"
        )
        
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=5,
            max_value=60,
            value=DEFAULT_REFRESH_INTERVAL,
            step=5,
            help="How often to poll the API"
        )
        
        # Check API health
        api_healthy = check_api_health(api_url)
        if api_healthy:
            st.sidebar.success("✓ API Connected")
        else:
            st.sidebar.error("✗ API Unreachable")
            st.sidebar.caption(f"Make sure the API is running at {api_url}")
    else:
        api_url = DEFAULT_API_URL
        refresh_interval = DEFAULT_REFRESH_INTERVAL
    
    st.sidebar.markdown("---")
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    df = None
    data_mode = "batch"
    last_refresh = datetime.now()
    
    if live_mode:
        # Live mode: fetch from API
        df = load_data_from_api(api_url, limit=10000)
        if df is not None and len(df) > 0:
            data_mode = "live"
            st.success(f"🔴 **LIVE MODE** | {len(df):,} records | Last refresh: {last_refresh.strftime('%H:%M:%S')}")
        else:
            # Fallback to CSV if API returns no data
            df = load_data_from_csv()
            if df is not None:
                data_mode = "batch"
                st.warning(f"⚠️ API returned no data. Falling back to batch CSV ({len(df):,} records)")
    else:
        # Batch mode: read from CSV
        df = load_data_from_csv()
        if df is not None:
            st.info(f"📁 **BATCH MODE** | {len(df):,} records from {df['bucket_ts'].min()} to {df['bucket_ts'].max()}")
    
    # Check if data exists
    if df is None or len(df) == 0:
        st.error("❌ No data available!")
        st.markdown("""
        ### To get data:
        
        **Option 1: Start Live Mode**
        1. Start the API: `uvicorn backend_api.main:app --reload`
        2. Start the sender: `python -m simulate.live_sender`
        3. Enable Live Mode in sidebar
        
        **Option 2: Generate Batch Data**
        ```bash
        python -m ingest.generate_logs
        python -m ingest.feature_build
        ```
        
        Then refresh this page.
        """)
        return
    
    # =========================================================================
    # AUTO-REFRESH FOR LIVE MODE
    # =========================================================================
    if live_mode:
        # Store last refresh time
        if 'last_refresh_time' not in st.session_state:
            st.session_state.last_refresh_time = time.time()
        
        # Check if we need to refresh
        elapsed = time.time() - st.session_state.last_refresh_time
        if elapsed >= refresh_interval:
            st.session_state.last_refresh_time = time.time()
            st.rerun()
        
        # Show countdown to next refresh
        remaining = max(0, refresh_interval - elapsed)
        st.sidebar.caption(f"Next refresh in {remaining:.0f}s")
    
    # =========================================================================
    # 🔴 LIVE DEMO PANEL (Only shown in Live Mode)
    # =========================================================================
    if live_mode:
        st.markdown("---")
        st.header("🔴 LIVE DEMO MODE")
        
        # ---------------------------------------------------------------------
        # STATS SUMMARY
        # ---------------------------------------------------------------------
        stats = fetch_stats(api_url)
        
        if stats:
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric(
                    label="📊 Data Mode",
                    value=stats.get("mode", "unknown").upper(),
                    delta="Real-time" if stats.get("mode") == "live" else None
                )
            
            with col_stat2:
                st.metric(
                    label="📈 Total Data Points",
                    value=f"{stats.get('total_points', 0):,}"
                )
            
            with col_stat3:
                total_anomalies = stats.get("total_anomalies", 0)
                st.metric(
                    label="⚠️ Total Anomalies",
                    value=f"{total_anomalies:,}",
                    delta="Active" if total_anomalies > 0 else "None"
                )
            
            with col_stat4:
                by_severity = stats.get("by_severity", {})
                critical = by_severity.get("Critical", 0)
                warning = by_severity.get("Warning", 0)
                st.metric(
                    label="🔴 Critical / 🟡 Warning",
                    value=f"{critical} / {warning}"
                )
            
            # Latest timestamp
            latest_ts = stats.get("latest_bucket_ts")
            if latest_ts:
                st.caption(f"📍 Latest data: {latest_ts}")
        
        st.markdown("---")
        
        # ---------------------------------------------------------------------
        # TOP CULPRIT (Big Display)
        # ---------------------------------------------------------------------
        st.subheader("🎯 TOP CULPRIT")
        
        culprit_data = fetch_culprit(api_url, minutes=15, top_k=1)
        
        if culprit_data and culprit_data.get("culprit"):
            culprit = culprit_data["culprit"]
            
            # Big severity badge
            severity_icons = {"Critical": "🔴", "Warning": "🟡", "Info": "🟢"}
            severity_colors = {"Critical": "#ff4b4b", "Warning": "#ffa500", "Info": "#00cc00"}
            sev = culprit.get("peak_severity", "Info")
            sev_icon = severity_icons.get(sev, "⚪")
            sev_color = severity_colors.get(sev, "#888")
            
            col_badge, col_details = st.columns([1, 2])
            
            with col_badge:
                st.markdown(f"""
                <div style="background: {sev_color}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h1 style="color: white; margin: 0;">{sev_icon} {sev}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col_details:
                st.markdown(f"""
                ### `{culprit['service']}:{culprit['endpoint']}`
                
                | Metric | Value |
                |--------|-------|
                | **MAD Z-Score** | `{culprit.get('peak_abs_mad', 0):.2f}` |
                | **P95 Latency** | `{culprit.get('peak_p95_latency_ms', 0):.0f} ms` |
                | **Error Rate** | `{culprit.get('peak_error_rate', 0):.2%}` |
                | **Anomaly Points** | `{culprit.get('points', 0)}` |
                """)
        else:
            st.success("✅ **No anomalies detected** — All systems nominal")
        
        st.markdown("---")
        
        # ---------------------------------------------------------------------
        # LATEST INCIDENTS (Top 5)
        # ---------------------------------------------------------------------
        st.subheader("🚨 LATEST INCIDENTS")
        
        incidents_data = fetch_incidents(api_url, {"limit": 5})
        
        if incidents_data and len(incidents_data.get("data", [])) > 0:
            incidents_list = incidents_data["data"]
            
            # Find most severe incident
            most_severe_idx = 0
            most_severe_rank = -1
            for i, inc in enumerate(incidents_list):
                rank = SEVERITY_ORDER.get(inc.get("peak_severity", "Info"), 0)
                if rank > most_severe_rank:
                    most_severe_rank = rank
                    most_severe_idx = i
            
            # Store for selection
            if 'selected_incident_idx' not in st.session_state:
                st.session_state.selected_incident_idx = most_severe_idx
            
            # Display incidents as cards
            for i, inc in enumerate(incidents_list):
                sev = inc.get("peak_severity", "Info")
                sev_icon = {"Critical": "🔴", "Warning": "🟡", "Info": "🟢"}.get(sev, "⚪")
                is_selected = (i == st.session_state.selected_incident_idx)
                is_most_severe = (i == most_severe_idx)
                
                # Highlight most severe
                border = "3px solid #ff4b4b" if is_most_severe else "1px solid #444"
                bg = "#1a1a2e" if is_selected else "#0e1117"
                
                col_inc, col_btn = st.columns([4, 1])
                
                with col_inc:
                    st.markdown(f"""
                    <div style="background: {bg}; border: {border}; padding: 10px; border-radius: 8px; margin-bottom: 5px;">
                        <strong>{sev_icon} {inc['service']}:{inc['endpoint']}</strong>
                        <br/>
                        <small>⏱️ {inc['start_ts']} — {inc['end_ts']}</small>
                        <br/>
                        <small>📊 P95: {inc.get('max_p95_latency_ms', 0):.0f}ms | Errors: {float(inc.get('max_error_rate', 0)):.1%} | Points: {inc.get('count_points', 0)}</small>
                        {' <span style="color: #ff4b4b; font-weight: bold;">← MOST SEVERE</span>' if is_most_severe else ''}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_btn:
                    if st.button("Select", key=f"select_inc_{i}", use_container_width=True):
                        st.session_state.selected_incident_idx = i
                        st.rerun()
            
            # ---------------------------------------------------------------------
            # INCIDENT EXPLANATION
            # ---------------------------------------------------------------------
            st.markdown("---")
            st.subheader("💡 INCIDENT EXPLANATION")
            
            selected_inc = incidents_list[st.session_state.selected_incident_idx]
            
            st.markdown(f"**Selected:** `{selected_inc['service']}:{selected_inc['endpoint']}` ({selected_inc['start_ts']} — {selected_inc['end_ts']})")
            
            if st.button("📋 Explain This Incident", use_container_width=True, type="primary"):
                # Try API first
                explanation = fetch_explain_incident(
                    api_url,
                    selected_inc['service'],
                    selected_inc['endpoint'],
                    selected_inc['start_ts'],
                    selected_inc['end_ts']
                )
                
                # Fallback to local template if API fails
                if explanation is None:
                    explanation = generate_local_explanation(selected_inc)
                
                st.session_state.current_explanation = explanation
            
            # Display explanation if available
            if 'current_explanation' in st.session_state and st.session_state.current_explanation:
                exp = st.session_state.current_explanation
                
                if 'error' not in exp:
                    # Confidence badge
                    conf = exp.get('confidence', 'Unknown')
                    conf_icons = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
                    conf_icon = conf_icons.get(conf, "⚪")
                    
                    st.markdown(f"### {conf_icon} Confidence: **{conf}**")
                    st.markdown(f"**Summary:** {exp.get('summary', 'N/A')}")
                    
                    col_why, col_action = st.columns(2)
                    
                    with col_why:
                        st.markdown("**🚩 Why Flagged:**")
                        for item in exp.get('why_flagged', []):
                            st.markdown(f"- {item}")
                        
                        st.markdown("**⚠️ Impact:**")
                        for item in exp.get('impact', []):
                            st.markdown(f"- {item}")
                    
                    with col_action:
                        st.markdown("**🔍 Likely Causes:**")
                        for item in exp.get('likely_causes', []):
                            st.markdown(f"- {item}")
                        
                        st.markdown("**✅ Actions:**")
                        for item in exp.get('recommended_actions', []):
                            st.markdown(f"- {item}")
                else:
                    st.error(exp['error'])
        else:
            st.success("✅ **No incidents detected** — System healthy")
        
        st.markdown("---")
    
    # =========================================================================
    # DEMO MODE: CULPRIT ANALYSIS
    # =========================================================================
    st.header("🎯 Culprit Analysis")
    
    col_culprit_1, col_culprit_2 = st.columns([1, 3])
    
    with col_culprit_1:
        window_minutes = st.number_input(
            "Lookback Window (minutes)",
            min_value=1,
            max_value=120,
            value=15,
            step=5,
            help="How far back to look for anomalies"
        )
        refresh_culprit = st.button("🔄 Refresh Culprit", use_container_width=True)
    
    # Compute culprit (use API in live mode, local in batch mode)
    if 'culprit_result' not in st.session_state or refresh_culprit or live_mode:
        if live_mode:
            culprit_result = fetch_culprit(api_url, minutes=window_minutes, top_k=3)
            if culprit_result is None:
                culprit_result = find_top_culprits_local(df, minutes=window_minutes, top_k=3)
        else:
            culprit_result = find_top_culprits_local(df, minutes=window_minutes, top_k=3)
        st.session_state.culprit_result = culprit_result
    else:
        culprit_result = st.session_state.culprit_result
    
    with col_culprit_2:
        if culprit_result['culprit'] is not None:
            c = culprit_result['culprit']
            
            # Severity color
            severity_colors = {"Critical": "🔴", "Warning": "🟡", "Info": "🟢"}
            sev_icon = severity_colors.get(c['peak_severity'], "⚪")
            
            # Display culprit badge
            st.markdown(f"""
            ### {sev_icon} Top Culprit: `{c['service']}:{c['endpoint']}`
            
            | Metric | Value |
            |--------|-------|
            | **Peak MAD Z-Score** | {c['peak_abs_mad']:.2f} |
            | **Peak Severity** | {c['peak_severity']} |
            | **Peak P95 Latency** | {c['peak_p95_latency_ms']:.0f} ms |
            | **Peak Error Rate** | {c['peak_error_rate']:.2%} |
            | **Anomaly Points** | {c['points']} |
            """)
        else:
            st.info(f"✅ No anomalies detected in the last {window_minutes} minutes")
    
    # Top suspects table
    if culprit_result['top'] and len(culprit_result['top']) > 0:
        st.subheader("🔍 Top Suspects")
        
        suspects_data = []
        for i, suspect in enumerate(culprit_result['top'], start=1):
            suspects_data.append({
                "Rank": f"#{i}",
                "Service": suspect['service'],
                "Endpoint": suspect['endpoint'],
                "MAD Z": f"{suspect['peak_abs_mad']:.2f}",
                "Severity": suspect['peak_severity'],
                "P95 Latency": f"{suspect['peak_p95_latency_ms']:.0f} ms",
                "Error Rate": f"{suspect['peak_error_rate']:.2%}",
                "Points": suspect['points']
            })
        
        suspects_df = pd.DataFrame(suspects_data)
        st.dataframe(suspects_df, use_container_width=True, hide_index=True)
        
        st.caption(f"Window: {culprit_result['window_start']} to {culprit_result['window_end']} ({culprit_result['total_anomalies']} total anomalies)")
    
    st.markdown("---")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Service filter
    services = sorted(df['service'].unique())
    selected_service = st.sidebar.selectbox(
        "Service",
        options=['All'] + services,
        index=0
    )
    
    # Endpoint filter (depends on service selection)
    if selected_service == 'All':
        endpoints = sorted(df['endpoint'].unique())
    else:
        endpoints = sorted(df[df['service'] == selected_service]['endpoint'].unique())
    
    selected_endpoint = st.sidebar.selectbox(
        "Endpoint",
        options=['All'] + endpoints,
        index=0
    )
    
    # Severity filter
    severity_options = ['All', 'Info', 'Warning', 'Critical']
    selected_severity = st.sidebar.selectbox(
        "Severity",
        options=severity_options,
        index=0
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_service != 'All':
        filtered_df = filtered_df[filtered_df['service'] == selected_service]
    
    if selected_endpoint != 'All':
        filtered_df = filtered_df[filtered_df['endpoint'] == selected_endpoint]
    
    if selected_severity != 'All':
        filtered_df = filtered_df[filtered_df['severity'] == selected_severity]
    
    # Display filter summary
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df):,}")
    anomaly_count = filtered_df['is_anomaly'].sum()
    st.sidebar.markdown(f"**Anomalies:** {anomaly_count:,} ({anomaly_count/len(filtered_df)*100:.1f}%)")
    
    # Main content area
    
    # 1. Line chart of p95_latency_ms over time
    st.header("📈 P95 Latency Over Time")
    
    if len(filtered_df) > 0:
        # Prepare data for line chart
        chart_data = filtered_df.sort_values('bucket_ts')[['bucket_ts', 'p95_latency_ms']].set_index('bucket_ts')
        
        # Display chart
        st.line_chart(chart_data, height=400)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean P95 Latency", f"{filtered_df['p95_latency_ms'].mean():.1f} ms")
        with col2:
            st.metric("Max P95 Latency", f"{filtered_df['p95_latency_ms'].max():.1f} ms")
        with col3:
            st.metric("Mean Error Rate", f"{filtered_df['error_rate'].mean():.2%}")
        with col4:
            st.metric("Total Requests", f"{filtered_df['req_count'].sum():,}")
    else:
        st.info("No data matching the selected filters.")
    
    # 2. Anomaly points table
    st.header("🔍 Detected Anomalies")
    
    anomalies_df = filtered_df[filtered_df['is_anomaly'] == True].copy()
    
    if len(anomalies_df) > 0:
        # Sort by timestamp descending
        anomalies_df = anomalies_df.sort_values('bucket_ts', ascending=False)
        
        # Select columns to display
        display_columns = [
            'bucket_ts', 'service', 'endpoint', 
            'p95_latency_ms', 'error_rate', 'mad_z', 'severity'
        ]
        
        # Format the dataframe for display
        display_df = anomalies_df[display_columns].copy()
        display_df['error_rate'] = display_df['error_rate'].apply(lambda x: f"{x:.2%}")
        display_df['mad_z'] = display_df['mad_z'].apply(lambda x: f"{x:.2f}")
        display_df['p95_latency_ms'] = display_df['p95_latency_ms'].apply(lambda x: f"{x:.1f}")
        
        # Rename columns for better readability
        display_df.columns = [
            'Timestamp', 'Service', 'Endpoint', 
            'P95 Latency (ms)', 'Error Rate', 'MAD Z-Score', 'Severity'
        ]
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        st.info(f"Showing {len(anomalies_df)} anomalies")
    else:
        st.success("✓ No anomalies detected with current filters")
    
    # 3. Incidents summary table
    st.header("🚨 Incident Summary")
    
    # Compute incidents from the filtered data
    incidents = compute_incidents(filtered_df)
    
    if len(incidents) > 0:
        # Convert to DataFrame for display
        incidents_df = pd.DataFrame(incidents)
        
        # Keep original timestamps for selection
        incidents_df_original = incidents_df.copy()
        
        # Format timestamps for display
        incidents_df['start_ts'] = incidents_df['start_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
        incidents_df['end_ts'] = incidents_df['end_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format numeric columns
        incidents_df['max_p95_latency_ms'] = incidents_df['max_p95_latency_ms'].apply(lambda x: f"{x:.1f}")
        incidents_df['max_error_rate'] = incidents_df['max_error_rate'].apply(lambda x: f"{x:.2%}")
        
        # Rename columns for display
        incidents_df.columns = [
            'Service', 'Endpoint', 'Start Time', 'End Time',
            'Max P95 Latency (ms)', 'Max Error Rate', 'Peak Severity', 'Data Points'
        ]
        
        st.dataframe(
            incidents_df,
            use_container_width=True,
            hide_index=True
        )
        
        st.info(f"Found {len(incidents)} incident(s)")
        
        # Add explanation
        with st.expander("ℹ️ What is an incident?"):
            st.markdown("""
            An **incident** is a group of consecutive anomalous data points 
            for the same service and endpoint. The dashboard groups anomalies 
            that occur within 2 minutes of each other into a single incident 
            window, providing a high-level view of significant events.
            """)
        
        # =====================================================================
        # INCIDENT EXPLANATION SECTION
        # =====================================================================
        st.subheader("💡 Incident Explanation")
        
        # Create incident selection options
        incident_options = []
        for i, inc in enumerate(incidents):
            label = f"{inc['service']}:{inc['endpoint']} ({inc['start_ts'].strftime('%H:%M')} - {inc['end_ts'].strftime('%H:%M')}, {inc['peak_severity']})"
            incident_options.append(label)
        
        selected_incident_idx = st.selectbox(
            "Select an incident to explain:",
            options=range(len(incident_options)),
            format_func=lambda x: incident_options[x],
            key="incident_selector"
        )
        
        if st.button("📋 Generate Explanation", use_container_width=False):
            selected_incident = incidents[selected_incident_idx]
            
            # Generate explanation
            explanation = generate_incident_explanation(
                df=df,
                service=selected_incident['service'],
                endpoint=selected_incident['endpoint'],
                start_ts=selected_incident['start_ts'],
                end_ts=selected_incident['end_ts']
            )
            
            if 'error' not in explanation:
                # Display explanation in a nice format
                st.markdown("---")
                
                # Confidence badge
                confidence_colors = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
                conf_icon = confidence_colors.get(explanation['confidence'], "⚪")
                
                st.markdown(f"### {conf_icon} Confidence: {explanation['confidence']}")
                
                # Summary
                st.markdown(f"**Summary:** {explanation['summary']}")
                
                # Two columns for details
                col_exp_1, col_exp_2 = st.columns(2)
                
                with col_exp_1:
                    st.markdown("**🚩 Why Flagged:**")
                    for item in explanation['why_flagged']:
                        st.markdown(f"- {item}")
                    
                    st.markdown("**⚠️ Impact:**")
                    for item in explanation['impact']:
                        st.markdown(f"- {item}")
                
                with col_exp_2:
                    st.markdown("**🔍 Likely Causes:**")
                    for item in explanation['likely_causes']:
                        st.markdown(f"- {item}")
                    
                    st.markdown("**✅ Recommended Actions:**")
                    for item in explanation['recommended_actions']:
                        st.markdown(f"- {item}")
                
                # Metrics expander
                with st.expander("📊 Raw Metrics"):
                    metrics_data = {
                        "Metric": list(explanation['metrics'].keys()),
                        "Value": list(explanation['metrics'].values())
                    }
                    st.table(pd.DataFrame(metrics_data))
            else:
                st.error(explanation['error'])
    else:
        st.success("✓ No incidents detected with current filters")
    
    # Footer
    st.markdown("---")
    st.caption("Dashboard powered by Streamlit | Anomaly detection using MAD (Median Absolute Deviation)")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

"""
FastAPI Backend for Microservice Anomaly Detection

This API provides endpoints to query metrics, anomalies, and incidents
from the analyzed microservice logs.

Run Instructions:
-----------------
From the repository root, run:
    uvicorn backend_api.main:app --reload

Or with custom host/port:
    uvicorn backend_api.main:app --reload --host 0.0.0.0 --port 8000

Dependencies:
-------------
    pip install fastapi uvicorn pandas pydantic
"""

import os
import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ingest.culprit import find_top_culprits
from backend_api import telemetry_store
from backend_api import realtime_processor


# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORED_DATA_PATH = "data/scored_1min.csv"
REALTIME_LOOKBACK_MINUTES = 120  # Lookback window for real-time scoring

app = FastAPI(
    title="Microservice Anomaly Detection API",
    description="API for querying microservice metrics and detected anomalies",
    version="1.0.0"
)


# =============================================================================
# PYDANTIC MODELS FOR LIVE INGESTION
# =============================================================================

class LogEvent(BaseModel):
    """Single log event from a microservice."""
    timestamp: str = Field(..., description="ISO format timestamp (e.g., 2026-01-07T10:00:00)")
    service: str = Field(..., description="Service name (e.g., 'auth', 'payments')")
    endpoint: str = Field(..., description="Endpoint path (e.g., '/login', '/pay')")
    status_code: int = Field(..., description="HTTP status code (e.g., 200, 500)")
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    trace_id: Optional[str] = Field(None, description="Optional trace ID for distributed tracing")


class LogEventBatch(BaseModel):
    """Batch of log events for bulk ingestion."""
    events: List[LogEvent] = Field(..., description="List of log events")


# =============================================================================
# DATA LOADING
# =============================================================================

# Expected columns in scored DataFrame
SCORED_COLUMNS = [
    "bucket_ts", "service", "endpoint", "req_count", "error_count",
    "error_rate", "avg_latency_ms", "p95_latency_ms", "mad_z", "is_anomaly", "severity"
]

# Track current data mode
_current_mode = "batch"


def load_scored_data() -> pd.DataFrame:
    """
    Load scored data from real-time store or fallback to CSV file.
    
    Priority:
    1. Real-time in-memory data (if available and has rows)
    2. Batch CSV file (data/scored_1min.csv)
    3. Empty DataFrame with correct columns
    
    Returns
    -------
    pd.DataFrame
        Scored data with anomaly detection results.
        bucket_ts column is always datetime type.
    """
    global _current_mode
    
    # Try real-time data first
    realtime_df = realtime_processor.get_scored_df()
    if len(realtime_df) > 0:
        _current_mode = "live"
        df = realtime_df.copy()
        # Ensure bucket_ts is datetime
        if df['bucket_ts'].dtype == 'object':
            df['bucket_ts'] = pd.to_datetime(df['bucket_ts'])
        logger.info(f"Using live data: {len(df)} rows")
        return df
    
    # Fallback to CSV file
    if os.path.exists(SCORED_DATA_PATH):
        _current_mode = "batch"
        df = pd.read_csv(SCORED_DATA_PATH)
        df['bucket_ts'] = pd.to_datetime(df['bucket_ts'])
        logger.info(f"Using batch CSV data: {len(df)} rows")
        return df
    
    # No data available - return empty DataFrame with correct columns
    _current_mode = "batch"
    logger.warning("No scored data available (neither live nor batch)")
    return pd.DataFrame(columns=SCORED_COLUMNS)


def get_data() -> pd.DataFrame:
    """
    Get scored data with error handling.
    
    Returns
    -------
    pd.DataFrame
        Scored data (may be empty if no data available)
    
    Raises
    ------
    HTTPException
        If data cannot be loaded due to an error
    """
    try:
        return load_scored_data()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to load data",
                "message": str(e)
            }
        )


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns
    -------
    Dict[str, str]
        Status information
    """
    return {"status": "ok"}


@app.get("/stats")
def get_stats() -> Dict[str, Any]:
    """
    Get overall statistics about the scored data.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - mode: "live" or "batch" indicating data source
        - total_points: Total number of data points
        - total_anomalies: Number of anomalies detected
        - by_severity: Breakdown by severity level
        - latest_bucket_ts: Most recent timestamp or null
    """
    df = get_data()
    
    if len(df) == 0:
        return {
            "mode": _current_mode,
            "total_points": 0,
            "total_anomalies": 0,
            "by_severity": {"Info": 0, "Warning": 0, "Critical": 0},
            "latest_bucket_ts": None
        }
    
    # Count by severity
    severity_counts = df['severity'].value_counts().to_dict()
    by_severity = {
        "Info": severity_counts.get("Info", 0),
        "Warning": severity_counts.get("Warning", 0),
        "Critical": severity_counts.get("Critical", 0)
    }
    
    # Get latest timestamp
    latest_ts = df['bucket_ts'].max()
    if pd.notna(latest_ts):
        latest_ts_str = latest_ts.strftime('%Y-%m-%dT%H:%M:%S')
    else:
        latest_ts_str = None
    
    return {
        "mode": _current_mode,
        "total_points": len(df),
        "total_anomalies": int(df['is_anomaly'].sum()),
        "by_severity": by_severity,
        "latest_bucket_ts": latest_ts_str
    }


@app.get("/metrics")
def get_metrics(
    service: Optional[str] = Query(None, description="Filter by service name"),
    endpoint: Optional[str] = Query(None, description="Filter by endpoint"),
    limit: int = Query(200, ge=1, le=10000, description="Maximum number of rows to return")
) -> Dict[str, Any]:
    """
    Get recent metrics data.
    
    Parameters
    ----------
    service : Optional[str]
        Filter by service name
    endpoint : Optional[str]
        Filter by endpoint
    limit : int
        Maximum number of rows to return (default: 200)
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - count: Number of rows returned
        - data: List of metric records
    """
    df = get_data()
    
    # Apply filters
    if service:
        df = df[df['service'] == service]
    
    if endpoint:
        df = df[df['endpoint'] == endpoint]
    
    # Sort by timestamp descending (most recent first)
    df = df.sort_values('bucket_ts', ascending=False)
    
    # Apply limit
    df = df.head(limit)
    
    # Convert to records for JSON response
    # Convert timestamp to string for JSON serialization
    df['bucket_ts'] = df['bucket_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
    records = df.to_dict('records')
    
    return {
        "count": len(records),
        "data": records
    }


@app.get("/anomalies")
def get_anomalies(
    service: Optional[str] = Query(None, description="Filter by service name"),
    endpoint: Optional[str] = Query(None, description="Filter by endpoint"),
    severity: Optional[str] = Query(None, description="Filter by severity (Info, Warning, Critical)"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of anomalies to return")
) -> Dict[str, Any]:
    """
    Get detected anomalies with optional filters.
    
    Parameters
    ----------
    service : Optional[str]
        Filter by service name
    endpoint : Optional[str]
        Filter by endpoint
    severity : Optional[str]
        Filter by severity level (Info, Warning, Critical)
    limit : int
        Maximum number of anomalies to return (default: 50)
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - count: Number of anomalies returned
        - filters: Applied filters
        - data: List of anomaly records
    """
    df = get_data()
    
    # Filter only anomalies
    df = df[df['is_anomaly'] == True]
    
    # Apply optional filters
    applied_filters = {}
    
    if service:
        df = df[df['service'] == service]
        applied_filters['service'] = service
    
    if endpoint:
        df = df[df['endpoint'] == endpoint]
        applied_filters['endpoint'] = endpoint
    
    if severity:
        df = df[df['severity'] == severity]
        applied_filters['severity'] = severity
    
    # Sort by timestamp descending (most recent first)
    df = df.sort_values('bucket_ts', ascending=False)
    
    # Apply limit
    df = df.head(limit)
    
    # Convert to records for JSON response
    df['bucket_ts'] = df['bucket_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
    records = df.to_dict('records')
    
    return {
        "count": len(records),
        "filters": applied_filters,
        "data": records
    }


@app.get("/incidents")
def get_incidents(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of incidents to return")
) -> Dict[str, Any]:
    """
    Get grouped incident windows from anomalies.
    
    This endpoint groups consecutive anomalies by service and endpoint into
    incident windows, providing a high-level view of significant events.
    
    Parameters
    ----------
    limit : int
        Maximum number of incidents to return (default: 20)
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - count: Number of incidents returned
        - data: List of incident objects with:
            - service: Service name
            - endpoint: Endpoint path
            - start_ts: Incident start timestamp
            - end_ts: Incident end timestamp
            - max_p95_latency_ms: Peak latency during incident
            - max_error_rate: Peak error rate during incident
            - peak_severity: Highest severity level
            - count_points: Number of anomalous data points
    """
    df = get_data()
    
    # Filter only anomalies
    anomalies = df[df['is_anomaly'] == True].copy()
    
    if len(anomalies) == 0:
        return {
            "count": 0,
            "data": []
        }
    
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
            'count_points': 1,
            'timestamps': [group.iloc[0]['bucket_ts']]
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
                current_incident['timestamps'].append(curr_ts)
            else:
                # Save current incident and start a new one
                # Remove timestamps list before saving
                del current_incident['timestamps']
                incidents.append(current_incident)
                
                # Start new incident
                current_incident = {
                    'service': service,
                    'endpoint': endpoint,
                    'start_ts': curr_ts,
                    'end_ts': curr_ts,
                    'max_p95_latency_ms': row['p95_latency_ms'],
                    'max_error_rate': row['error_rate'],
                    'peak_severity': row['severity'],
                    'count_points': 1,
                    'timestamps': [curr_ts]
                }
        
        # Don't forget to save the last incident
        del current_incident['timestamps']
        incidents.append(current_incident)
    
    # Sort incidents by start time descending (most recent first)
    incidents.sort(key=lambda x: x['start_ts'], reverse=True)
    
    # Apply limit
    incidents = incidents[:limit]
    
    # Convert timestamps to strings for JSON serialization
    for incident in incidents:
        incident['start_ts'] = incident['start_ts'].strftime('%Y-%m-%d %H:%M:%S')
        incident['end_ts'] = incident['end_ts'].strftime('%Y-%m-%d %H:%M:%S')
    
    return {
        "count": len(incidents),
        "data": incidents
    }


@app.get("/culprit")
def get_culprit(
    minutes: int = Query(15, ge=1, le=1440, description="Lookback window in minutes"),
    top_k: int = Query(3, ge=1, le=20, description="Number of top culprits to return")
) -> Dict[str, Any]:
    """
    Get the top culprit service/endpoint combinations from recent anomalies.
    
    This endpoint analyzes recent anomalies and ranks service/endpoint combinations
    by severity to identify the most likely root causes of issues.
    
    Parameters
    ----------
    minutes : int
        Number of minutes to look back (default: 15, max: 1440 = 24 hours)
    top_k : int
        Number of top culprits to return (default: 3, max: 20)
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - window_minutes: The lookback window used
        - window_start: ISO timestamp of window start
        - window_end: ISO timestamp of window end
        - total_anomalies: Total anomaly rows in window
        - culprit: Top culprit details (or null if no anomalies)
        - top: List of top_k culprits ranked by severity
    
    Ranking Logic
    -------------
    Culprits are ranked by (in priority order):
    1. peak_abs_mad (descending) - Highest absolute MAD z-score
    2. peak_severity (descending) - Critical > Warning > Info
    3. peak_error_rate (descending) - Highest error rate
    """
    df = get_data()
    
    # Call the culprit finder
    result = find_top_culprits(
        df_scored=df,
        minutes=minutes,
        now_ts=None,  # Use latest timestamp in data
        top_k=top_k
    )
    
    return result


@app.get("/explain_incident")
def explain_incident(
    service: str = Query(..., description="Service name (required)"),
    endpoint: str = Query(..., description="Endpoint path (required)"),
    start_ts: str = Query(..., description="Start timestamp in ISO format (required)"),
    end_ts: str = Query(..., description="End timestamp in ISO format (required)")
) -> Dict[str, Any]:
    """
    Get a deterministic explanation for an incident.
    
    This endpoint analyzes the specified incident window and generates
    a templated explanation based on the observed metrics and severity.
    No external API calls are made - all explanations are rule-based.
    
    Parameters
    ----------
    service : str
        Service name (required)
    endpoint : str
        Endpoint path (required)
    start_ts : str
        Start timestamp in ISO format (required)
    end_ts : str
        End timestamp in ISO format (required)
    
    Returns
    -------
    Dict[str, Any]
        Incident explanation containing:
        - service, endpoint, start_ts, end_ts
        - summary: Brief description of the incident
        - why_flagged: Reasons the anomaly was detected
        - impact: Potential user/business impact
        - likely_causes: Possible root causes
        - recommended_actions: Suggested remediation steps
        - confidence: Low, Medium, or High
        - metrics: Computed metric values
    """
    df = get_data()
    
    # Parse timestamps
    try:
        start_dt = pd.to_datetime(start_ts)
        end_dt = pd.to_datetime(end_ts)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid timestamp format",
                "message": f"Could not parse timestamps: {e}",
                "expected_format": "ISO format, e.g., 2026-01-06T10:45:00"
            }
        )
    
    # Filter to the specified service, endpoint, and time window
    mask = (
        (df['service'] == service) &
        (df['endpoint'] == endpoint) &
        (df['bucket_ts'] >= start_dt) &
        (df['bucket_ts'] <= end_dt)
    )
    filtered = df[mask]
    
    if len(filtered) == 0:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "No data found",
                "message": f"No records found for {service}:{endpoint} between {start_ts} and {end_ts}",
                "suggestion": "Check that the service, endpoint, and time range are correct"
            }
        )
    
    # Compute metrics
    has_error_rate = 'error_rate' in filtered.columns
    
    peak_p95_latency_ms = float(filtered['p95_latency_ms'].max())
    peak_abs_mad_z = float(filtered['mad_z'].abs().max())
    avg_abs_mad_z = float(filtered['mad_z'].abs().mean())
    peak_error_rate = float(filtered['error_rate'].max()) if has_error_rate else 0.0
    avg_error_rate = float(filtered['error_rate'].mean()) if has_error_rate else 0.0
    count_points = len(filtered)
    anomaly_points = int(filtered['is_anomaly'].sum())
    
    # Determine peak severity
    severity_order = {"Info": 0, "Warning": 1, "Critical": 2}
    severity_ranks = filtered['severity'].map(severity_order)
    peak_severity_rank = int(severity_ranks.max())
    peak_severity = [k for k, v in severity_order.items() if v == peak_severity_rank][0]
    
    # Build metrics dict
    metrics = {
        "peak_p95_latency_ms": round(peak_p95_latency_ms, 2),
        "peak_abs_mad_z": round(peak_abs_mad_z, 3),
        "avg_abs_mad_z": round(avg_abs_mad_z, 3),
        "peak_error_rate": round(peak_error_rate, 4),
        "avg_error_rate": round(avg_error_rate, 4),
        "count_points": count_points,
        "anomaly_points": anomaly_points,
        "peak_severity": peak_severity
    }
    
    # Determine confidence based on data quality
    if anomaly_points >= 5 and peak_abs_mad_z > 5.0:
        confidence = "High"
    elif anomaly_points >= 3 and peak_abs_mad_z > 3.5:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # Determine if error rate is elevated (> 5%)
    error_elevated = peak_error_rate >= 0.05
    
    # Generate templated explanations based on severity and error rate
    # Summary
    if peak_severity == "Critical":
        summary = (
            f"Critical incident detected on {service}:{endpoint}. "
            f"Peak latency reached {peak_p95_latency_ms:.0f}ms (MAD z-score: {peak_abs_mad_z:.1f}) "
            f"with {anomaly_points} anomalous data points over {count_points} minutes."
        )
    elif peak_severity == "Warning":
        summary = (
            f"Warning-level anomaly detected on {service}:{endpoint}. "
            f"Elevated latency of {peak_p95_latency_ms:.0f}ms observed (MAD z-score: {peak_abs_mad_z:.1f})."
        )
    else:
        summary = (
            f"Minor deviation detected on {service}:{endpoint}. "
            f"Metrics showed slight elevation but remained within acceptable bounds."
        )
    
    # Why flagged
    why_flagged = []
    why_flagged.append(f"P95 latency spiked to {peak_p95_latency_ms:.0f}ms")
    why_flagged.append(f"MAD z-score of {peak_abs_mad_z:.1f} exceeded threshold of 3.5")
    if error_elevated:
        why_flagged.append(f"Error rate elevated to {peak_error_rate:.1%} (threshold: 5%)")
    if anomaly_points >= 3:
        why_flagged.append(f"Sustained anomaly over {anomaly_points} consecutive data points")
    
    # Impact
    impact = []
    if peak_severity == "Critical":
        impact.append("Users likely experiencing failed requests or timeouts")
        impact.append("Service reliability SLO may be breached")
        if error_elevated:
            impact.append(f"Approximately {peak_error_rate:.1%} of requests are failing")
    elif peak_severity == "Warning":
        impact.append("Users may experience degraded performance")
        impact.append("Response times elevated but service is functional")
    else:
        impact.append("Minimal user impact expected")
        impact.append("Performance slightly degraded but within tolerance")
    
    if peak_p95_latency_ms > 1000:
        impact.append("Latency exceeds 1 second, causing poor user experience")
    
    # Likely causes (templated based on patterns)
    likely_causes = []
    
    if error_elevated and peak_p95_latency_ms > 500:
        likely_causes.append("Downstream service failure or timeout")
        likely_causes.append("Database connection pool exhaustion")
        likely_causes.append("External API dependency failure")
    elif peak_p95_latency_ms > 1000:
        likely_causes.append("Resource contention (CPU/memory pressure)")
        likely_causes.append("Database query performance degradation")
        likely_causes.append("Network latency spike")
    elif error_elevated:
        likely_causes.append("Application exception or bug")
        likely_causes.append("Invalid input data causing failures")
        likely_causes.append("Authentication/authorization failures")
    else:
        likely_causes.append("Increased traffic load")
        likely_causes.append("Garbage collection pauses")
        likely_causes.append("Cold start or cache miss")
    
    # Add service-specific hints based on endpoint patterns
    if "pay" in endpoint.lower() or "checkout" in endpoint.lower():
        likely_causes.append("Payment gateway latency or failure")
    elif "auth" in endpoint.lower() or "login" in endpoint.lower():
        likely_causes.append("Identity provider slowdown")
    elif "search" in endpoint.lower():
        likely_causes.append("Search index performance issue")
    
    # Recommended actions
    recommended_actions = []
    
    if peak_severity == "Critical":
        recommended_actions.append("Immediately check service health and logs")
        recommended_actions.append("Verify downstream dependencies are healthy")
        recommended_actions.append("Consider enabling circuit breaker if available")
        if error_elevated:
            recommended_actions.append("Check error logs for exception stack traces")
            recommended_actions.append("Verify database connectivity and pool status")
    
    if peak_p95_latency_ms > 1000:
        recommended_actions.append("Review recent deployments for performance regressions")
        recommended_actions.append("Check resource utilization (CPU, memory, connections)")
    
    recommended_actions.append("Review metrics dashboard for correlated issues")
    recommended_actions.append("Check if issue correlates with traffic spike")
    
    if len(recommended_actions) < 3:
        recommended_actions.append("Monitor for recurrence")
        recommended_actions.append("Consider adding alerting for this endpoint")
    
    # Build response
    return {
        "service": service,
        "endpoint": endpoint,
        "start_ts": start_dt.isoformat(),
        "end_ts": end_dt.isoformat(),
        "summary": summary,
        "why_flagged": why_flagged,
        "impact": impact,
        "likely_causes": likely_causes[:5],  # Limit to top 5
        "recommended_actions": recommended_actions[:5],  # Limit to top 5
        "confidence": confidence,
        "metrics": metrics
    }


# =============================================================================
# LIVE INGESTION ENDPOINT
# =============================================================================

@app.post("/ingest")
def ingest_events(
    payload: Union[LogEvent, LogEventBatch]
) -> Dict[str, Any]:
    """
    Ingest live log events into the telemetry store.

    This endpoint accepts either a single LogEvent or a batch of events.
    Events are stored in memory for real-time analysis.

    Parameters
    ----------
    payload : Union[LogEvent, LogEventBatch]
        Either a single LogEvent object or a LogEventBatch containing
        a list of events.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - accepted: Number of events successfully stored
        - retention_minutes: Current retention window

    Example Request (single event):
    -------------------------------
    POST /ingest
    {
        "timestamp": "2026-01-07T10:00:00",
        "service": "auth",
        "endpoint": "/login",
        "status_code": 200,
        "latency_ms": 45.2,
        "trace_id": "abc123"
    }

    Example Request (batch):
    ------------------------
    POST /ingest
    {
        "events": [
            {"timestamp": "2026-01-07T10:00:00", "service": "auth", ...},
            {"timestamp": "2026-01-07T10:00:01", "service": "payments", ...}
        ]
    }

    Example Response:
    -----------------
    {
        "accepted": 2,
        "retention_minutes": 180,
        "scored_rows": 10
    }
    """
    # Determine if single event or batch
    if isinstance(payload, LogEvent):
        events = [payload.model_dump()]
    else:
        events = [e.model_dump() for e in payload.events]

    # Store events
    accepted_count = telemetry_store.add_events(events)

    # Prune old events
    pruned_count = telemetry_store.prune_older_than(telemetry_store.RETENTION_MINUTES)

    # Rebuild real-time scores from all events
    all_events = telemetry_store.get_all_events()
    scored_df = realtime_processor.rebuild_scores(all_events, lookback_minutes=REALTIME_LOOKBACK_MINUTES)
    realtime_processor.set_scored_df(scored_df)
    scored_rows = len(scored_df)

    # Log the ingestion
    logger.info(f"Ingested {accepted_count} events (pruned {pruned_count} old, scored {scored_rows} rows)")

    return {
        "accepted": accepted_count,
        "retention_minutes": telemetry_store.RETENTION_MINUTES,
        "scored_rows": scored_rows
    }


@app.get("/ingest/stats")
def get_ingest_stats() -> Dict[str, Any]:
    """
    Get statistics about the live telemetry store.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - count: Number of events in store
        - max_capacity: Maximum events allowed
        - oldest_ts: Oldest event timestamp
        - newest_ts: Newest event timestamp
        - retention_minutes: Retention window setting
        - realtime_stats: Real-time scoring statistics
    """
    stats = telemetry_store.get_store_stats()
    stats["retention_minutes"] = telemetry_store.RETENTION_MINUTES
    stats["realtime_stats"] = realtime_processor.get_realtime_stats()
    return stats


# =============================================================================
# STARTUP EVENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    print("=" * 70)
    print("Microservice Anomaly Detection API")
    print("=" * 70)
    print(f"Scored data path: {SCORED_DATA_PATH}")
    
    try:
        df = load_scored_data()
        print(f"✓ Loaded {len(df):,} scored records")
        print(f"  Time range: {df['bucket_ts'].min()} to {df['bucket_ts'].max()}")
        anomaly_count = df['is_anomaly'].sum()
        print(f"  Anomalies: {anomaly_count:,} ({anomaly_count/len(df)*100:.1f}%)")
    except FileNotFoundError:
        print("✗ Scored data not found")
        print("  Run: python -m ingest.generate_logs")
        print("  Run: python -m ingest.feature_build")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
    
    print("-" * 70)
    print("Live ingestion endpoint: POST /ingest")
    print(f"  Retention: {telemetry_store.RETENTION_MINUTES} minutes")
    print("  Stats: GET /ingest/stats")
    print("=" * 70)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend_api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

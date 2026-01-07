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
    pip install fastapi uvicorn pandas
"""

import os
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORED_DATA_PATH = "data/scored_1min.csv"

app = FastAPI(
    title="Microservice Anomaly Detection API",
    description="API for querying microservice metrics and detected anomalies",
    version="1.0.0"
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_scored_data() -> pd.DataFrame:
    """
    Load scored data from CSV file.
    
    Returns
    -------
    pd.DataFrame
        Scored data with anomaly detection results
    
    Raises
    ------
    FileNotFoundError
        If the scored data file doesn't exist
    """
    if not os.path.exists(SCORED_DATA_PATH):
        raise FileNotFoundError(
            f"Scored data file not found at {SCORED_DATA_PATH}. "
            "Please run the pipeline first:\n"
            "  1. python -m ingest.generate_logs\n"
            "  2. python -m ingest.feature_build"
        )
    
    df = pd.read_csv(SCORED_DATA_PATH)
    
    # Parse bucket_ts as datetime for sorting and filtering
    df['bucket_ts'] = pd.to_datetime(df['bucket_ts'])
    
    return df


def get_data() -> pd.DataFrame:
    """
    Get scored data with error handling.
    
    Returns
    -------
    pd.DataFrame
        Scored data
    
    Raises
    ------
    HTTPException
        If data cannot be loaded
    """
    try:
        return load_scored_data()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Data not found",
                "message": str(e),
                "next_steps": [
                    "Run: python -m ingest.generate_logs",
                    "Run: python -m ingest.feature_build"
                ]
            }
        )
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

"""
Log Generation Module

Purpose:
--------
This module simulates microservice log data for testing and development purposes.
It generates realistic log patterns including normal behavior and anomalous events.

Key Responsibilities:
--------------------
- Generate synthetic microservice logs with realistic patterns
- Simulate normal operational behavior (baseline metrics)
- Inject anomalous patterns (spikes, drops, errors)
- Create various log formats (JSON, structured, unstructured)
- Simulate multiple microservices with different characteristics
- Generate temporal patterns (daily cycles, seasonal trends)
- Create correlated anomalies across multiple services

Future Implementation:
---------------------
- Log pattern generators
- Anomaly injection strategies
- Configurable log schemas
- Real-time log streaming simulation
- Integration with message queues (Kafka, RabbitMQ)
"""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
import uuid

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Number of log entries to generate
N_ROWS = 10000

# Start time for log generation (relative to now)
START_TIME = datetime.now() - timedelta(hours=2)

# Incident injection parameters
INCIDENT_START_MINUTE = 45  # Start incident at minute 45 (45 minutes into the 2-hour window)
INCIDENT_DURATION_MINUTES = 8  # Incident lasts for 8 minutes
INCIDENT_SERVICE = "payments"
INCIDENT_ENDPOINT = "/pay"

# Random seed for reproducibility (set to None for random behavior)
RANDOM_SEED = 42

# Output configuration
OUTPUT_DIR = "data"
OUTPUT_FILE = "raw_logs.csv"

# =============================================================================
# MICROSERVICE CONFIGURATION
# =============================================================================

# Define services and their endpoints with baseline characteristics
SERVICES = {
    "auth": {
        "endpoints": [
            ("/login", "POST", 50, 100, 0.01),      # (endpoint, method, min_latency, max_latency, error_rate)
            ("/logout", "POST", 30, 60, 0.005),
            ("/verify", "GET", 20, 50, 0.02),
        ],
        "traffic_weight": 2.0,  # Relative traffic volume
    },
    "catalog": {
        "endpoints": [
            ("/items", "GET", 40, 120, 0.008),
            ("/search", "GET", 80, 200, 0.015),
            ("/item/{id}", "GET", 30, 80, 0.01),
        ],
        "traffic_weight": 3.0,
    },
    "checkout": {
        "endpoints": [
            ("/cart", "GET", 50, 150, 0.01),
            ("/cart/add", "POST", 60, 180, 0.02),
            ("/checkout", "POST", 100, 300, 0.025),
        ],
        "traffic_weight": 1.5,
    },
    "payments": {
        "endpoints": [
            ("/pay", "POST", 150, 400, 0.015),
            ("/refund", "POST", 100, 250, 0.02),
            ("/status", "GET", 40, 100, 0.01),
        ],
        "traffic_weight": 1.0,
    },
}


# =============================================================================
# LOG GENERATION FUNCTIONS
# =============================================================================

def generate_trace_id() -> str:
    """Generate a unique trace ID for request tracing."""
    return str(uuid.uuid4())


def is_in_incident_window(timestamp: datetime, incident_start: datetime, incident_end: datetime) -> bool:
    """Check if timestamp falls within incident window."""
    return incident_start <= timestamp <= incident_end


def generate_log_entry(
    timestamp: datetime,
    service: str,
    endpoint: str,
    method: str,
    baseline_latency_min: int,
    baseline_latency_max: int,
    baseline_error_rate: float,
    is_incident: bool = False
) -> Dict:
    """
    Generate a single log entry with realistic values.
    
    Parameters
    ----------
    timestamp : datetime
        Timestamp for the log entry
    service : str
        Service name
    endpoint : str
        API endpoint
    method : str
        HTTP method
    baseline_latency_min : int
        Minimum latency in ms (normal conditions)
    baseline_latency_max : int
        Maximum latency in ms (normal conditions)
    baseline_error_rate : float
        Error rate (normal conditions)
    is_incident : bool
        Whether this entry is during an incident
    
    Returns
    -------
    Dict
        Log entry dictionary
    """
    # Determine if this request results in an error
    if is_incident:
        # During incident: increase error rate to 10-30%
        error_rate = random.uniform(0.10, 0.30)
        # Spike latency 5x-20x
        latency_multiplier = random.uniform(5, 20)
        latency_ms = int(random.randint(baseline_latency_min, baseline_latency_max) * latency_multiplier)
    else:
        error_rate = baseline_error_rate
        latency_ms = random.randint(baseline_latency_min, baseline_latency_max)
    
    # Determine status code based on error rate
    is_error = random.random() < error_rate
    status_code = 500 if is_error else 200
    
    # If there's an error, increase latency slightly (errors often take longer due to retries)
    if is_error and not is_incident:
        latency_ms = int(latency_ms * random.uniform(1.2, 2.0))
    
    return {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "service": service,
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "trace_id": generate_trace_id(),
    }


def build_weighted_service_list() -> List[Tuple[str, str, str, int, int, float]]:
    """
    Build a weighted list of (service, endpoint, method, min_latency, max_latency, error_rate)
    based on traffic weights.
    
    Returns
    -------
    List[Tuple]
        Weighted list of service configurations
    """
    weighted_list = []
    for service, config in SERVICES.items():
        weight = int(config["traffic_weight"] * 100)  # Convert to integer weight
        for endpoint, method, min_lat, max_lat, err_rate in config["endpoints"]:
            weighted_list.extend([(service, endpoint, method, min_lat, max_lat, err_rate)] * weight)
    return weighted_list


def generate_logs(
    n_rows: int,
    start_time: datetime,
    incident_start_minute: int,
    incident_duration_minutes: int,
    incident_service: str,
    incident_endpoint: str
) -> List[Dict]:
    """
    Generate microservice log dataset with incident injection.
    
    Parameters
    ----------
    n_rows : int
        Number of log entries to generate
    start_time : datetime
        Start timestamp for log generation
    incident_start_minute : int
        Minute offset from start_time when incident begins
    incident_duration_minutes : int
        Duration of incident in minutes
    incident_service : str
        Service affected by incident
    incident_endpoint : str
        Endpoint affected by incident
    
    Returns
    -------
    List[Dict]
        List of log entry dictionaries
    """
    # Calculate incident window
    incident_start = start_time + timedelta(minutes=incident_start_minute)
    incident_end = incident_start + timedelta(minutes=incident_duration_minutes)
    
    # Build weighted service list for traffic distribution
    weighted_services = build_weighted_service_list()
    
    # Calculate time range (2 hours = 7200 seconds for per-second granularity)
    end_time = start_time + timedelta(hours=2)
    total_seconds = int((end_time - start_time).total_seconds())
    
    logs = []
    
    # Generate timestamps with per-second granularity and randomize traffic
    for i in range(n_rows):
        # Random timestamp within the 2-hour window
        random_offset = random.randint(0, total_seconds - 1)
        timestamp = start_time + timedelta(seconds=random_offset)
        
        # Select a service/endpoint combination based on weighted distribution
        service, endpoint, method, min_lat, max_lat, err_rate = random.choice(weighted_services)
        
        # Check if this is during an incident for the affected service/endpoint
        is_incident = (
            service == incident_service and
            endpoint == incident_endpoint and
            is_in_incident_window(timestamp, incident_start, incident_end)
        )
        
        # Generate log entry
        log_entry = generate_log_entry(
            timestamp=timestamp,
            service=service,
            endpoint=endpoint,
            method=method,
            baseline_latency_min=min_lat,
            baseline_latency_max=max_lat,
            baseline_error_rate=err_rate,
            is_incident=is_incident
        )
        
        logs.append(log_entry)
    
    # Sort logs by timestamp
    logs.sort(key=lambda x: x["timestamp"])
    
    return logs, incident_start, incident_end


def write_logs_to_csv(logs: List[Dict], output_path: str) -> None:
    """
    Write logs to CSV file.
    
    Parameters
    ----------
    logs : List[Dict]
        List of log entries
    output_path : str
        Path to output CSV file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write CSV with header
    with open(output_path, "w") as f:
        # Write header
        header = "timestamp,service,endpoint,method,status_code,latency_ms,trace_id\n"
        f.write(header)
        
        # Write data rows
        for log in logs:
            row = f"{log['timestamp']},{log['service']},{log['endpoint']},{log['method']},{log['status_code']},{log['latency_ms']},{log['trace_id']}\n"
            f.write(row)


def print_summary(
    logs: List[Dict],
    incident_start: datetime,
    incident_end: datetime,
    output_path: str
) -> None:
    """
    Print summary statistics to console.
    
    Parameters
    ----------
    logs : List[Dict]
        Generated log entries
    incident_start : datetime
        Incident start time
    incident_end : datetime
        Incident end time
    output_path : str
        Path to output file
    """
    print("=" * 70)
    print("LOG GENERATION SUMMARY")
    print("=" * 70)
    print(f"Total rows generated: {len(logs):,}")
    print(f"Time range: {logs[0]['timestamp']} to {logs[-1]['timestamp']}")
    print()
    print("INCIDENT DETAILS:")
    print(f"  Service: {INCIDENT_SERVICE}")
    print(f"  Endpoint: {INCIDENT_ENDPOINT}")
    print(f"  Start: {incident_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End: {incident_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration: {INCIDENT_DURATION_MINUTES} minutes")
    print()
    
    # Count incident-affected logs
    incident_logs = [
        log for log in logs
        if log['service'] == INCIDENT_SERVICE and
        log['endpoint'] == INCIDENT_ENDPOINT and
        incident_start.strftime('%Y-%m-%d %H:%M:%S') <= log['timestamp'] <= incident_end.strftime('%Y-%m-%d %H:%M:%S')
    ]
    print(f"Incident-affected entries: {len(incident_logs)}")
    
    if incident_logs:
        error_count = sum(1 for log in incident_logs if log['status_code'] == 500)
        avg_latency = sum(log['latency_ms'] for log in incident_logs) / len(incident_logs)
        print(f"  Error rate during incident: {error_count / len(incident_logs):.1%}")
        print(f"  Average latency during incident: {avg_latency:.0f}ms")
    
    print()
    print(f"Output file: {output_path}")
    print("=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point for log generation."""
    # Set random seed if specified
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    
    print("Generating microservice logs...")
    
    # Generate logs
    logs, incident_start, incident_end = generate_logs(
        n_rows=N_ROWS,
        start_time=START_TIME,
        incident_start_minute=INCIDENT_START_MINUTE,
        incident_duration_minutes=INCIDENT_DURATION_MINUTES,
        incident_service=INCIDENT_SERVICE,
        incident_endpoint=INCIDENT_ENDPOINT
    )
    
    # Write to CSV
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    write_logs_to_csv(logs, output_path)
    
    # Print summary
    print_summary(logs, incident_start, incident_end, output_path)


if __name__ == "__main__":
    main()

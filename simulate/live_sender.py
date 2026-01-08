"""
Live Event Sender Script

Simulates realistic microservice telemetry and sends events to the API in real-time.
Used for testing and demonstrating the live anomaly detection pipeline.

Usage:
------
    # Start the API server first:
    uvicorn backend_api.main:app --reload

    # Then run this script:
    python -m simulate.live_sender

    # Or with custom rate:
    python -m simulate.live_sender --tps 10

Features:
---------
- Generates realistic trace-based events (simulating page loads)
- Injects incident after configurable delay
- Configurable event rate (TPS)
- Batch sends for efficiency
"""

import argparse
import random
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any

import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

API_URL = "http://127.0.0.1:8000/ingest"

# Event generation rate
DEFAULT_TPS = 5  # Events per second (actually traces per second, each trace = 5 events)

# Services and their endpoints
SERVICES = ["auth", "catalog", "payments", "checkout"]

ENDPOINTS: Dict[str, List[Dict[str, Any]]] = {
    "auth": [
        {"path": "/login", "method": "POST", "base_latency": (50, 120), "error_rate": 0.01},
        {"path": "/logout", "method": "POST", "base_latency": (30, 60), "error_rate": 0.005},
        {"path": "/verify", "method": "GET", "base_latency": (20, 50), "error_rate": 0.02},
    ],
    "catalog": [
        {"path": "/items", "method": "GET", "base_latency": (40, 120), "error_rate": 0.008},
        {"path": "/search", "method": "GET", "base_latency": (80, 200), "error_rate": 0.015},
        {"path": "/item/{id}", "method": "GET", "base_latency": (30, 80), "error_rate": 0.01},
    ],
    "checkout": [
        {"path": "/cart", "method": "GET", "base_latency": (50, 150), "error_rate": 0.01},
        {"path": "/cart/add", "method": "POST", "base_latency": (60, 180), "error_rate": 0.02},
        {"path": "/checkout", "method": "POST", "base_latency": (100, 300), "error_rate": 0.025},
    ],
    "payments": [
        {"path": "/pay", "method": "POST", "base_latency": (150, 400), "error_rate": 0.015},
        {"path": "/refund", "method": "POST", "base_latency": (100, 250), "error_rate": 0.02},
        {"path": "/status", "method": "GET", "base_latency": (40, 100), "error_rate": 0.01},
    ],
}

# Incident configuration
INCIDENT_SERVICE = "payments"
INCIDENT_ENDPOINT = "/pay"
INCIDENT_AFTER_SECONDS = 30  # Start incident after this many seconds
INCIDENT_DURATION_SECONDS = 60  # Incident lasts this long

# Incident parameters
INCIDENT_LATENCY_MULTIPLIER = (3, 10)  # Latency multiplied by 3x to 10x
INCIDENT_ERROR_RATE = 0.25  # 25% error rate during incident


# =============================================================================
# EVENT GENERATION
# =============================================================================

def generate_latency(base_range: tuple, is_incident: bool = False) -> float:
    """Generate latency value, optionally with incident multiplier."""
    base = random.uniform(base_range[0], base_range[1])
    
    if is_incident:
        multiplier = random.uniform(INCIDENT_LATENCY_MULTIPLIER[0], INCIDENT_LATENCY_MULTIPLIER[1])
        return round(base * multiplier, 2)
    
    return round(base, 2)


def generate_status_code(error_rate: float, is_incident: bool = False) -> int:
    """Generate HTTP status code based on error rate."""
    effective_error_rate = INCIDENT_ERROR_RATE if is_incident else error_rate
    
    if random.random() < effective_error_rate:
        # Return error status codes
        return random.choice([500, 502, 503, 504])
    
    return 200


def generate_trace_events(trace_id: str, timestamp: datetime, in_incident: bool) -> List[Dict[str, Any]]:
    """
    Generate a trace of events simulating a user page load.
    
    A typical page load involves:
    1. Auth verification
    2. Catalog item lookup
    3. Cart status check
    4. (Sometimes) Payment processing
    5. Random additional service call
    """
    events = []
    
    # Define a realistic trace pattern
    trace_pattern = [
        ("auth", "/verify"),
        ("catalog", random.choice(["/items", "/search", "/item/{id}"])),
        ("checkout", "/cart"),
    ]
    
    # Sometimes add payment (30% of traces)
    if random.random() < 0.3:
        trace_pattern.append(("payments", "/pay"))
    
    # Add a random additional call
    random_service = random.choice(SERVICES)
    random_endpoint = random.choice(ENDPOINTS[random_service])["path"]
    trace_pattern.append((random_service, random_endpoint))
    
    # Generate events for each step in the trace
    current_time = timestamp
    
    for service, endpoint_path in trace_pattern:
        # Find endpoint config
        endpoint_config = None
        for ep in ENDPOINTS[service]:
            if ep["path"] == endpoint_path:
                endpoint_config = ep
                break
        
        if endpoint_config is None:
            endpoint_config = ENDPOINTS[service][0]
            endpoint_path = endpoint_config["path"]
        
        # Check if this is the incident endpoint
        is_incident_event = (
            in_incident and 
            service == INCIDENT_SERVICE and 
            endpoint_path == INCIDENT_ENDPOINT
        )
        
        # Generate event
        latency = generate_latency(endpoint_config["base_latency"], is_incident_event)
        status_code = generate_status_code(endpoint_config["error_rate"], is_incident_event)
        
        event = {
            "timestamp": current_time.isoformat(),
            "service": service,
            "endpoint": endpoint_path,
            "status_code": status_code,
            "latency_ms": latency,
            "trace_id": trace_id
        }
        events.append(event)
        
        # Add small delay between events in trace (simulate sequential calls)
        # This is just for timestamp variation, actual latency is in latency_ms
        current_time = current_time  # Keep same second for simplicity
    
    return events


def send_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Send events batch to the API."""
    try:
        response = requests.post(
            API_URL,
            json={"events": events},
            timeout=5
        )
        return {
            "status_code": response.status_code,
            "response": response.json() if response.ok else response.text
        }
    except requests.exceptions.ConnectionError:
        return {"status_code": 0, "response": "Connection refused - is the API running?"}
    except requests.exceptions.Timeout:
        return {"status_code": 0, "response": "Request timeout"}
    except Exception as e:
        return {"status_code": 0, "response": str(e)}


# =============================================================================
# MAIN LOOP
# =============================================================================

def print_status(
    elapsed_seconds: int,
    events_sent: int,
    in_incident: bool,
    response: Dict[str, Any]
) -> None:
    """Print formatted status line."""
    incident_status = "🔴 INCIDENT" if in_incident else "🟢 Normal"
    
    if response["status_code"] == 200:
        resp_info = f"✓ {response['response'].get('accepted', '?')} accepted"
    else:
        resp_info = f"✗ {response['status_code']}: {str(response['response'])[:50]}"
    
    print(f"[{elapsed_seconds:4d}s] {incident_status} | Sent: {events_sent:4d} events | {resp_info}")


def main():
    """Main event sender loop."""
    parser = argparse.ArgumentParser(description="Send simulated microservice events to the API")
    parser.add_argument("--tps", type=int, default=DEFAULT_TPS, help="Traces per second")
    parser.add_argument("--url", type=str, default=API_URL, help="API endpoint URL")
    parser.add_argument("--incident-after", type=int, default=INCIDENT_AFTER_SECONDS, 
                        help="Start incident after N seconds")
    parser.add_argument("--incident-duration", type=int, default=INCIDENT_DURATION_SECONDS,
                        help="Incident duration in seconds")
    args = parser.parse_args()
    
    api_url = args.url
    tps = args.tps
    incident_start = args.incident_after
    incident_end = incident_start + args.incident_duration
    
    print("=" * 70)
    print("  Microservice Live Event Sender")
    print("=" * 70)
    print(f"  API URL:          {api_url}")
    print(f"  Traces/second:    {tps} (≈{tps * 5} events/sec)")
    print(f"  Incident window:  {incident_start}s - {incident_end}s")
    print(f"  Incident target:  {INCIDENT_SERVICE}:{INCIDENT_ENDPOINT}")
    print("=" * 70)
    print()
    print("Press Ctrl+C to stop\n")
    
 
    start_time = time.time()
    total_events_sent = 0
    
    try:
        while True:
            loop_start = time.time()
            elapsed = int(loop_start - start_time)
            
            # Determine if we're in incident window
            in_incident = incident_start <= elapsed < incident_end
            
            # Generate batch of traces for this second
            events_batch = []
            for _ in range(tps):
                trace_id = str(uuid.uuid4())
                timestamp = datetime.now()
                trace_events = generate_trace_events(trace_id, timestamp, in_incident)
                events_batch.extend(trace_events)
            
            # Send batch
            response = send_events(events_batch)
            total_events_sent += len(events_batch)
            
            # Print status
            print_status(elapsed, len(events_batch), in_incident, response)
            
            # Sleep to maintain rate (account for processing time)
            elapsed_in_loop = time.time() - loop_start
            sleep_time = max(0, 1.0 - elapsed_in_loop)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n")
        print("=" * 70)
        print(f"  Stopped after {int(time.time() - start_time)} seconds")
        print(f"  Total events sent: {total_events_sent:,}")
        print("=" * 70)


if __name__ == "__main__":
    main()

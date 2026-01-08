"""
Telemetry Store Module

Thread-safe in-memory storage for live telemetry events.
Designed for real-time log ingestion without breaking existing batch mode.

Usage:
------
    from backend_api.telemetry_store import add_events, get_events_since, get_all_events, prune_older_than

    # Add events
    events = [{"timestamp": "2026-01-07T10:00:00", "service": "auth", ...}]
    count = add_events(events)

    # Query events
    recent = get_events_since("2026-01-07T09:45:00")
    all_events = get_all_events()

    # Prune old events
    removed = prune_older_than(180)
"""

import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

RETENTION_MINUTES = 180  # Default retention: 3 hours
MAX_EVENTS = 100000  # Maximum events to store (safety limit)


# =============================================================================
# THREAD-SAFE IN-MEMORY STORE
# =============================================================================

# Internal storage
_store: deque = deque(maxlen=MAX_EVENTS)
_lock = threading.Lock()


# =============================================================================
# PUBLIC API
# =============================================================================

def add_events(events: List[Dict[str, Any]]) -> int:
    """
    Add log events to the in-memory store.

    Parameters
    ----------
    events : List[Dict[str, Any]]
        List of event dictionaries. Each event should have keys:
        - timestamp: ISO format string (e.g., "2026-01-07T10:00:00")
        - service: Service name (str)
        - endpoint: Endpoint path (str)
        - status_code: HTTP status code (int)
        - latency_ms: Response latency in milliseconds (float/int)
        - trace_id: Optional trace ID for distributed tracing (str|None)

    Returns
    -------
    int
        Number of events successfully added.

    Example
    -------
    >>> events = [
    ...     {"timestamp": "2026-01-07T10:00:00", "service": "auth", 
    ...      "endpoint": "/login", "status_code": 200, "latency_ms": 45.2, "trace_id": None}
    ... ]
    >>> add_events(events)
    1
    """
    if not events:
        return 0

    added_count = 0
    with _lock:
        for event in events:
            # Validate required keys exist
            required_keys = {"timestamp", "service", "endpoint", "status_code", "latency_ms"}
            if not required_keys.issubset(event.keys()):
                continue  # Skip invalid events

            # Normalize event structure
            normalized = {
                "timestamp": str(event["timestamp"]),
                "service": str(event["service"]),
                "endpoint": str(event["endpoint"]),
                "status_code": int(event["status_code"]),
                "latency_ms": float(event["latency_ms"]),
                "trace_id": event.get("trace_id")
            }
            _store.append(normalized)
            added_count += 1

    return added_count


def get_events_since(iso_ts: str) -> List[Dict[str, Any]]:
    """
    Get all events since a given timestamp.

    Parameters
    ----------
    iso_ts : str
        ISO format timestamp string (e.g., "2026-01-07T09:45:00")

    Returns
    -------
    List[Dict[str, Any]]
        List of events with timestamp >= iso_ts, sorted by timestamp ascending.

    Example
    -------
    >>> events = get_events_since("2026-01-07T09:45:00")
    >>> len(events)
    42
    """
    try:
        cutoff = datetime.fromisoformat(iso_ts.replace("Z", "+00:00").replace("+00:00", ""))
    except ValueError:
        # Try parsing without timezone
        cutoff = datetime.fromisoformat(iso_ts.split("+")[0].replace("Z", ""))

    result = []
    with _lock:
        for event in _store:
            try:
                event_ts_str = event["timestamp"].replace("Z", "+00:00").replace("+00:00", "")
                event_ts = datetime.fromisoformat(event_ts_str.split("+")[0])
                if event_ts >= cutoff:
                    result.append(event.copy())
            except (ValueError, KeyError):
                continue

    # Sort by timestamp
    result.sort(key=lambda x: x["timestamp"])
    return result


def get_all_events() -> List[Dict[str, Any]]:
    """
    Get all events currently in the store.

    Returns
    -------
    List[Dict[str, Any]]
        List of all stored events, sorted by timestamp ascending.

    Example
    -------
    >>> all_events = get_all_events()
    >>> len(all_events)
    1500
    """
    with _lock:
        result = [event.copy() for event in _store]

    # Sort by timestamp
    result.sort(key=lambda x: x["timestamp"])
    return result


def prune_older_than(minutes: int) -> int:
    """
    Remove events older than the specified number of minutes.

    Parameters
    ----------
    minutes : int
        Remove events older than this many minutes from now.

    Returns
    -------
    int
        Number of events removed.

    Example
    -------
    >>> removed = prune_older_than(180)  # Remove events older than 3 hours
    >>> print(f"Pruned {removed} old events")
    Pruned 50 old events
    """
    cutoff = datetime.now() - timedelta(minutes=minutes)
    removed_count = 0

    with _lock:
        # Since deque doesn't support efficient removal from middle,
        # we rebuild the deque with only valid events
        events_to_keep = []
        for event in _store:
            try:
                event_ts_str = event["timestamp"].replace("Z", "+00:00").replace("+00:00", "")
                event_ts = datetime.fromisoformat(event_ts_str.split("+")[0])
                if event_ts >= cutoff:
                    events_to_keep.append(event)
                else:
                    removed_count += 1
            except (ValueError, KeyError):
                removed_count += 1  # Remove malformed events

        _store.clear()
        _store.extend(events_to_keep)

    return removed_count


def clear_store() -> int:
    """
    Clear all events from the store.

    Returns
    -------
    int
        Number of events removed.
    """
    with _lock:
        count = len(_store)
        _store.clear()
    return count


def get_store_stats() -> Dict[str, Any]:
    """
    Get statistics about the current store state.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - count: Number of events in store
        - max_capacity: Maximum events allowed
        - oldest_ts: Oldest event timestamp (or None)
        - newest_ts: Newest event timestamp (or None)
    """
    with _lock:
        count = len(_store)
        if count == 0:
            return {
                "count": 0,
                "max_capacity": MAX_EVENTS,
                "oldest_ts": None,
                "newest_ts": None
            }

        # Find oldest and newest
        timestamps = []
        for event in _store:
            try:
                timestamps.append(event["timestamp"])
            except KeyError:
                continue

        if timestamps:
            timestamps.sort()
            oldest = timestamps[0]
            newest = timestamps[-1]
        else:
            oldest = None
            newest = None

        return {
            "count": count,
            "max_capacity": MAX_EVENTS,
            "oldest_ts": oldest,
            "newest_ts": newest
        }

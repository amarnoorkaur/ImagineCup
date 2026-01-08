"""
Backend API Package

Provides FastAPI endpoints for microservice anomaly detection.
"""

from backend_api import telemetry_store
from backend_api import realtime_processor

__all__ = ["telemetry_store", "realtime_processor"]

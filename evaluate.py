"""
Evaluation Module

Purpose:
--------
Compute evaluation metrics for the anomaly detection system using the
injected incident metadata as ground truth.

This script compares detected anomalies against the known incident window
to calculate precision-like metrics and detection delay.

Run Instructions:
-----------------
From the repository root, run:
    python evaluate.py

Prerequisites:
--------------
    1. python -m ingest.generate_logs  (creates incident_meta.json)
    2. python -m ingest.feature_build  (creates scored_1min.csv)

Dependencies:
-------------
    pandas, json (standard library)
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

INCIDENT_META_PATH = "data/incident_meta.json"
SCORED_DATA_PATH = "data/scored_1min.csv"
EVAL_REPORT_PATH = "data/eval_report.json"


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def load_incident_meta(path: str) -> Dict[str, Any]:
    """
    Load incident metadata (ground truth) from JSON file.
    
    Parameters
    ----------
    path : str
        Path to incident_meta.json
    
    Returns
    -------
    Dict[str, Any]
        Incident metadata including service, endpoint, and time window
    
    Raises
    ------
    FileNotFoundError
        If the file doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Incident metadata not found at {path}. "
            "Please run: python -m ingest.generate_logs"
        )
    
    with open(path, 'r') as f:
        return json.load(f)


def load_scored_data(path: str) -> pd.DataFrame:
    """
    Load scored anomaly detection results.
    
    Parameters
    ----------
    path : str
        Path to scored_1min.csv
    
    Returns
    -------
    pd.DataFrame
        Scored data with anomaly flags
    
    Raises
    ------
    FileNotFoundError
        If the file doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Scored data not found at {path}. "
            "Please run: python -m ingest.feature_build"
        )
    
    df = pd.read_csv(path)
    df['bucket_ts'] = pd.to_datetime(df['bucket_ts'])
    return df


def compute_evaluation_metrics(
    incident_meta: Dict[str, Any],
    scored_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compute evaluation metrics comparing detected anomalies to ground truth.
    
    Parameters
    ----------
    incident_meta : Dict[str, Any]
        Ground truth incident metadata
    scored_df : pd.DataFrame
        Scored anomaly detection results
    
    Returns
    -------
    Dict[str, Any]
        Evaluation metrics including TP, FP, detection delay, and precision
    """
    # Extract incident parameters
    incident_service = incident_meta['incident_service']
    incident_endpoint = incident_meta['incident_endpoint']
    incident_start = pd.to_datetime(incident_meta['incident_start_ts'])
    incident_end = pd.to_datetime(incident_meta['incident_end_ts'])
    
    # Filter to anomalies only
    anomalies = scored_df[scored_df['is_anomaly'] == True].copy()
    
    # True Positives: Anomalies inside incident window for incident service+endpoint
    tp_mask = (
        (anomalies['service'] == incident_service) &
        (anomalies['endpoint'] == incident_endpoint) &
        (anomalies['bucket_ts'] >= incident_start) &
        (anomalies['bucket_ts'] <= incident_end)
    )
    tp_anomalies = anomalies[tp_mask]
    tp = len(tp_anomalies)
    
    # False Positives: Anomalies outside incident window (all services/endpoints)
    # This includes:
    #   - Anomalies for other service+endpoint combinations
    #   - Anomalies for incident service+endpoint but outside time window
    fp_mask = ~tp_mask
    fp_anomalies = anomalies[fp_mask]
    fp = len(fp_anomalies)
    
    # Detection Delay: Time from incident start to first detected anomaly
    detection_delay_minutes: Optional[float] = None
    first_detection_ts: Optional[str] = None
    
    if tp > 0:
        first_tp_ts = tp_anomalies['bucket_ts'].min()
        detection_delay_minutes = (first_tp_ts - incident_start).total_seconds() / 60
        first_detection_ts = first_tp_ts.isoformat()
    
    # Precision Proxy: TP / (TP + FP)
    precision_proxy: Optional[float] = None
    if (tp + fp) > 0:
        precision_proxy = tp / (tp + fp)
    
    # Additional context metrics
    total_anomalies = len(anomalies)
    total_rows = len(scored_df)
    incident_duration_minutes = (incident_end - incident_start).total_seconds() / 60
    
    # Count expected incident rows (rows inside incident window for incident service+endpoint)
    expected_mask = (
        (scored_df['service'] == incident_service) &
        (scored_df['endpoint'] == incident_endpoint) &
        (scored_df['bucket_ts'] >= incident_start) &
        (scored_df['bucket_ts'] <= incident_end)
    )
    expected_incident_rows = len(scored_df[expected_mask])
    
    # Recall proxy: TP / expected incident rows
    recall_proxy: Optional[float] = None
    if expected_incident_rows > 0:
        recall_proxy = tp / expected_incident_rows
    
    return {
        "ground_truth": {
            "incident_service": incident_service,
            "incident_endpoint": incident_endpoint,
            "incident_start_ts": incident_start.isoformat(),
            "incident_end_ts": incident_end.isoformat(),
            "incident_duration_minutes": incident_duration_minutes,
            "expected_incident_rows": expected_incident_rows
        },
        "metrics": {
            "TP": tp,
            "FP": fp,
            "total_anomalies": total_anomalies,
            "detection_delay_minutes": detection_delay_minutes,
            "first_detection_ts": first_detection_ts,
            "precision_proxy": round(precision_proxy, 4) if precision_proxy is not None else None,
            "recall_proxy": round(recall_proxy, 4) if recall_proxy is not None else None
        },
        "context": {
            "total_rows": total_rows,
            "anomaly_rate": round(total_anomalies / total_rows, 4) if total_rows > 0 else 0,
            "seed": incident_meta.get('seed'),
            "generated_rows": incident_meta.get('rows')
        }
    }


def print_report(eval_result: Dict[str, Any]) -> None:
    """
    Print evaluation report to console.
    
    Parameters
    ----------
    eval_result : Dict[str, Any]
        Evaluation metrics
    """
    gt = eval_result['ground_truth']
    metrics = eval_result['metrics']
    ctx = eval_result['context']
    
    print("=" * 70)
    print("ANOMALY DETECTION EVALUATION REPORT")
    print("=" * 70)
    
    print("\n📋 GROUND TRUTH (Injected Incident):")
    print(f"   Service:    {gt['incident_service']}")
    print(f"   Endpoint:   {gt['incident_endpoint']}")
    print(f"   Start:      {gt['incident_start_ts']}")
    print(f"   End:        {gt['incident_end_ts']}")
    print(f"   Duration:   {gt['incident_duration_minutes']:.0f} minutes")
    print(f"   Expected Rows: {gt['expected_incident_rows']}")
    
    print("\n📊 EVALUATION METRICS:")
    print(f"   True Positives (TP):   {metrics['TP']}")
    print(f"   False Positives (FP):  {metrics['FP']}")
    print(f"   Total Anomalies:       {metrics['total_anomalies']}")
    
    if metrics['detection_delay_minutes'] is not None:
        print(f"\n   Detection Delay:       {metrics['detection_delay_minutes']:.1f} minutes")
        print(f"   First Detection:       {metrics['first_detection_ts']}")
    else:
        print(f"\n   Detection Delay:       N/A (no true positives)")
    
    print(f"\n   Precision Proxy:       ", end="")
    if metrics['precision_proxy'] is not None:
        print(f"{metrics['precision_proxy']:.2%} (TP / (TP + FP))")
    else:
        print("N/A")
    
    print(f"   Recall Proxy:          ", end="")
    if metrics['recall_proxy'] is not None:
        print(f"{metrics['recall_proxy']:.2%} (TP / Expected)")
    else:
        print("N/A")
    
    print("\n📈 CONTEXT:")
    print(f"   Total Scored Rows:     {ctx['total_rows']:,}")
    print(f"   Overall Anomaly Rate:  {ctx['anomaly_rate']:.2%}")
    print(f"   Random Seed Used:      {ctx['seed']}")
    
    # Interpretation
    print("\n💡 INTERPRETATION:")
    if metrics['TP'] > 0 and metrics['precision_proxy'] is not None:
        if metrics['precision_proxy'] >= 0.8:
            print("   ✅ High precision - most detected anomalies are true incidents")
        elif metrics['precision_proxy'] >= 0.5:
            print("   ⚠️ Moderate precision - some false positives present")
        else:
            print("   ❌ Low precision - many false positives")
        
        if metrics['detection_delay_minutes'] is not None:
            if metrics['detection_delay_minutes'] <= 1:
                print("   ✅ Fast detection - incident caught within 1 minute")
            elif metrics['detection_delay_minutes'] <= 5:
                print("   ⚠️ Moderate delay - detected within 5 minutes")
            else:
                print("   ❌ Slow detection - took more than 5 minutes")
    else:
        print("   ❌ No true positives detected - incident was missed")
    
    print("=" * 70)


def save_report(eval_result: Dict[str, Any], path: str) -> None:
    """
    Save evaluation report to JSON file.
    
    Parameters
    ----------
    eval_result : Dict[str, Any]
        Evaluation metrics
    path : str
        Output path for JSON file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(eval_result, f, indent=2)
    
    print(f"\n📁 Report saved to: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for evaluation."""
    print("Loading data...")
    
    try:
        # Load ground truth and scored data
        incident_meta = load_incident_meta(INCIDENT_META_PATH)
        scored_df = load_scored_data(SCORED_DATA_PATH)
        
        print(f"  ✓ Loaded incident metadata from {INCIDENT_META_PATH}")
        print(f"  ✓ Loaded {len(scored_df):,} scored rows from {SCORED_DATA_PATH}")
        
        # Compute evaluation metrics
        eval_result = compute_evaluation_metrics(incident_meta, scored_df)
        
        # Print report
        print_report(eval_result)
        
        # Save report
        save_report(eval_result, EVAL_REPORT_PATH)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run the pipeline first:")
        print("  1. python -m ingest.generate_logs")
        print("  2. python -m ingest.feature_build")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise
    
    return 0


if __name__ == "__main__":
    exit(main())

"""
Feature Build Module

Purpose:
--------
This module is responsible for building and engineering features from raw microservice logs
for anomaly detection. It processes ingested logs and transforms them into meaningful
features that can be used by machine learning models.

Key Responsibilities:
--------------------
- Extract relevant metrics from microservice logs (latency, error rates, request counts)
- Perform time-series feature engineering (rolling windows, lag features)
- Aggregate metrics across different time intervals
- Normalize and scale features for model input
- Handle missing values and data quality issues
- Generate statistical features (mean, std, percentiles)
- Create derived features (rate of change, anomaly scores)

Future Implementation:
---------------------
- Feature extraction pipeline
- Time-series transformations
- Feature store integration
- Real-time feature computation
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

from ingest.detector_mad import add_mad_anomalies


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = "data/raw_logs.csv"
FEATURES_OUTPUT = "data/features_1min.csv"
SCORED_OUTPUT = "data/scored_1min.csv"


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def load_raw_logs(file_path: str) -> pd.DataFrame:
    """
    Load raw logs from CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to the raw logs CSV file
    
    Returns
    -------
    pd.DataFrame
        DataFrame with parsed timestamps
    """
    print(f"Loading raw logs from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Parse timestamp column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"  Loaded {len(df):,} raw log entries")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def build_per_minute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-minute aggregated features from raw logs.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw logs DataFrame with columns: timestamp, service, endpoint, 
        status_code, latency_ms
    
    Returns
    -------
    pd.DataFrame
        Aggregated features with columns: bucket_ts, service, endpoint,
        req_count, error_count, error_rate, avg_latency_ms, p95_latency_ms
    """
    print("\nBuilding per-minute features...")
    
    # Create 1-minute buckets
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
        p95_latency_ms=('latency_ms', lambda x: np.percentile(x, 95))
    ).reset_index()
    
    # Calculate error rate
    features['error_rate'] = features['error_count'] / features['req_count']
    
    # Sort by bucket_ts ascending for consistent ordering
    features = features.sort_values('bucket_ts').reset_index(drop=True)
    
    # Convert bucket_ts to ISO string for CSV compatibility
    features['bucket_ts'] = features['bucket_ts'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    print(f"  Generated {len(features):,} feature rows")
    print(f"  Unique services: {features['service'].nunique()}")
    print(f"  Unique endpoints: {features['endpoint'].nunique()}")
    
    return features


def save_features(df: pd.DataFrame, output_path: str) -> None:
    """
    Save features to CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Features DataFrame
    output_path : str
        Path to output CSV file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nSaved features to {output_path}")


def apply_mad_anomaly_detection(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply MAD-based anomaly detection to features.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Features DataFrame with per-minute metrics
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added anomaly detection columns: mad_z, is_anomaly, severity
    """
    print("\nApplying MAD anomaly detection...")
    
    # Convert bucket_ts back to datetime for the anomaly detection function
    features_df['bucket_ts'] = pd.to_datetime(features_df['bucket_ts'])
    
    # Apply MAD anomaly detection
    scored_df = add_mad_anomalies(
        df=features_df,
        group_cols=("service", "endpoint"),
        ts_col="bucket_ts",
        metric_col="p95_latency_ms",
        threshold=3.5
    )
    
    # Sort by bucket_ts ascending for consistent ordering
    scored_df = scored_df.sort_values('bucket_ts').reset_index(drop=True)
    
    # Convert bucket_ts back to ISO string for output
    scored_df['bucket_ts'] = scored_df['bucket_ts'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Count anomalies
    anomaly_count = scored_df['is_anomaly'].sum()
    print(f"  Detected {anomaly_count:,} anomalies out of {len(scored_df):,} rows ({anomaly_count/len(scored_df)*100:.1f}%)")
    
    return scored_df


def save_scored_output(df: pd.DataFrame, output_path: str) -> None:
    """
    Save scored output with selected columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Scored DataFrame with all columns
    output_path : str
        Path to output CSV file
    
    Notes
    -----
    Output columns (in order):
        bucket_ts, service, endpoint, req_count, error_count, error_rate,
        avg_latency_ms, p95_latency_ms, mad_z, is_anomaly, severity
    
    Sorting: bucket_ts ascending (consistent with API/dashboard)
    """
    # Select and order columns for output
    output_columns = [
        'bucket_ts', 'service', 'endpoint', 
        'req_count', 'error_count', 'error_rate', 'avg_latency_ms', 'p95_latency_ms',
        'mad_z', 'is_anomaly', 'severity'
    ]
    
    output_df = df[output_columns]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved scored output to {output_path}")


def print_summary(features_df: pd.DataFrame, scored_df: pd.DataFrame) -> None:
    """
    Print summary statistics of the feature engineering and anomaly detection.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Features DataFrame
    scored_df : pd.DataFrame
        Scored DataFrame with anomaly detection results
    """
    print("\n" + "=" * 70)
    print("FEATURE BUILD AND ANOMALY DETECTION SUMMARY")
    print("=" * 70)
    
    # Total rows
    print(f"\nTotal Rows: {len(scored_df):,}")
    
    # Time range
    print(f"Time Range: {scored_df['bucket_ts'].min()} to {scored_df['bucket_ts'].max()}")
    
    # Anomalies count
    total_anomalies = scored_df['is_anomaly'].sum()
    anomaly_pct = total_anomalies / len(scored_df) * 100 if len(scored_df) > 0 else 0
    print(f"\nAnomalies Count: {total_anomalies:,} / {len(scored_df):,} ({anomaly_pct:.2f}%)")
    
    # Anomalies by severity
    print("\nAnomalies by Severity:")
    severity_order = ['Info', 'Warning', 'Critical']
    severity_counts = scored_df['severity'].value_counts()
    for severity in severity_order:
        count = severity_counts.get(severity, 0)
        percentage = count / len(scored_df) * 100 if len(scored_df) > 0 else 0
        marker = "" if severity == "Info" else " [ANOMALY]"
        print(f"  {severity:10s}: {count:6,} ({percentage:5.2f}%){marker}")
    
    if total_anomalies > 0:
        # Show top affected service/endpoint combinations
        anomalies_df = scored_df[scored_df['is_anomaly']]
        top_affected = anomalies_df.groupby(['service', 'endpoint']).size().sort_values(ascending=False).head(5)
        
        print("\nTop Affected Service/Endpoint Combinations:")
        for (service, endpoint), count in top_affected.items():
            print(f"  {service:15s} {endpoint:20s}: {count:3,} anomalies")
    
    # Output files
    print(f"\nOutput Files:")
    print(f"  Features: {FEATURES_OUTPUT}")
    print(f"  Scored:   {SCORED_OUTPUT}")
    
    print("=" * 70)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    input_file: str = INPUT_FILE,
    features_output: str = FEATURES_OUTPUT,
    scored_output: str = SCORED_OUTPUT
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete feature engineering and anomaly detection pipeline.
    
    Parameters
    ----------
    input_file : str
        Path to raw logs CSV
    features_output : str
        Path to save features CSV
    scored_output : str
        Path to save scored output CSV
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (features_df, scored_df)
    """
    # Step 1: Load raw logs
    raw_logs = load_raw_logs(input_file)
    
    # Step 2: Build per-minute features
    features_df = build_per_minute_features(raw_logs)
    
    # Step 3: Save features
    save_features(features_df, features_output)
    
    # Step 4: Apply MAD anomaly detection
    scored_df = apply_mad_anomaly_detection(features_df.copy())
    
    # Step 5: Save scored output
    save_scored_output(scored_df, scored_output)
    
    # Step 6: Print summary
    print_summary(features_df, scored_df)
    
    return features_df, scored_df


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for feature building and anomaly detection."""
    print("Starting Feature Build Pipeline...")
    print("=" * 70)
    
    try:
        run_pipeline()
        print("\nPipeline completed successfully!")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run 'python -m ingest.generate_logs' first to generate raw logs.")
    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        raise


if __name__ == "__main__":
    main()

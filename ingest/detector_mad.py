"""
MAD-based anomaly detection module.

This module provides robust anomaly scoring using Median Absolute Deviation (MAD)
and modified z-scores for time-series metrics.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def mad_scores(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Compute modified z-scores using Median Absolute Deviation (MAD).
    
    The modified z-score is calculated as:
        modified_z = 0.6745 * (x - median) / MAD
    
    where MAD is the median absolute deviation from the median.
    The constant 0.6745 is used to make the MAD comparable to standard deviation
    for normally distributed data.
    
    Parameters
    ----------
    values : np.ndarray
        Array of numeric values to compute scores for.
    eps : float, optional
        Small epsilon value to prevent division by zero (default: 1e-9).
    
    Returns
    -------
    np.ndarray
        Array of modified z-scores with the same shape as input.
    
    Examples
    --------
    >>> values = np.array([1, 2, 3, 4, 100])
    >>> scores = mad_scores(values)
    >>> scores[-1] > 3  # The outlier has high score
    True
    """
    if len(values) == 0:
        return np.array([])
    
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    
    # Prevent division by zero
    if mad < eps:
        mad = eps
    
    # Calculate modified z-scores
    modified_z = 0.6745 * (values - median) / mad
    
    return modified_z


def add_mad_anomalies(
    df: pd.DataFrame,
    group_cols: Tuple[str, ...] = ("service", "endpoint"),
    ts_col: str = "bucket_ts",
    metric_col: str = "p95_latency_ms",
    threshold: float = 3.5,
    min_points: int = 20,
    warmup_points: int = 15,
    min_req_count: int = 20,
    req_col: str = "req_count"
) -> pd.DataFrame:
    """
    Add MAD-based anomaly detection columns to a DataFrame.
    
    This function groups data by specified columns, sorts by timestamp,
    and computes MAD scores for each group. It then flags anomalies
    and assigns severity levels.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing time-series metrics.
    group_cols : Tuple[str, ...], optional
        Column names to group by (default: ("service", "endpoint")).
    ts_col : str, optional
        Timestamp column name for sorting (default: "bucket_ts").
    metric_col : str, optional
        Metric column to analyze (default: "p95_latency_ms").
    threshold : float, optional
        MAD z-score threshold for anomaly detection (default: 3.5).
    min_points : int, optional
        Minimum number of data points required in a group to flag any anomalies.
        Groups with fewer points will have all rows set to is_anomaly=False.
        This prevents false positives from small sample sizes (default: 20).
    warmup_points : int, optional
        Number of initial rows per group to skip for anomaly detection.
        These "warmup" rows allow the MAD baseline to stabilize before
        flagging anomalies. mad_z is still computed but is_anomaly=False
        for these rows (default: 15).
    min_req_count : int, optional
        Minimum request count required to flag a row as anomalous.
        Rows with low traffic are unreliable for anomaly detection due to
        high variance from small sample sizes (default: 20).
    req_col : str, optional
        Column name containing request counts (default: "req_count").
        If this column doesn't exist, the min_req_count guardrail is skipped.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns:
        - mad_z: Modified z-score for each row
        - is_anomaly: Boolean flag indicating anomaly
        - severity: Severity level (Info, Warning, or Critical)
    
    Notes
    -----
    Severity rules:
    - Info: abs(mad_z) <= threshold
    - Warning: abs(mad_z) > threshold
    - Critical: abs(mad_z) > threshold AND error_rate >= 0.05 (if error_rate exists)
    
    Production Guardrails (to reduce false positives):
    1. **min_points**: Groups with < min_points rows never flag anomalies.
       Rationale: MAD requires sufficient data for a reliable baseline.
    2. **warmup_points**: First warmup_points rows per group are never flagged.
       Rationale: Initial data points lack historical context for comparison.
    3. **min_req_count**: Rows with req_col < min_req_count are never flagged.
       Rationale: Low-traffic periods have high metric variance (noise).
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'service': ['api', 'api', 'api'],
    ...     'endpoint': ['/users', '/users', '/users'],
    ...     'bucket_ts': pd.date_range('2024-01-01', periods=3, freq='1h'),
    ...     'p95_latency_ms': [100, 105, 500]
    ... })
    >>> result = add_mad_anomalies(df)
    >>> result['is_anomaly'].iloc[-1]
    True
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Sort by group columns and timestamp
    sort_cols = list(group_cols) + [ts_col]
    result = result.sort_values(by=sort_cols).reset_index(drop=True)
    
    # Initialize columns
    result['mad_z'] = 0.0
    result['is_anomaly'] = False
    result['severity'] = 'Info'
    
    # Check if error_rate column exists
    has_error_rate = 'error_rate' in result.columns
    
    # Check if req_col exists for min_req_count guardrail
    has_req_col = req_col in result.columns
    
    # Compute MAD scores for each group
    for group_values, group_df in result.groupby(list(group_cols)):
        indices = group_df.index
        values = group_df[metric_col].values
        group_size = len(indices)
        
        # Compute MAD scores for this group
        scores = mad_scores(values)
        result.loc[indices, 'mad_z'] = scores
        
        # Guardrail 1: Skip anomaly flagging if group has too few points
        if group_size < min_points:
            # Leave is_anomaly=False, severity='Info' for entire group
            continue
        
        # Flag anomalies based on threshold
        is_anomalous = np.abs(scores) > threshold
        
        # Convert indices to list for positional access
        indices_list = indices.tolist()
        
        # Assign anomaly flags and severity levels
        for i, (idx, is_anom, score) in enumerate(zip(indices_list, is_anomalous, scores)):
            # Guardrail 2: Skip warmup period (first warmup_points rows)
            if i < warmup_points:
                result.loc[idx, 'is_anomaly'] = False
                result.loc[idx, 'severity'] = 'Info'
                continue
            
            # Guardrail 3: Skip rows with low request count
            if has_req_col and result.loc[idx, req_col] < min_req_count:
                result.loc[idx, 'is_anomaly'] = False
                result.loc[idx, 'severity'] = 'Info'
                continue
            
            # Apply standard anomaly detection logic
            if not is_anom:
                result.loc[idx, 'is_anomaly'] = False
                result.loc[idx, 'severity'] = 'Info'
            else:
                result.loc[idx, 'is_anomaly'] = True
                # Start with Warning
                severity = 'Warning'
                
                # Upgrade to Critical if error_rate is high
                if has_error_rate:
                    error_rate = result.loc[idx, 'error_rate']
                    if error_rate >= 0.05:
                        severity = 'Critical'
                
                result.loc[idx, 'severity'] = severity
    
    return result


def explain_anomaly(row: dict) -> str:
    """
    Generate a human-readable explanation for an anomaly.
    
    Parameters
    ----------
    row : dict
        Dictionary containing anomaly information with keys like:
        service, endpoint, severity, mad_z, and metric values.
    
    Returns
    -------
    str
        Human-readable summary of the anomaly.
    
    Examples
    --------
    >>> row = {
    ...     'service': 'payment-api',
    ...     'endpoint': '/checkout',
    ...     'severity': 'Critical',
    ...     'mad_z': 5.2,
    ...     'p95_latency_ms': 850,
    ...     'error_rate': 0.08
    ... }
    >>> explanation = explain_anomaly(row)
    >>> 'payment-api' in explanation
    True
    """
    service = row.get('service', 'unknown')
    endpoint = row.get('endpoint', 'unknown')
    severity = row.get('severity', 'Unknown')
    mad_z = row.get('mad_z', 0.0)
    
    # Build base message
    direction = "elevated" if mad_z > 0 else "reduced"
    message = f"{severity}: {service} {endpoint} shows {direction} metrics (MAD z-score: {mad_z:.2f})"
    
    # Add metric details if available
    metric_details = []
    
    if 'p95_latency_ms' in row:
        metric_details.append(f"p95 latency: {row['p95_latency_ms']:.1f}ms")
    
    if 'error_rate' in row:
        metric_details.append(f"error rate: {row['error_rate']:.2%}")
    
    if 'throughput' in row:
        metric_details.append(f"throughput: {row['throughput']:.1f} req/s")
    
    if metric_details:
        message += " [" + ", ".join(metric_details) + "]"
    
    return message


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    dates = pd.date_range('2024-01-01', periods=50, freq='1h')
    
    data = {
        'service': ['api'] * 50,
        'endpoint': ['/users'] * 50,
        'bucket_ts': dates,
        'p95_latency_ms': np.concatenate([
            np.random.normal(100, 10, 45),  # Normal data
            np.random.normal(300, 20, 5)    # Anomalies
        ]),
        'error_rate': np.concatenate([
            np.random.uniform(0.001, 0.01, 45),  # Low error rate
            np.random.uniform(0.05, 0.10, 5)     # High error rate
        ])
    }
    
    df = pd.DataFrame(data)
    
    # Add anomaly detection
    result = add_mad_anomalies(df)
    
    # Show anomalies
    anomalies = result[result['is_anomaly']]
    print(f"Found {len(anomalies)} anomalies:\n")
    
    for _, row in anomalies.iterrows():
        print(explain_anomaly(row.to_dict()))

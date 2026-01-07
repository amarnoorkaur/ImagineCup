"""
Microservice Anomaly Dashboard

A Streamlit dashboard for visualizing microservice metrics and detected anomalies
using MAD (Median Absolute Deviation) based detection without requiring training.

Run Instructions:
-----------------
From the repository root, run:
    streamlit run dashboard/app.py

Dependencies:
-------------
    pip install streamlit pandas
"""

import os
from typing import Optional, List, Dict
from datetime import datetime

import pandas as pd
import streamlit as st


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORED_DATA_PATH = "data/scored_1min.csv"


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    """
    Load scored data from CSV file with caching.
    
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
    
    # Load data
    df = load_data()
    
    # Check if data exists
    if df is None:
        st.error("❌ Data file not found!")
        st.markdown("""
        ### Steps to generate data:
        
        1. Generate raw logs:
        ```bash
        python -m ingest.generate_logs
        ```
        
        2. Build features and run anomaly detection:
        ```bash
        python -m ingest.feature_build
        ```
        
        Then refresh this page.
        """)
        return
    
    # Display data info
    st.success(f"✓ Loaded {len(df):,} records from {df['bucket_ts'].min()} to {df['bucket_ts'].max()}")
    
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
        
        # Format timestamps
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

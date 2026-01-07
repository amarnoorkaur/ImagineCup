# Microservice Anomaly Detection (MAD)

Real-time anomaly detection for microservices using Median Absolute Deviation (MAD) - no training required.

## Overview

This project demonstrates training-free anomaly detection for microservice APIs using statistical methods. It generates synthetic logs, detects anomalies using MAD (Median Absolute Deviation), and provides both a REST API and interactive dashboard for analysis.

## Features

- 🔍 **Training-free anomaly detection** using MAD statistical method
- 📊 **Synthetic log generation** with realistic traffic patterns and incident injection
- 🚀 **FastAPI backend** for querying metrics, anomalies, and incidents
- 📈 **Streamlit dashboard** for interactive visualization
- ⚡ **Real-time analysis** with per-minute feature aggregation

## Project Structure

```
microservice-anomaly-ai/
├── ingest/
│   ├── generate_logs.py       # Generate synthetic microservice logs
│   ├── feature_build.py        # Build features and detect anomalies
│   └── detector_mad.py         # MAD-based anomaly detection module
├── backend_api/
│   └── main.py                 # FastAPI REST API
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── data/                       # Generated data files (created at runtime)
├── requirements.txt
└── README.md
```

## Setup Instructions (Windows)

### 1. Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If you haven't created a virtual environment yet:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Generate Synthetic Logs

```powershell
python -m ingest.generate_logs
```

This creates `data/raw_logs.csv` with 10,000 log entries spanning 2 hours, including an injected incident in the payments service.

**Expected output:**
- 10,000 rows of synthetic logs
- Incident window: payments:/pay service (8 minutes of anomalous behavior)
- Output: `data/raw_logs.csv`

### 4. Build Features and Detect Anomalies

```powershell
python -m ingest.feature_build
```

This processes raw logs into per-minute features and applies MAD anomaly detection.

**Expected output:**
- `data/features_1min.csv` - Aggregated per-minute metrics
- `data/scored_1min.csv` - Features with anomaly scores and severity levels
- Console summary showing anomaly counts by severity

### 5. Run the API (Optional)

```powershell
uvicorn backend_api.main:app --reload
```

The API will be available at `http://localhost:8000`

**Available endpoints:**
- `GET /health` - Health check
- `GET /metrics?service=&endpoint=&limit=200` - Query metrics
- `GET /anomalies?service=&endpoint=&severity=&limit=50` - Query anomalies
- `GET /incidents?limit=20` - Query grouped incidents

API docs available at: `http://localhost:8000/docs`

### 6. Run the Dashboard

```powershell
streamlit run dashboard/app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## Demo Script for Imagine Cup

Follow these steps to demonstrate the system:

### Step 1: Generate Data
```powershell
python -m ingest.generate_logs
```
**Show**: Console output with incident details (payments:/pay service, 8-minute window)

### Step 2: Run Feature Engineering & Anomaly Detection
```powershell
python -m ingest.feature_build
```
**Show**: 
- Summary statistics
- Anomaly counts by severity (Info/Warning/Critical)
- Top affected services

### Step 3: Launch Dashboard
```powershell
streamlit run dashboard/app.py
```

**Demo Flow:**
1. **Overview**: Show the P95 latency chart - point out the anomaly spike
2. **Filter by Service**: Select "payments" service to isolate the incident
3. **Filter by Endpoint**: Select "/pay" endpoint
4. **Anomaly Table**: Show detected anomalies with MAD z-scores and severity
5. **Incident Summary**: Highlight the grouped incident window with:
   - Start and end times
   - Peak latency (5-20x normal)
   - Elevated error rate (10-30%)
   - Severity level (Warning/Critical)

### Step 4: Show API Endpoints (Optional)

Start the API:
```powershell
uvicorn backend_api.main:app --reload
```

**Demo in browser or with curl:**

1. **Health check:**
   ```
   http://localhost:8000/health
   ```

2. **Query anomalies for payments service:**
   ```
   http://localhost:8000/anomalies?service=payments&endpoint=/pay
   ```
   **Show**: List of anomalies with timestamps, latency, error rates, and MAD scores

3. **Query incidents:**
   ```
   http://localhost:8000/incidents?limit=10
   ```
   **Show**: Grouped incident windows with aggregated statistics

4. **Interactive API docs:**
   ```
   http://localhost:8000/docs
   ```
   **Show**: Try out endpoints directly in the Swagger UI

## Key Talking Points for Judges

1. **No Training Required**: Uses MAD (Median Absolute Deviation), a robust statistical method that works immediately without historical training data

2. **Real-time Detection**: Processes logs in 1-minute windows, suitable for production monitoring

3. **Explainable**: MAD z-scores provide interpretable anomaly severity (threshold of 3.5 standard deviations)

4. **Practical Implementation**: 
   - Handles multiple services and endpoints
   - Detects both latency spikes and error rate increases
   - Groups consecutive anomalies into incidents
   - Configurable thresholds and parameters

5. **Scalable Architecture**: Modular design with separate components for ingestion, analysis, API, and visualization

## Configuration

### Log Generation Parameters
Edit `ingest/generate_logs.py`:
- `N_ROWS`: Number of log entries (default: 10,000)
- `INCIDENT_START_MINUTE`: When to inject incident (default: 45)
- `INCIDENT_DURATION_MINUTES`: Incident duration (default: 8)

### Anomaly Detection Parameters
Edit `ingest/feature_build.py` or call directly:
- MAD threshold: 3.5 (modify in `detector_mad.add_mad_anomalies()`)
- Time window: 1 minute (change bucket size in `build_per_minute_features()`)

## Troubleshooting

**Issue**: Dashboard shows "Data file not found"
- **Solution**: Run steps 3 and 4 to generate the data files

**Issue**: API returns 404 for data
- **Solution**: Ensure `data/scored_1min.csv` exists (run step 4)

**Issue**: Import errors
- **Solution**: Make sure you're in the repository root and virtual environment is activated

## Technical Details

**Anomaly Detection Method**: MAD (Median Absolute Deviation)
- Modified z-score: `0.6745 * (x - median) / MAD`
- Threshold: 3.5 (approximately 3.5 standard deviations)
- Groups by: (service, endpoint)
- Metric: P95 latency

**Severity Classification**:
- **Info**: Normal behavior (|MAD z-score| ≤ 3.5)
- **Warning**: Anomalous latency (|MAD z-score| > 3.5)
- **Critical**: Anomalous latency + high error rate (≥5%)

## Future Enhancements

- [ ] Real-time log streaming with Kafka/RabbitMQ
- [ ] Multiple detection algorithms (Isolation Forest, LSTM)
- [ ] Alert notifications (email, Slack, PagerDuty)
- [ ] Historical baseline comparison
- [ ] Automated root cause analysis
- [ ] Multi-metric correlation detection

## License

MIT License - Feel free to use for your projects!

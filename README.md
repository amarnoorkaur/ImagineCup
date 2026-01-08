# 🎯 Microservice Anomaly Detection (MAD)

> **Imagine Cup 2026 Submission**  
> Training-free, real-time anomaly detection for microservices using statistical methods.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

---

## 📋 Table of Contents

- [Problem](#-problem)
- [Solution Overview](#-solution-overview)
- [Data Flow](#-data-flow)
- [Quick Start](#-quick-start)
- [Demo Script](#-demo-script)
- [API Reference](#-api-reference)
- [Evaluation](#-evaluation)
- [Future Work](#-future-work)

---

## 🔴 Problem

**Microservice architectures are hard to monitor.**

Modern applications consist of dozens of interconnected services. When something goes wrong:
- 🕐 **Detection is slow** — Teams rely on user complaints or manual dashboard watching
- 🔍 **Root cause is unclear** — Which service caused the cascade failure?
- 📊 **ML solutions need training data** — New services have no historical baseline
- 💰 **Downtime is expensive** — Every minute of outage costs revenue and reputation

**We need anomaly detection that works immediately, without training.**

---

## ✅ Solution Overview

**MAD (Median Absolute Deviation)** — A robust statistical approach that detects anomalies in real-time without requiring any training data.

### Core Components

| Component | Description |
|-----------|-------------|
| **🔍 MAD Detector** | Statistical anomaly scoring using modified z-scores |
| **🚨 Incident Grouping** | Clusters consecutive anomalies into actionable incidents |
| **🎯 Culprit Finder** | Ranks service/endpoints by severity to identify root cause |
| **💡 Explanation Engine** | Generates human-readable incident explanations |

### How MAD Works

```
Modified Z-Score = 0.6745 × (value - median) / MAD
```

- **Threshold**: |z-score| > 3.5 → Anomaly
- **Why MAD?** Robust to outliers (unlike standard deviation)
- **No training**: Works on first data point

### Severity Classification

| Severity | Condition | Action |
|----------|-----------|--------|
| 🟢 **Info** | Normal behavior | None |
| 🟡 **Warning** | Latency spike (z > 3.5) | Monitor |
| 🔴 **Critical** | Spike + Error rate ≥ 5% | Alert |

### Production Guardrails

To reduce false positives, we implement:
- **Warmup Period**: First 15 minutes per service ignored
- **Minimum Data Points**: Need 20+ rows before flagging
- **Traffic Threshold**: Low-traffic periods (< 20 req/min) skipped

---

## 🔄 Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW DIAGRAM                             │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐
  │  Microservices   │     (auth, catalog, checkout, payments)
  │    Log Data      │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐     generate_logs.py
  │  Raw Logs CSV    │───► 10,000 entries, 2-hour window
  │  + Incident Meta │     Injected incident: payments:/pay
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐     feature_build.py
  │  Feature Engine  │───► 1-minute aggregation
  │  (Per-Minute)    │     req_count, error_rate, p95_latency
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐     detector_mad.py
  │   MAD Detector   │───► mad_z scores, is_anomaly, severity
  │   + Guardrails   │     Warmup, min_points, min_req_count
  └────────┬─────────┘
           │
           ├─────────────────┬─────────────────┐
           ▼                 ▼                 ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │   REST API   │   │  Dashboard   │   │  Evaluation  │
  │   (FastAPI)  │   │  (Streamlit) │   │   Metrics    │
  └──────────────┘   └──────────────┘   └──────────────┘
           │                 │
           ▼                 ▼
  ┌──────────────┐   ┌──────────────┐
  │   /culprit   │   │   Culprit    │
  │   /explain   │   │   Analysis   │
  └──────────────┘   └──────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip

### One-Command Demo

```powershell
# Clone and setup
git clone https://github.com/amarnoorkaur/ImagineCup.git
cd ImagineCup

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run complete demo pipeline
python run_demo.py
```

### Manual Step-by-Step

```powershell
# 1. Generate synthetic logs with injected incident
python -m ingest.generate_logs

# 2. Build features and detect anomalies
python -m ingest.feature_build

# 3. Run evaluation metrics
python evaluate.py

# 4. Start API (Terminal 1)
uvicorn backend_api.main:app --reload

# 5. Start Dashboard (Terminal 2)
streamlit run dashboard/app.py
```

### Output Files

| File | Description |
|------|-------------|
| `data/raw_logs.csv` | 10,000 synthetic log entries |
| `data/incident_meta.json` | Ground truth incident info |
| `data/features_1min.csv` | Per-minute aggregated features |
| `data/scored_1min.csv` | Features + anomaly scores |
| `data/eval_report.json` | Evaluation metrics |

---

## 🎬 Demo Script

> **Total Time: ~5 minutes**

### Act 1: The Problem (30 seconds)

*"Imagine you're running an e-commerce platform with 4 microservices. At 10:45 AM, your payments service starts failing. How quickly can you detect and identify the issue?"*

### Act 2: Generate Data (1 minute)

```powershell
python run_demo.py
```

**Show console output:**
- ✅ 10,000 logs generated
- ✅ Incident injected: `payments:/pay` (minute 45-53)
- ✅ Anomalies detected with MAD
- ✅ Evaluation: Precision, Recall, Detection Delay

### Act 3: Dashboard Demo (2 minutes)

```powershell
streamlit run dashboard/app.py
```

**Click sequence:**

1. **🎯 Culprit Analysis Section**
   - Point out: Top culprit identified as `payments:/pay`
   - Show: Peak MAD z-score, severity, error rate

2. **📈 P95 Latency Chart**
   - Point out: Visible spike around minute 45

3. **🔍 Filters → Service: payments**
   - Show: Anomalies isolated to payments service

4. **🚨 Incident Summary**
   - Show: Grouped incident with start/end times

5. **💡 Incident Explanation**
   - Select incident from dropdown
   - Click "Generate Explanation"
   - Show: Why flagged, impact, likely causes, recommended actions

### Act 4: API Demo (1 minute)

Open browser to: `http://localhost:8000/docs`

**Live API calls:**

```
GET /culprit?minutes=15
```
→ Returns top culprit with ranking

```
GET /explain_incident?service=payments&endpoint=/pay&start_ts=...&end_ts=...
```
→ Returns structured explanation

### Act 5: Evaluation (30 seconds)

```powershell
python evaluate.py
```

**Show metrics:**
- True Positives: 8
- False Positives: ~2
- Detection Delay: < 1 minute
- Precision: ~80%

*"We detected the incident within 1 minute with 80% precision — no training required."*

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Query raw metrics |
| `/anomalies` | GET | Query detected anomalies |
| `/incidents` | GET | Query grouped incidents |
| `/culprit` | GET | Find top culprit service/endpoint |
| `/explain_incident` | GET | Generate incident explanation |

### Example: Get Top Culprit

```bash
curl "http://localhost:8000/culprit?minutes=15&top_k=3"
```

**Response:**
```json
{
  "window_minutes": 15,
  "total_anomalies": 8,
  "culprit": {
    "service": "payments",
    "endpoint": "/pay",
    "peak_abs_mad": 12.45,
    "peak_severity": "Critical",
    "peak_error_rate": 0.25
  },
  "top": [...]
}
```

### Example: Explain Incident

```bash
curl "http://localhost:8000/explain_incident?service=payments&endpoint=/pay&start_ts=2026-01-06T10:45:00&end_ts=2026-01-06T10:53:00"
```

**Response:**
```json
{
  "summary": "Critical incident on payments:/pay...",
  "why_flagged": ["P95 latency spiked to 4500ms", "MAD z-score of 12.5 exceeded threshold"],
  "impact": ["Users experiencing failed requests", "SLO may be breached"],
  "likely_causes": ["Downstream service failure", "Database issues"],
  "recommended_actions": ["Check service logs immediately", "Verify dependencies"],
  "confidence": "High"
}
```

---

## 📊 Evaluation

We evaluate against injected ground truth incidents.

### Metrics Computed

| Metric | Formula | Description |
|--------|---------|-------------|
| **True Positives (TP)** | Anomalies inside incident window | Correctly detected |
| **False Positives (FP)** | Anomalies outside incident window | False alarms |
| **Detection Delay** | First TP timestamp - Incident start | Time to detect |
| **Precision Proxy** | TP / (TP + FP) | Accuracy of alerts |
| **Recall Proxy** | TP / Expected rows | Coverage of incident |

### Sample Results

```
======================================================================
ANOMALY DETECTION EVALUATION REPORT
======================================================================

📋 GROUND TRUTH:
   Service:    payments
   Endpoint:   /pay
   Duration:   8 minutes

📊 METRICS:
   True Positives:    8
   False Positives:   2
   Detection Delay:   0.0 minutes
   Precision Proxy:   80.00%
   Recall Proxy:      100.00%

💡 INTERPRETATION:
   ✅ High precision - most detected anomalies are true incidents
   ✅ Fast detection - incident caught within 1 minute
======================================================================
```

### Run Evaluation

```powershell
python evaluate.py
```

Output saved to: `data/eval_report.json`

---

## 🔮 Future Work

### 1. Azure OpenAI Integration (Planned)

Replace templated explanations with AI-generated insights:

```python
# Current: Rule-based templates
explanation = generate_incident_explanation(...)  # Local, deterministic

# Future: Azure OpenAI
explanation = await azure_openai_explain(...)     # AI-powered, contextual
```

**Fallback Strategy:**
- Primary: Azure OpenAI for rich explanations
- Fallback: Local templates (current implementation)
- Graceful degradation if API unavailable

### 2. Additional Enhancements

| Feature | Status | Description |
|---------|--------|-------------|
| Azure OpenAI Explanations | 🔜 Planned | AI-powered root cause analysis |
| Real-time Streaming | 🔜 Planned | Kafka/Event Hubs integration |
| Multi-metric Correlation | 🔜 Planned | Cross-service dependency detection |
| Alerting Integration | 🔜 Planned | PagerDuty, Slack, Teams |
| Azure Monitor Integration | 🔜 Planned | Native Azure observability |
| Isolation Forest | 🔜 Planned | ML-based detection option |

### 3. Azure Architecture (Target)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Azure Cloud                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐              │
│  │ Azure Event  │───▶│ Azure        │───▶│ Azure     │              │
│  │ Hubs         │    │ Functions    │    │ Cosmos DB │              │
│  └──────────────┘    └──────────────┘    └───────────┘              │
│                             │                                        │
│                             ▼                                        │
│                      ┌──────────────┐                                │
│                      │ Azure OpenAI │                                │
│                      │ (Explanations)│                                │
│                      └──────────────┘                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ImagineCup/
├── ingest/
│   ├── __init__.py
│   ├── generate_logs.py      # Synthetic log generation
│   ├── feature_build.py      # Feature engineering pipeline
│   ├── detector_mad.py       # MAD anomaly detection
│   └── culprit.py            # Culprit identification
├── backend_api/
│   └── main.py               # FastAPI REST API
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── data/                     # Generated at runtime
├── evaluate.py               # Evaluation metrics
├── run_demo.py               # One-command demo runner
├── requirements.txt
└── README.md
```

---

## 👥 Team

**Imagine Cup 2026**

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for Imagine Cup 2026** 🏆

[Report Bug](https://github.com/amarnoorkaur/ImagineCup/issues) · [Request Feature](https://github.com/amarnoorkaur/ImagineCup/issues)

</div>

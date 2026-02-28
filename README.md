# README.md

# 📡 LogPulse — AI-Powered Infrastructure Log Intelligence Dashboard

An enterprise-grade DevOps analytics platform that ingests raw system logs, performs severity analysis, detects anomalies using machine learning, and generates AI-powered root cause explanations.

---

## 🏗️ Project Structure
```
logpulse/
├── app.py                  # Streamlit dashboard entry point
├── log_parser.py           # Log ingestion and parsing engine
├── anomaly_detection.py    # Isolation Forest anomaly detection
├── ai_explainer.py         # Claude-powered root cause analysis
├── health_score.py         # System health score computation
├── utils.py                # Shared utilities and helpers
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourorg/logpulse.git
cd logpulse
```

### 2. Create a virtual environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the dashboard
```bash
streamlit run app.py
```

---

## 🔑 API Key Configuration

LogPulse uses the [Anthropic Claude API](https://console.anthropic.com/) for AI root cause analysis.

You can provide your API key in two ways:

**Option A — Sidebar input (recommended for demos):**
Enter the key directly in the Streamlit sidebar under **AI Settings**.

**Option B — Environment variable:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

---

## 🧠 Feature Overview

### Log Parsing Engine
- Supports multiple log formats: ISO timestamps, bracket notation, syslog, severity-first
- Graceful handling of malformed lines
- Returns structured pandas DataFrame

### Severity Analytics
- Bar and pie charts for severity distribution
- Time-series log volume by severity
- Health score trend over time

### Anomaly Detection
- Aggregates error counts into configurable time buckets
- Applies `IsolationForest` from scikit-learn
- Flags anomalous time windows with visual markers

### AI Root Cause Analysis
- Extracts top 5 recurring ERROR/CRITICAL messages
- Sends structured prompt to Claude
- Returns per-error root cause, impact, and remediation steps

### System Health Score
- Formula: `Health Score = 100 − Normalized Weighted Impact`
- Weights: CRITICAL=5, ERROR=3, WARNING=1
- Interpretations: Healthy / Moderate Risk / Critical State

---

## 📊 Supported Log Formats

| Format | Example |
|---|---|
| Standard | `2024-01-15 14:23:01 ERROR Failed to connect` |
| Bracket | `[2024-01-15 14:23:01] [ERROR] Failed to connect` |
| Syslog | `Jan 15 14:23:01 hostname ERROR Failed to connect` |
| Severity-first | `ERROR 2024-01-15 14:23:01 Failed to connect` |

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| ML / Anomaly Detection | scikit-learn (IsolationForest) |
| Visualization | Plotly |
| AI Layer | Anthropic Claude API |
| Language | Python 3.11 |

---

## 🎥 Demo Flow

1. Click **Load Sample Log** to ingest a synthetic infrastructure log
2. Review the **Health Score** and severity breakdown
3. Inspect the **Timeline Analysis** for log volume spikes
4. Navigate to **Anomaly Detection** to see flagged time windows
5. Enter your Anthropic API key and click **Generate AI Explanation**
6. Export filtered logs as CSV using the download button

---

## 📄 License

MIT License. See `LICENSE` for details.
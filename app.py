# app.py
"""
LogPulse — AI-Powered Infrastructure Log Intelligence Dashboard
Main Streamlit application entry point.
"""
from ai_explainer import explain_errors, get_top_recurring_errors, PROVIDER_MODELS
from streaming import stream_log_dashboard
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import logging

from log_parser import parse_log_file, get_parse_summary
from anomaly_detection import detect_anomalies, get_anomaly_summary
from health_score import compute_health_score, get_score_color, compute_score_trend
from ai_explainer import explain_errors, get_top_recurring_errors
from utils import (
    configure_logging,
    df_to_csv_bytes,
    format_health_score_display,
    generate_sample_log_file,
    severity_color_map,
    summarize_dataframe,
    truncate_message,
)

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="LogPulse",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal global styling ────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .metric-label { font-size: 0.85rem; color: #aaaaaa; }
        .section-divider { border-top: 1px solid #2e2e2e; margin: 1.5rem 0; }
        .stAlert { border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/pulse.png", width=40)
    st.title("LogPulse")
    st.caption("Infrastructure Log Intelligence")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Log File",
        type=["log", "txt"],
        help="Supports standard syslog, application logs, and custom formats.",
    )

    use_sample = st.button("Load Sample Log", width="stretch")

    st.divider()
    st.subheader("Filters")

    severity_filter = st.multiselect(
        "Severity Levels",
        options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    st.subheader("AI Settings")

    ai_provider = st.selectbox(
        "LLM Provider",
        options=list(PROVIDER_MODELS.keys()),
    )

    ai_model = st.selectbox(
        "Model",
        options=PROVIDER_MODELS[ai_provider],
    )

    anthropic_api_key = st.text_input(
        f"{ai_provider} API Key",
        type="password",
        placeholder="Enter your API key...",
    )

    st.divider()
    st.subheader("Detection Settings")
    anomaly_freq = st.selectbox(
        "Anomaly Bucket Size",
        options=["1min", "2min", "5min", "10min"],
        index=0,
    )
    contamination = st.slider(
        "Contamination Rate",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        help="Expected proportion of anomalous time buckets.",
    )


# ── Session state ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None


# ── File ingestion ────────────────────────────────────────────────────────────
def load_log(file_obj):
    with st.spinner("Parsing log file..."):
        try:
            df = parse_log_file(file_obj)
            st.session_state.df = df
        except ValueError as exc:
            st.error(f"Parse error: {exc}")
            st.stop()


if use_sample:
    load_log(generate_sample_log_file())

if uploaded_file:
    load_log(uploaded_file)


# ── No data state ─────────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown("## 📡 LogPulse")
    st.info(
        "Upload a `.log` or `.txt` file using the sidebar, or click **Load Sample Log** to explore a demo."
    )
    st.stop()


# ── Apply filters ─────────────────────────────────────────────────────────────
df_full: pd.DataFrame = st.session_state.df
df: pd.DataFrame = df_full[df_full["severity"].isin(severity_filter)].copy()

if df.empty:
    st.warning("No log entries match the selected severity filters.")
    st.stop()


# ── Health score ──────────────────────────────────────────────────────────────
health = compute_health_score(df)
score_color = get_score_color(health["interpretation"])


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📡 LogPulse — Infrastructure Log Intelligence")

col_score, col_interp, col_total, col_span, col_nullts = st.columns(5)

with col_score:
    st.metric("Health Score", format_health_score_display(health["score"]))

with col_interp:
    st.markdown(
        f"<div style='padding-top:0.6rem;font-size:1.1rem;font-weight:600;color:{score_color}'>"
        f"{health['interpretation']}</div>",
        unsafe_allow_html=True,
    )

summary = summarize_dataframe(df)

with col_total:
    st.metric("Total Entries", f"{summary['total_entries']:,}")

with col_span:
    st.metric("Time Span", f"{summary['time_span_minutes']} min")

with col_nullts:
    st.metric("Unparsed Timestamps", summary["null_timestamps"])

st.divider()


# ── Section 1: Severity Distribution ─────────────────────────────────────────
st.subheader("① Severity Distribution")

col_bar, col_pie = st.columns(2)
color_map = severity_color_map()

sev_counts = (
    df["severity"]
    .value_counts()
    .reindex(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], fill_value=0)
    .reset_index()
)
sev_counts.columns = ["Severity", "Count"]

with col_bar:
    fig_bar = px.bar(
        sev_counts,
        x="Severity",
        y="Count",
        color="Severity",
        color_discrete_map=color_map,
        template="plotly_dark",
        title="Log Entries by Severity",
    )
    fig_bar.update_layout(showlegend=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig_bar, width="stretch")

with col_pie:
    fig_pie = px.pie(
        sev_counts[sev_counts["Count"] > 0],
        names="Severity",
        values="Count",
        color="Severity",
        color_discrete_map=color_map,
        template="plotly_dark",
        title="Severity Composition",
        hole=0.4,
    )
    fig_pie.update_layout(margin=dict(t=40, b=20))
    st.plotly_chart(fig_pie, width="stretch")

st.divider()


# ── Section 2: Timeline Analysis ──────────────────────────────────────────────
st.subheader("② Timeline Analysis")

df_valid_ts = df.dropna(subset=["timestamp"]).copy()

if df_valid_ts.empty:
    st.warning("No valid timestamps available for timeline analysis.")
else:
    df_valid_ts["bucket"] = df_valid_ts["timestamp"].dt.floor("1min")

    timeline = (
        df_valid_ts.groupby(["bucket", "severity"], observed=True)
        .size()
        .reset_index(name="count")
    )

    fig_line = px.line(
        timeline,
        x="bucket",
        y="count",
        color="severity",
        color_discrete_map=color_map,
        template="plotly_dark",
        title="Log Volume Over Time (1-min buckets)",
        labels={"bucket": "Time", "count": "Log Count"},
    )
    fig_line.update_layout(margin=dict(t=40, b=20))
    st.plotly_chart(fig_line, width="stretch")

    col_trend_title, col_dl = st.columns([4, 1])
    with col_trend_title:
        st.caption("Health Score Trend Over Time")
    with col_dl:
        csv_bytes = df_to_csv_bytes(df)
        st.download_button(
            "⬇ Export CSV",
            data=csv_bytes,
            file_name="logpulse_export.csv",
            mime="text/csv",
            width="stretch",
        )

    score_trend = compute_score_trend(df, freq="5min")
    if not score_trend.empty:
        fig_trend = px.line(
            score_trend,
            x="timestamp",
            y="score",
            template="plotly_dark",
            title="System Health Score Trend",
            labels={"timestamp": "Time", "score": "Health Score"},
        )
        fig_trend.add_hline(y=75, line_dash="dot", line_color="#2ecc71", annotation_text="Healthy threshold")
        fig_trend.add_hline(y=40, line_dash="dot", line_color="#e74c3c", annotation_text="Critical threshold")
        fig_trend.update_layout(margin=dict(t=40, b=20), yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_trend, width="stretch")

st.divider()


# ── Section 3: Anomaly Detection ─────────────────────────────────────────────
st.subheader("③ Anomaly Detection")

try:
    aggregated, anomalies = detect_anomalies(
        df, freq=anomaly_freq, contamination=contamination
    )
    anomaly_summary = get_anomaly_summary(aggregated, anomalies)

    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    col_a1.metric("Total Buckets", anomaly_summary["total_buckets"])
    col_a2.metric("Anomalous Buckets", anomaly_summary["anomalous_buckets"])
    col_a3.metric("Anomaly Rate", f"{anomaly_summary['anomaly_rate_pct']}%")
    col_a4.metric("Peak Error Count", anomaly_summary["peak_error_count"])

    fig_anomaly = go.Figure()

    fig_anomaly.add_trace(
        go.Scatter(
            x=aggregated["timestamp"],
            y=aggregated["total_errors"],
            mode="lines",
            name="Error Frequency",
            line=dict(color="#3498db", width=1.5),
        )
    )

    if not anomalies.empty:
        fig_anomaly.add_trace(
            go.Scatter(
                x=anomalies["timestamp"],
                y=anomalies["total_errors"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#e74c3c", size=10, symbol="x"),
            )
        )

    fig_anomaly.update_layout(
        template="plotly_dark",
        title="Error Frequency with Anomaly Markers",
        xaxis_title="Time",
        yaxis_title="Error Count",
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_anomaly, width="stretch")

    if not anomalies.empty:
        with st.expander("View Anomalous Time Windows"):
            st.dataframe(
                anomalies[["timestamp", "WARNING", "ERROR", "CRITICAL", "total_errors", "anomaly_score"]]
                .sort_values("anomaly_score")
                .reset_index(drop=True),
                width="stretch",
            )

except ValueError as exc:
    st.warning(f"Anomaly detection skipped: {exc}")

st.divider()


# ── Section 4: Top Errors + AI Explanation ────────────────────────────────────
st.subheader("④ Top Errors & AI Root Cause Analysis")

top_errors = get_top_recurring_errors(df)

if top_errors.empty:
    st.info("No ERROR or CRITICAL entries found in the current filtered dataset.")
else:
    display_errors = top_errors.copy()
    display_errors["message"] = display_errors["message"].apply(truncate_message)

    st.dataframe(display_errors, width="stretch")

    st.markdown("#### AI Diagnostic Report")

    if not anthropic_api_key:
        st.warning("Enter your Anthropic API key in the sidebar to enable AI root cause analysis.")
    else:
        if st.button("🧠 Generate AI Explanation", width="content"):
            with st.spinner(f"Analyzing errors with {ai_provider} — {ai_model}..."):
                result = explain_errors(
                    df,
                    provider=ai_provider,
                    model=ai_model,
                    api_key=anthropic_api_key,
                )
    st.divider()

# ── Section 5: Real-Time Streaming Replay ─────────────────────────────────────
st.subheader("⑤ Real-Time Log Stream Replay")

col_speed, col_launch = st.columns([2, 1])

with col_speed:
    stream_speed = st.select_slider(
        "Replay Speed",
        options=[0.001, 0.01, 0.03, 0.05, 0.1, 0.2],
        value=0.03,
        format_func=lambda x: {
            0.001: "⚡ Turbo",
            0.01: "🚀 Fast",
            0.03: "▶️ Normal",
            0.05: "🐢 Slow",
            0.1: "🔬 Step",
            0.2: "🧪 Debug",
        }[x],
    )

with col_launch:
    st.write("")
    st.write("")
    if st.button("▶ Start Live Stream", width="stretch"):
        stream_log_dashboard(df, speed=stream_speed)
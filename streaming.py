# streaming.py
"""
Simulated real-time log streaming module for LogPulse.
Replays a parsed log DataFrame line by line with live dashboard updates.
"""

import streamlit as st
import pandas as pd
import time
import plotly.express as px
from health_score import compute_health_score, get_score_color
from utils import severity_color_map, format_health_score_display

SEVERITY_COLORS = severity_color_map()


def stream_log_dashboard(df: pd.DataFrame, speed: float = 0.05) -> None:
    """
    Replay a parsed log DataFrame in real time, updating dashboard components
    incrementally to simulate live log ingestion.

    Parameters
    ----------
    df : pd.DataFrame
        Fully parsed log DataFrame from log_parser.parse_log_file.
    speed : float
        Delay in seconds between each ingested line. Lower = faster replay.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    st.markdown("### 🔴 Live Log Stream")
    st.caption("Replaying log file in real time — watch the health score update as events are ingested.")

    col_score, col_interp, col_total, col_errors = st.columns(4)

    score_placeholder = col_score.empty()
    interp_placeholder = col_interp.empty()
    total_placeholder = col_total.empty()
    error_placeholder = col_errors.empty()

    chart_placeholder = st.empty()
    log_feed_placeholder = st.empty()

    stop_button = st.button("⏹ Stop Stream", key="stop_stream")

    ingested_rows = []

    for i, row in df.iterrows():
        if stop_button:
            st.warning("Stream stopped by user.")
            break

        ingested_rows.append(row)
        current_df = pd.DataFrame(ingested_rows)

        # Health score
        health = compute_health_score(current_df)
        score_color = get_score_color(health["interpretation"])

        score_placeholder.metric(
            "Health Score",
            format_health_score_display(health["score"]),
        )

        interp_placeholder.markdown(
            f"<div style='padding-top:0.6rem;font-size:1rem;"
            f"font-weight:600;color:{score_color}'>"
            f"{health['interpretation']}</div>",
            unsafe_allow_html=True,
        )

        total_placeholder.metric("Lines Ingested", f"{len(current_df):,}")

        error_count = current_df[
            current_df["severity"].isin(["ERROR", "CRITICAL"])
        ].shape[0]
        error_placeholder.metric("Errors Detected", error_count)

        # Live severity bar chart — update every 10 lines for performance
        if i % 10 == 0 or i == len(df) - 1:
            sev_counts = (
                current_df["severity"]
                .value_counts()
                .reindex(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], fill_value=0)
                .reset_index()
            )
            sev_counts.columns = ["Severity", "Count"]

            fig = px.bar(
                sev_counts,
                x="Severity",
                y="Count",
                color="Severity",
                color_discrete_map=SEVERITY_COLORS,
                template="plotly_dark",
                title="Live Severity Distribution",
            )
            fig.update_layout(showlegend=False, margin=dict(t=40, b=10), height=300)
            chart_placeholder.plotly_chart(fig, use_container_width=True)

        # Rolling log feed — last 12 lines
        recent = pd.DataFrame(ingested_rows[-12:])
        recent_display = recent[["timestamp", "severity", "message"]].copy()
        recent_display["message"] = recent_display["message"].str[:80]
        log_feed_placeholder.dataframe(recent_display, use_container_width=True)

        time.sleep(speed)

    st.success(f"Stream complete. {len(ingested_rows)} log entries processed.")
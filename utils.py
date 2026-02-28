# utils.py
"""
Shared utility functions for LogPulse.
Covers formatting, sample data generation, CSV export, and display helpers.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from io import StringIO, BytesIO
import logging

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

SAMPLE_MESSAGES = {
    "DEBUG": [
        "Cache lookup initiated for key=session_token_8821",
        "DB connection pool size: 10/50",
        "Heartbeat ping sent to service mesh",
    ],
    "INFO": [
        "Service started on port 8080",
        "User authentication successful for uid=4421",
        "Scheduled job completed in 320ms",
        "Config reload triggered by SIGHUP",
    ],
    "WARNING": [
        "Memory usage at 78% — approaching threshold",
        "Response time degraded: avg 1.8s over last 60s",
        "Retry attempt 2/3 for downstream service call",
        "Disk usage on /var/log exceeding 85%",
    ],
    "ERROR": [
        "Failed to connect to PostgreSQL: connection refused on port 5432",
        "NullPointerException in OrderService.processPayment() line 214",
        "Timeout waiting for response from auth-service after 5000ms",
        "Kafka consumer group lag exceeded threshold: 45000 messages",
        "Failed to write to S3 bucket: AccessDenied",
    ],
    "CRITICAL": [
        "Segmentation fault detected in core process PID 1042",
        "Database cluster unreachable — all replicas down",
        "OOM Killer invoked: process nginx killed",
        "SSL certificate expired for api.internal.corp",
    ],
}


def generate_sample_logs(
    n_lines: int = 500,
    start_time: datetime = None,
    inject_anomaly: bool = True,
) -> str:
    """
    Generate a synthetic log file string for demo and testing purposes.

    Parameters
    ----------
    n_lines : int
        Total number of log lines to generate.
    start_time : datetime, optional
        Starting timestamp. Defaults to 2 hours ago.
    inject_anomaly : bool
        If True, inject a burst of ERROR/CRITICAL entries to simulate an anomaly spike.

    Returns
    -------
    str
        Newline-delimited log string in standard format.
    """
    if start_time is None:
        start_time = datetime.now() - timedelta(hours=2)

    severity_weights = [0.10, 0.50, 0.20, 0.15, 0.05]
    lines = []
    current_time = start_time

    anomaly_start = int(n_lines * 0.60)
    anomaly_end = int(n_lines * 0.70)

    for i in range(n_lines):
        current_time += timedelta(seconds=random.randint(1, 15))

        if inject_anomaly and anomaly_start <= i <= anomaly_end:
            severity = random.choice(["ERROR", "CRITICAL"])
        else:
            severity = random.choices(SEVERITY_LEVELS, weights=severity_weights, k=1)[0]

        message = random.choice(SAMPLE_MESSAGES[severity])
        timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"{timestamp_str} {severity} {message}")

    return "\n".join(lines)


def generate_sample_log_file() -> BytesIO:
    """
    Wrap generate_sample_logs output in a BytesIO object for Streamlit file simulation.

    Returns
    -------
    BytesIO
        In-memory file-like object containing sample log content.
    """
    content = generate_sample_logs()
    buffer = BytesIO(content.encode("utf-8"))
    buffer.name = "sample_infrastructure.log"
    return buffer


def format_health_score_display(score: float) -> str:
    """
    Format health score float for clean dashboard display.

    Parameters
    ----------
    score : float

    Returns
    -------
    str
        Formatted string like '87.4 / 100'
    """
    return f"{score:.1f} / 100"


def severity_color_map() -> dict:
    """
    Return a consistent color mapping for severity levels used across charts.

    Returns
    -------
    dict
        Severity level to hex color string.
    """
    return {
        "DEBUG": "#95a5a6",
        "INFO": "#3498db",
        "WARNING": "#f39c12",
        "ERROR": "#e67e22",
        "CRITICAL": "#e74c3c",
    }


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Serialize a DataFrame to UTF-8 encoded CSV bytes for Streamlit download.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    bytes
    """
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def truncate_message(message: str, max_length: int = 80) -> str:
    """
    Truncate a log message string for table display.

    Parameters
    ----------
    message : str
    max_length : int

    Returns
    -------
    str
    """
    return message if len(message) <= max_length else message[:max_length] + "..."


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger for LogPulse application.

    Parameters
    ----------
    level : int
        Logging level constant from the logging module.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def summarize_dataframe(df: pd.DataFrame) -> dict:
    """
    Return lightweight metadata about a parsed log DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
    """
    valid_ts = df["timestamp"].dropna()
    return {
        "total_entries": len(df),
        "unique_messages": df["message"].nunique(),
        "time_span_minutes": round(
            (valid_ts.max() - valid_ts.min()).total_seconds() / 60, 1
        )
        if len(valid_ts) >= 2
        else 0,
        "null_timestamps": int(df["timestamp"].isna().sum()),
    }
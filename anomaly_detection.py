# anomaly_detection.py
"""
Anomaly Detection Module for LogPulse.
Applies Isolation Forest on time-bucketed error frequency to surface anomalous spikes.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

ANOMALY_SEVERITY_LEVELS = {"ERROR", "CRITICAL", "WARNING"}


def aggregate_error_frequency(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Aggregate log counts per time bucket, segmented by severity.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed log DataFrame from log_parser.parse_log_file.
    freq : str
        Pandas time frequency string for bucketing (default: '1min').

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (period index), ERROR, WARNING, CRITICAL, total_errors
    """
    df_valid = df.dropna(subset=["timestamp"]).copy()
    df_errors = df_valid[df_valid["severity"].isin(ANOMALY_SEVERITY_LEVELS)].copy()

    if df_errors.empty:
        raise ValueError("No ERROR/WARNING/CRITICAL entries found for anomaly detection.")

    df_errors["bucket"] = df_errors["timestamp"].dt.floor(freq)

    pivot = (
        df_errors.groupby(["bucket", "severity"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["WARNING", "ERROR", "CRITICAL"], fill_value=0)
    )

    # Fill missing time buckets across full range
    full_range = pd.date_range(pivot.index.min(), pivot.index.max(), freq=freq)
    pivot = pivot.reindex(full_range, fill_value=0)
    pivot.index.name = "timestamp"

    pivot["total_errors"] = pivot.sum(axis=1)
    return pivot.reset_index()


def detect_anomalies(
    df: pd.DataFrame,
    freq: str = "1min",
    contamination: float = 0.05,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Isolation Forest anomaly detection on time-bucketed error frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed log DataFrame.
    freq : str
        Time bucketing frequency.
    contamination : float
        Expected proportion of anomalies in the dataset.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - aggregated: full frequency DataFrame with anomaly labels
        - anomalies: subset of rows flagged as anomalous
    """
    aggregated = aggregate_error_frequency(df, freq=freq)

    features = aggregated[["WARNING", "ERROR", "CRITICAL", "total_errors"]].values

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    raw_predictions = model.fit_predict(features)
    anomaly_scores = model.decision_function(features)

    aggregated["anomaly_score"] = anomaly_scores
    aggregated["is_anomaly"] = raw_predictions == -1

    anomalies = aggregated[aggregated["is_anomaly"]].copy()

    logger.info(
        "Anomaly detection complete. %d anomalous buckets detected out of %d total.",
        len(anomalies),
        len(aggregated),
    )

    return aggregated, anomalies


def get_anomaly_summary(aggregated: pd.DataFrame, anomalies: pd.DataFrame) -> dict:
    """
    Summarize anomaly detection results for dashboard display.

    Parameters
    ----------
    aggregated : pd.DataFrame
        Full aggregated DataFrame with anomaly flags.
    anomalies : pd.DataFrame
        Subset of anomalous rows.

    Returns
    -------
    dict
        Summary statistics for the anomaly detection run.
    """
    return {
        "total_buckets": len(aggregated),
        "anomalous_buckets": len(anomalies),
        "anomaly_rate_pct": round(len(anomalies) / max(len(aggregated), 1) * 100, 2),
        "peak_error_bucket": aggregated.loc[aggregated["total_errors"].idxmax(), "timestamp"]
        if not aggregated.empty
        else None,
        "peak_error_count": int(aggregated["total_errors"].max()) if not aggregated.empty else 0,
        "mean_errors_per_bucket": round(aggregated["total_errors"].mean(), 2),
    }
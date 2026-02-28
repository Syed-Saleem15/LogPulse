# health_score.py
"""
System Health Score Engine for LogPulse.
Computes a normalized 0-100 health score based on weighted severity distribution.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

SEVERITY_WEIGHTS = {
    "CRITICAL": 5,
    "ERROR": 3,
    "WARNING": 1,
    "INFO": 0,
    "DEBUG": 0,
}

HEALTH_THRESHOLDS = {
    "healthy": 75,
    "moderate": 40,
}


def compute_weighted_impact(df: pd.DataFrame) -> Tuple[float, dict]:
    """
    Compute the weighted error impact score from severity distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed log DataFrame from log_parser.parse_log_file.

    Returns
    -------
    Tuple[float, dict]
        - weighted_impact: raw weighted sum normalized by total log volume
        - severity_breakdown: per-severity count and contribution
    """
    total_lines = max(len(df), 1)
    severity_counts = df["severity"].value_counts().to_dict()

    breakdown = {}
    weighted_sum = 0.0

    for severity, weight in SEVERITY_WEIGHTS.items():
        count = severity_counts.get(severity, 0)
        contribution = count * weight
        weighted_sum += contribution
        breakdown[severity] = {
            "count": count,
            "weight": weight,
            "contribution": contribution,
        }

    # Normalize: max possible impact if all lines were CRITICAL
    max_possible_impact = total_lines * SEVERITY_WEIGHTS["CRITICAL"]
    normalized_impact = (weighted_sum / max_possible_impact) * 100 if max_possible_impact > 0 else 0.0

    return normalized_impact, breakdown


def compute_health_score(df: pd.DataFrame) -> dict:
    """
    Compute the system health score and return a full result payload.

    Health Score = 100 - normalized_weighted_impact
    Clamped to [0, 100].

    Parameters
    ----------
    df : pd.DataFrame
        Parsed log DataFrame.

    Returns
    -------
    dict
        score, interpretation, severity_breakdown, weighted_impact_pct
    """
    weighted_impact_pct, severity_breakdown = compute_weighted_impact(df)
    score = round(np.clip(100.0 - weighted_impact_pct, 0.0, 100.0), 2)
    interpretation = _interpret_score(score)

    logger.info(
        "Health score computed: %.2f (%s). Weighted impact: %.2f%%",
        score,
        interpretation,
        weighted_impact_pct,
    )

    return {
        "score": score,
        "interpretation": interpretation,
        "weighted_impact_pct": round(weighted_impact_pct, 2),
        "severity_breakdown": severity_breakdown,
    }


def _interpret_score(score: float) -> str:
    """
    Map a numeric health score to a human-readable interpretation.

    Parameters
    ----------
    score : float
        Health score in range [0, 100].

    Returns
    -------
    str
        One of: 'Healthy', 'Moderate Risk', 'Critical State'
    """
    if score >= HEALTH_THRESHOLDS["healthy"]:
        return "Healthy"
    elif score >= HEALTH_THRESHOLDS["moderate"]:
        return "Moderate Risk"
    else:
        return "Critical State"


def get_score_color(interpretation: str) -> str:
    """
    Return a hex color code corresponding to the health interpretation.
    Used for Streamlit metric styling.

    Parameters
    ----------
    interpretation : str

    Returns
    -------
    str
        Hex color string.
    """
    color_map = {
        "Healthy": "#2ecc71",
        "Moderate Risk": "#f39c12",
        "Critical State": "#e74c3c",
    }
    return color_map.get(interpretation, "#95a5a6")


def compute_score_trend(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """
    Compute health score over rolling time windows for trend visualization.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed log DataFrame.
    freq : str
        Time bucketing frequency for trend granularity.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, score, interpretation
    """
    df_valid = df.dropna(subset=["timestamp"]).copy()
    if df_valid.empty:
        return pd.DataFrame(columns=["timestamp", "score", "interpretation"])

    df_valid["bucket"] = df_valid["timestamp"].dt.floor(freq)
    records = []

    for bucket, group in df_valid.groupby("bucket"):
        result = compute_health_score(group)
        records.append({
            "timestamp": bucket,
            "score": result["score"],
            "interpretation": result["interpretation"],
        })

    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
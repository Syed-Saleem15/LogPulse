# log_parser.py
"""
Log Parsing Engine for LogPulse.
Responsible for ingesting raw log files and returning structured DataFrames.
"""

import re
import pandas as pd
from io import StringIO
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Common log format patterns
LOG_PATTERNS = [
    # Standard: 2024-01-15 14:23:01,123 ERROR Some message
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\s+(?P<severity>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\s+(?P<message>.+)",
    # Bracket format: [2024-01-15 14:23:01] [ERROR] Some message
    r"\[(?P<timestamp>\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\]\s+\[(?P<severity>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\]\s+(?P<message>.+)",
    # Syslog-style: Jan 15 14:23:01 hostname ERROR Some message
    r"(?P<timestamp>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+\S+\s+(?P<severity>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\s+(?P<message>.+)",
    # Severity first: ERROR 2024-01-15 14:23:01 Some message
    r"(?P<severity>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\s+(?P<timestamp>\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\s+(?P<message>.+)",
]

SEVERITY_ALIASES = {
    "WARN": "WARNING",
    "FATAL": "CRITICAL",
}

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in LOG_PATTERNS]


def _normalize_severity(severity: str) -> str:
    """Normalize severity aliases to canonical values."""
    return SEVERITY_ALIASES.get(severity.upper(), severity.upper())


def _parse_line(line: str) -> Optional[dict]:
    """
    Attempt to parse a single log line against known patterns.
    Returns a dict with keys: timestamp, severity, message — or None if unparseable.
    """
    line = line.strip()
    if not line:
        return None

    for pattern in COMPILED_PATTERNS:
        match = pattern.match(line)
        if match:
            groups = match.groupdict()
            return {
                "timestamp_raw": groups["timestamp"].strip(),
                "severity": _normalize_severity(groups["severity"]),
                "message": groups["message"].strip(),
            }

    # Fallback: attempt to extract any severity keyword present in the line
    severity_match = re.search(
        r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\b", line, re.IGNORECASE
    )
    if severity_match:
        severity = _normalize_severity(severity_match.group(1))
        return {
            "timestamp_raw": None,
            "severity": severity,
            "message": line,
        }

    return None


def _coerce_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to parse timestamp_raw into a proper datetime column.
    Rows with unparseable timestamps receive NaT.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp_raw"], infer_datetime_format=True, errors="coerce")
    df.drop(columns=["timestamp_raw"], inplace=True)
    return df


def parse_log_file(file_obj) -> pd.DataFrame:
    """
    Parse a log file object (Streamlit UploadedFile or file-like) into a structured DataFrame.

    Parameters
    ----------
    file_obj : file-like
        A readable object with .read() returning bytes or str.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (datetime), severity (str), message (str), line_number (int)
        Rows with completely unrecognized format are dropped; malformed timestamps receive NaT.

    Raises
    ------
    ValueError
        If the file is empty or no parseable lines are found.
    """
    try:
        raw = file_obj.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
    except Exception as exc:
        raise ValueError(f"Failed to read log file: {exc}") from exc

    lines = raw.splitlines()
    if not lines:
        raise ValueError("The uploaded file is empty.")

    records = []
    unparseable_count = 0

    for line_number, line in enumerate(lines, start=1):
        parsed = _parse_line(line)
        if parsed:
            parsed["line_number"] = line_number
            records.append(parsed)
        else:
            unparseable_count += 1

    if not records:
        raise ValueError(
            "No parseable log entries found. Ensure the file contains standard log formats."
        )

    df = pd.DataFrame(records)
    df = _coerce_timestamps(df)

    # Reorder columns for clarity
    df = df[["line_number", "timestamp", "severity", "message"]]
    df["severity"] = pd.Categorical(
        df["severity"],
        categories=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        ordered=True,
    )

    logger.info(
        "Parsed %d log entries. %d lines were unparseable and skipped.",
        len(df),
        unparseable_count,
    )

    return df.reset_index(drop=True)


def get_parse_summary(df: pd.DataFrame) -> dict:
    """
    Return a summary dictionary of a parsed log DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output from parse_log_file.

    Returns
    -------
    dict
        total_lines, severity_counts, time_range_start, time_range_end, null_timestamp_count
    """
    severity_counts = df["severity"].value_counts().to_dict()
    valid_ts = df["timestamp"].dropna()

    return {
        "total_lines": len(df),
        "severity_counts": severity_counts,
        "time_range_start": valid_ts.min() if not valid_ts.empty else None,
        "time_range_end": valid_ts.max() if not valid_ts.empty else None,
        "null_timestamp_count": df["timestamp"].isna().sum(),
    }
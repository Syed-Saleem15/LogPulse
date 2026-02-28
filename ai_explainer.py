# ai_explainer.py
"""
AI Root Cause Explanation Module for LogPulse.
Supports multiple LLM providers: Anthropic, OpenAI, and Google Gemini.
"""

import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

TOP_N_ERRORS = 5

PROVIDER_MODELS = {
    "Anthropic": ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"],
    "OpenAI": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    "Google Gemini": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash-lite"],
}


def get_top_recurring_errors(df: pd.DataFrame, n: int = TOP_N_ERRORS) -> pd.DataFrame:
    """
    Extract the top N most frequently occurring error/critical messages.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed log DataFrame.
    n : int
        Number of top errors to return.

    Returns
    -------
    pd.DataFrame
        Columns: message, severity, count
    """
    error_df = df[df["severity"].isin(["ERROR", "CRITICAL"])].copy()

    if error_df.empty:
        return pd.DataFrame(columns=["message", "severity", "count"])

    top = (
        error_df.groupby(["message", "severity"], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    return top


def _build_prompt(errors: pd.DataFrame) -> str:
    """
    Construct a structured diagnostic prompt for the LLM.

    Parameters
    ----------
    errors : pd.DataFrame

    Returns
    -------
    str
    """
    error_lines = "\n".join(
        f"{i+1}. [{row['severity']}] (x{row['count']}) {row['message']}"
        for i, row in errors.iterrows()
    )

    return f"""You are a senior site reliability engineer analyzing production system logs.

The following are the top recurring error messages extracted from a live infrastructure log file:

{error_lines}

For each error, provide a concise diagnostic report in this exact structure:

**Error N: <truncated message>**
- **Root Cause**: One sentence identifying the most likely technical cause.
- **Impact**: One sentence describing the operational impact.
- **Remediation**: Two to three actionable steps an on-call engineer should take immediately.

Be direct, technical, and avoid generic advice. Assume a Linux-based microservices environment unless the log message implies otherwise.
"""


def _call_anthropic(prompt: str, api_key: str, model: str) -> str:
    """Call Anthropic Claude API."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_openai(prompt: str, api_key: str, model: str) -> str:
    """Call OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _call_gemini(prompt: str, api_key: str, model: str) -> str:
    """Call Google Gemini API."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(prompt)
    return response.text


PROVIDER_CALLERS = {
    "Anthropic": _call_anthropic,
    "OpenAI": _call_openai,
    "Google Gemini": _call_gemini,
}


def explain_errors(
    df: pd.DataFrame,
    provider: str,
    model: str,
    api_key: str,
    n: int = TOP_N_ERRORS,
) -> dict:
    """
    Generate AI-powered root cause explanations for top recurring errors.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed log DataFrame.
    provider : str
        LLM provider name. One of: 'Anthropic', 'OpenAI', 'Google Gemini'.
    model : str
        Model identifier string corresponding to the selected provider.
    api_key : str
        API key for the selected provider.
    n : int
        Number of top errors to analyze.

    Returns
    -------
    dict
        Keys: top_errors (DataFrame), explanation (str), error (str or None)
    """
    top_errors = get_top_recurring_errors(df, n=n)

    if top_errors.empty:
        return {
            "top_errors": top_errors,
            "explanation": None,
            "error": "No ERROR or CRITICAL entries found in the log file.",
        }

    if provider not in PROVIDER_CALLERS:
        return {
            "top_errors": top_errors,
            "explanation": None,
            "error": f"Unsupported provider: {provider}. Choose from {list(PROVIDER_CALLERS.keys())}",
        }

    prompt = _build_prompt(top_errors)

    try:
        caller = PROVIDER_CALLERS[provider]
        explanation = caller(prompt, api_key, model)

        logger.info(
            "AI explanation generated via %s (%s) for %d errors.",
            provider, model, len(top_errors)
        )

        return {
            "top_errors": top_errors,
            "explanation": explanation,
            "error": None,
        }

    except Exception as exc:
        logger.exception("Error during AI explanation via %s.", provider)
        return {
            "top_errors": top_errors,
            "explanation": None,
            "error": f"[{provider}] {str(exc)}",
        }
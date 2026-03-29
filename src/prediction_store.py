"""
Persistence layer for saving and loading past predictions.

Predictions are stored as JSON files in predictions/{SYMBOL}_predictions.json
relative to the project root.
"""

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_VALID_SIGNALS = {"STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"}
_VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}

PREDICTIONS_DIR = Path(__file__).parent.parent / "predictions"


def _ensure_dir() -> None:
    PREDICTIONS_DIR.mkdir(exist_ok=True)


def parse_signal_from_report(report_text: str) -> tuple[str, str]:
    """
    Extract the overall signal and confidence from a prediction report.

    Primary: looks for the machine-readable sentinel line emitted by the
    prediction task as its very last line:
        PREDICTION_SIGNAL_JSON:{"signal":"BUY","confidence":"HIGH"}

    Fallback (for old reports or if the LLM omitted the sentinel): scans the
    text for the structured "OVERALL SIGNAL:" heading, then as a last resort
    picks the final standalone BUY/SELL/NEUTRAL keyword in the document.
    """
    # ── Primary: structured JSON sentinel ─────────────────────────────────────
    sentinel = re.search(
        r"PREDICTION_SIGNAL_JSON:\s*(\{[^}]+\})",
        report_text,
    )
    if sentinel:
        try:
            obj = json.loads(sentinel.group(1))
            signal     = str(obj.get("signal", "")).upper()
            confidence = str(obj.get("confidence", "")).upper()
            if signal in _VALID_SIGNALS and confidence in _VALID_CONFIDENCE:
                return signal, confidence
            logger.warning(
                "PREDICTION_SIGNAL_JSON present but values invalid: %s", obj
            )
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse PREDICTION_SIGNAL_JSON: %s", exc)

    # ── Fallback 1: "OVERALL SIGNAL: BUY" heading ─────────────────────────────
    heading_match = re.search(
        r"overall[\s_]signal[:\s*|]+\s*(STRONG_BUY|STRONG_SELL|BUY|SELL|NEUTRAL)",
        report_text,
        re.IGNORECASE,
    )
    signal = heading_match.group(1).upper() if heading_match else None

    # ── Fallback 2: last standalone keyword ───────────────────────────────────
    if not signal:
        keywords = re.findall(r"\b(STRONG_BUY|STRONG_SELL|BUY|SELL|NEUTRAL)\b", report_text)
        signal = keywords[-1].upper() if keywords else "UNKNOWN"
        if signal != "UNKNOWN":
            logger.warning(
                "Signal extracted via last-keyword fallback ('%s') — "
                "PREDICTION_SIGNAL_JSON line was missing from the report.",
                signal,
            )

    confidence_match = re.search(
        r"confidence[:\s*|]+\s*(HIGH|MEDIUM|LOW)",
        report_text,
        re.IGNORECASE,
    )
    confidence = confidence_match.group(1).upper() if confidence_match else "UNKNOWN"

    return signal, confidence


def save_prediction(symbol: str, report_text: str, price_at_prediction: float | None = None) -> str:
    """
    Save a completed prediction report to persistent storage.

    Args:
        symbol: Crypto symbol (e.g. 'BTC')
        report_text: The full prediction report string
        price_at_prediction: Current price at the time of prediction (optional)

    Returns:
        The prediction ID (short UUID)
    """
    _ensure_dir()

    signal, confidence = parse_signal_from_report(report_text)
    prediction_id = str(uuid.uuid4())[:8]

    prediction = {
        "id": prediction_id,
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": symbol.upper(),
        "signal": signal,
        "confidence": confidence,
        "price_at_prediction": price_at_prediction,
        "report": report_text,
    }

    filepath = PREDICTIONS_DIR / f"{symbol.upper()}_predictions.json"

    if filepath.exists():
        with open(filepath, "r") as f:
            predictions = json.load(f)
    else:
        predictions = []

    predictions.append(prediction)

    with open(filepath, "w") as f:
        json.dump(predictions, f, indent=2)

    return prediction_id


def load_predictions(symbol: str, limit: int = 10) -> list[dict]:
    """
    Load the most recent predictions for a symbol, newest first.

    Args:
        symbol: Crypto symbol (e.g. 'BTC')
        limit: Maximum number of predictions to return

    Returns:
        List of prediction dicts, sorted newest first.
    """
    _ensure_dir()
    filepath = PREDICTIONS_DIR / f"{symbol.upper()}_predictions.json"

    if not filepath.exists():
        return []

    with open(filepath, "r") as f:
        predictions = json.load(f)

    # Return most recent first
    return predictions[-limit:][::-1]

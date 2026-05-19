"""
Structured logging — Loguru + JSON sink.
Every module should do:  from backend.utils.logger import get_logger; log = get_logger(__name__)
"""

import sys
import json
from pathlib import Path
from loguru import logger

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)

TRACE_FILE = LOG_DIR / "rag_trace.jsonl"

# ─── Remove default handler ───────────────────────────────────────────────────
logger.remove()

# ─── Pretty console (human-readable) ─────────────────────────────────────────
logger.add(
    sys.stderr,
    level="DEBUG",
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# ─── JSONL file sink (machine-readable trace) ─────────────────────────────────
def _json_sink(message):
    record = message.record
    entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
        "extra": record["extra"],
    }
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


logger.add(_json_sink, level="DEBUG", enqueue=True)


def get_logger(name: str):
    """Return a logger bound with the given module name."""
    return logger.bind(module=name)

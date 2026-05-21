"""
Structured logging — Loguru + JSON sink.
Every module should do:  from backend.utils.logger import get_logger; log = get_logger(__name__)
"""

import sys
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

# ─── JSONL file sink (machine-readable trace, with rotation) ──────────────────
logger.add(
    str(TRACE_FILE),
    level="DEBUG",
    format="{message}",
    serialize=True,
    rotation="50 MB",
    retention=3,
    enqueue=True,
)


def get_logger(name: str):
    """Return a logger bound with the given module name."""
    return logger.bind(module=name)

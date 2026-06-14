"""
Structured JSON logger for the Spotify Recommender API.

Every log entry is a single JSON line — machine-parseable and ready
to be forwarded to any log aggregator (CloudWatch, ELK, Datadog, etc.).

Example log line:
  {"timestamp": "2026-06-09T14:32:01Z", "level": "INFO",
   "event": "recommendation_served", "song": "no surprises",
   "artist": "radiohead", "method": "hybrid", "k": 10,
   "duration_ms": 38.4, "results_count": 10}
"""
import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path


class _JSONFormatter(logging.Formatter):
    """Formats every log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Merge any extra fields passed via logger.info("...", extra={...})
        for key, value in record.__dict__.items():
            if key not in (
                "timestamp", "level", "message",
                "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "name",
                "taskName",
            ):
                entry[key] = value

        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


def get_logger(name: str = "recommender_api", log_dir: Path = Path("logs")) -> logging.Logger:
    """
    Returns a logger that writes structured JSON to both the console
    and a rotating log file (logs/app.log, max 10 MB, 5 backups).
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger() is called more than once
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = _JSONFormatter()

    # Console handler — visible in the terminal while running uvicorn
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler — persists logs across restarts
    log_dir.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,               # keep last 5 rotated files
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Don't propagate to the root logger (avoids duplicate lines)
    logger.propagate = False

    return logger

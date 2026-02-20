"""
Logger - Utilities Module
Structured logging + SQLite-backed security event persistence.
Cost: $0 — stdlib only.
"""

import logging
import logging.handlers
import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import colorlog


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a colourised logger for a module."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler with colour
    console = colorlog.StreamHandler()
    console.setLevel(level)
    console.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)-8s]%(reset)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "red,bg_white",
        }
    ))
    logger.addHandler(console)

    # File handler (rotating)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "bharatsecure.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    ))
    logger.addHandler(file_handler)

    return logger


class SecurityEventLogger:
    """
    SQLite-backed security event store for the dashboard.
    Stores: event_type, detail, timestamp.
    Cost: $0 — SQLite is stdlib.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    detail     TEXT,
                    timestamp  TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON security_events(timestamp)")
            conn.commit()

    def log(self, event_type: str, detail: str = ""):
        """Insert a security event."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO security_events (event_type, detail, timestamp) VALUES (?, ?, ?)",
                    (event_type, detail, datetime.utcnow().isoformat())
                )
                conn.commit()
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to log event: {e}")

    def get_recent(self, limit: int = 100) -> List[Dict]:
        """Retrieve most recent events."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM security_events ORDER BY id DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [dict(row) for row in rows]
        except Exception:
            return []

    def get_stats(self) -> Dict:
        """Aggregate event counts by type."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT event_type, COUNT(*) as cnt FROM security_events GROUP BY event_type"
                ).fetchall()
                stats = {row[0]: row[1] for row in rows}
                stats["total"] = sum(stats.values())
                return stats
        except Exception:
            return {"total": 0}

    def clear(self):
        """Delete all events."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM security_events")
            conn.commit()

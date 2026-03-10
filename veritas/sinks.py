"""Output destinations for cost events."""

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from veritas.core import CostEvent

# Schema for the events table. Matches CostEvent fields for change detection queries.
EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature TEXT NOT NULL,
    model TEXT NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL,
    cost_usd REAL NOT NULL,
    code_version TEXT,
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ok',
    estimated INTEGER NOT NULL DEFAULT 0
);
"""


class BaseSink(ABC):
    """Abstract sink for emitting cost events."""

    @abstractmethod
    def emit(self, event: "CostEvent") -> None:
        """Emit a cost event to the sink's destination."""
        pass


class ConsoleSink(BaseSink):
    """Sink that prints cost events as JSON lines to stdout."""

    def emit(self, event: "CostEvent") -> None:
        """Print the event as a single JSON line to stdout."""
        data = event.to_dict()
        print(json.dumps(data))


class SQLiteSink(BaseSink):
    """Sink that persists cost events to a local SQLite database."""

    def __init__(self, path: str | Path | None = None):
        """Initialize the sink with a DB path.

        Args:
            path: Path to the SQLite file. Use ":memory:" for an in-memory DB
                  (useful for tests). If None, defaults to VERITAS_DB_PATH env var
                  or 'veritas_events.db'.
        """
        if path is None:
            import os
            self._path = os.environ.get("VERITAS_DB_PATH", "veritas_events.db")
        else:
            self._path = str(path)
            
        self._conn = sqlite3.connect(self._path)
        self._conn.execute(EVENTS_SCHEMA)
        self._conn.commit()

    def emit(self, event: "CostEvent") -> None:
        """Insert the event as a row in the events table."""
        data = event.to_dict()
        # SQLite uses 0/1 for booleans
        data["estimated"] = 1 if data["estimated"] else 0

        self._conn.execute(
            """
            INSERT INTO events (
                feature, model, tokens_in, tokens_out, cache_creation_tokens, cache_read_tokens, latency_ms,
                cost_usd, code_version, timestamp, status, estimated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["feature"],
                data["model"],
                data["tokens_in"],
                data["tokens_out"],
                data["cache_creation_tokens"],
                data["cache_read_tokens"],
                data["latency_ms"],
                data["cost_usd"],
                data["code_version"],
                data["timestamp"],
                data["status"],
                data["estimated"],
            ),
        )
        self._conn.commit()

    def get_events(
        self, feature: str, commit: str | None = None, since_iso: str | None = None
    ) -> list[dict]:
        # TODO: Extract this into a dedicated Read/Storage Model later
        """Fetch events for a feature. Allows filtering by commit hash or ISO timestamp."""
        query = "SELECT * FROM events WHERE feature = ?"
        params = [feature]

        if commit:
            query += " AND code_version = ?"
            params.append(commit)
            
        if since_iso:
            query += " AND timestamp >= ?"
            params.append(since_iso)
            
        query += " ORDER BY timestamp ASC"

        # Return rows as dicts
        self._conn.row_factory = sqlite3.Row
        cursor = self._conn.execute(query, tuple(params))
        rows = cursor.fetchall()
        
        # Reset row_factory just to be safe
        self._conn.row_factory = None
        
        return [dict(row) for row in rows]

    def close(self) -> None:
        """Close the database connection. Call when done emitting events."""
        self._conn.close()


class HttpSink(BaseSink):
    """Sink that transmits cost events via HTTP POST to a centralized Veritas server."""
    
    def __init__(self, endpoint_url: str, api_key: str):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        # We use a memory buffer to batch events in real production,
        # but for this demo phase we dispatch synchronously to prove flow.
        import requests
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
    def emit(self, event: "CostEvent") -> None:
        """Send the event directly to the centralized server."""
        try:
            response = self._session.post(
                self.endpoint_url, 
                json=event.to_dict(), 
                timeout=2.0
            )
        except Exception:
            # Fail silently — never crash the host application due to telemetry errors.
            pass

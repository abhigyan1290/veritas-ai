"""Output destinations for cost events."""

import json
import logging
import queue
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

try:
    import requests as requests
except ImportError:
    requests = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from veritas.core import CostEvent

logger = logging.getLogger("veritas")

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
    estimated INTEGER NOT NULL DEFAULT 0,
    tags TEXT
);
"""

# Index on code_version speeds up compare_commits / dashboard GROUP BY queries.
EVENTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_events_code_version ON events(code_version);
"""



class BaseSink(ABC):
    """Abstract sink for emitting cost events."""

    def _check_version(self, event: "CostEvent") -> None:
        """Warn if code_version is 'unknown' — events won't be attributable to a deployment."""
        if event.code_version == "unknown":
            logger.warning(
                "veritas: emitting event for feature %r with code_version='unknown'. "
                "Events cannot be attributed to a specific deployment. "
                "Set the VERITAS_CODE_VERSION environment variable to the deployed commit SHA, "
                "or pass code_version to veritas.init().",
                event.feature,
            )

    @abstractmethod
    def emit(self, event: "CostEvent") -> None:
        """Emit a cost event to the sink's destination."""
        pass


class ConsoleSink(BaseSink):
    """Sink that prints cost events as JSON lines to stdout."""

    def emit(self, event: "CostEvent") -> None:
        """Print the event as a single JSON line to stdout."""
        self._check_version(event)
        data = event.to_dict()
        print(json.dumps(data))


class SQLiteSink(BaseSink):
    """Sink that persists cost events to a local SQLite database."""

    BATCH_SIZE = 25  # commit every N inserts to avoid per-write fsync overhead

    def __init__(self, path: Optional[Union[str, Path]] = None):
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

        self._conn = sqlite3.connect(self._path, check_same_thread=False)

        # WAL mode allows readers and the writer to coexist without blocking
        # each other, and makes individual writes significantly faster because
        # SQLite no longer needs to fsync the main database file on every commit.
        if self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
            # NORMAL is safe (no data loss on OS crash; only on power loss) and
            # much faster than the default FULL fsync.
            self._conn.execute("PRAGMA synchronous=NORMAL")

        self._conn.execute(EVENTS_SCHEMA)
        self._conn.execute(EVENTS_INDEX)
        self._conn.commit()
        self._pending = 0  # count of uncommitted inserts

    def emit(self, event: "CostEvent") -> None:
        """Insert the event as a row in the events table.

        Commits are batched every BATCH_SIZE events to avoid an fsync on every
        single insert. The last batch is committed when close() is called.
        """
        self._check_version(event)
        data = event.to_dict()
        # SQLite uses 0/1 for booleans
        data["estimated"] = 1 if data["estimated"] else 0

        self._conn.execute(
            """
            INSERT INTO events (
                feature, model, tokens_in, tokens_out, cache_creation_tokens, cache_read_tokens, latency_ms,
                cost_usd, code_version, timestamp, status, estimated, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(data.get("tags", {})),
            ),
        )
        self._pending += 1
        if self._pending >= self.BATCH_SIZE:
            self._conn.commit()
            self._pending = 0

    def get_events(
        self, feature: str, commit: Optional[str] = None, since_iso: Optional[str] = None
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
        """Flush any pending batch and close the database connection."""
        if self._pending > 0:
            self._conn.commit()
            self._pending = 0
        self._conn.close()


class HttpSink(BaseSink):
    """Sink that transmits cost events via HTTP POST to a centralized Veritas server.

    emit() is non-blocking: events are placed on an in-process queue and sent
    in the background by a daemon thread, so the host application never waits
    on network I/O. Events are sent individually (matching the existing server
    endpoint) with a short timeout. If the queue is full or the request fails
    the event is silently dropped — telemetry must never crash the host app.
    """

    # Maximum events to hold in memory before dropping new ones.
    QUEUE_MAXSIZE = 1000

    def __init__(self, endpoint_url: str, api_key: str):
        self.endpoint_url = endpoint_url
        self.api_key = api_key

        if requests is None:
            raise ImportError(
                "veritas: 'requests' is required for HttpSink. Install it with: pip install requests"
            )
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

        self._queue: queue.Queue = queue.Queue(maxsize=self.QUEUE_MAXSIZE)

        # Daemon thread so it never prevents the process from exiting.
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()

    def emit(self, event: "CostEvent") -> None:
        """Enqueue the event for background dispatch. Never blocks the caller."""
        self._check_version(event)
        try:
            self._queue.put_nowait(event.to_dict())
        except queue.Full:
            # Queue is full — drop the event rather than blocking the host app,
            # but log so operators know attribution data is being lost.
            logger.warning(
                "veritas: HttpSink queue full (%d/%d) — dropping event for feature %r. "
                "Consider increasing QUEUE_MAXSIZE or reducing emit frequency.",
                self._queue.qsize(),
                self.QUEUE_MAXSIZE,
                event.feature,
            )

    def _flush_loop(self) -> None:
        """Background thread: drain the queue and POST events to the server."""
        while True:
            try:
                payload = self._queue.get()  # blocks until an event is available
                self._session.post(
                    self.endpoint_url,
                    json=payload,
                    timeout=5.0,
                )
            except Exception:
                # Fail silently — never crash the host application.
                pass

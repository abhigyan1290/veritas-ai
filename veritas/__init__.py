"""Veritas — AI cost attribution and change detection."""

__version__ = "0.1.0"

from veritas.core import CostEvent, track, set_default_sink, get_default_sink
from veritas.sinks import BaseSink, ConsoleSink, SQLiteSink, HttpSink
from veritas.utils import get_current_commit_hash, utc_now_iso
from veritas.client import Anthropic
from veritas.openai_client import OpenAI

__all__ = [
    "__version__",
    "init",
    "CostEvent",
    "track",
    "set_default_sink",
    "get_default_sink",
    "BaseSink",
    "ConsoleSink",
    "SQLiteSink",
    "HttpSink",
    "get_current_commit_hash",
    "utc_now_iso",
    "Anthropic",
    "OpenAI",
]


def init(api_key: str, endpoint: str) -> None:
    """Configure Veritas to send events to your hosted server.

    Call this once at application startup before wrapping any clients.

    Args:
        api_key:  Your project's Veritas API key (starts with sk-vrt-).
        endpoint: Full URL to your server's event ingestion endpoint,
                  e.g. "https://your-server.com/api/v1/events"

    Example:
        import veritas
        veritas.init(
            api_key="sk-vrt-...",
            endpoint="https://your-server.com/api/v1/events",
        )
    """
    set_default_sink(HttpSink(endpoint_url=endpoint, api_key=api_key))


# Auto-configure from environment variables if present (alternative to calling init())
import os as _os
if _os.environ.get("VERITAS_API_KEY") and _os.environ.get("VERITAS_API_URL"):
    set_default_sink(HttpSink(_os.environ["VERITAS_API_URL"], _os.environ["VERITAS_API_KEY"]))
elif _os.environ.get("VERITAS_DB_PATH"):
    set_default_sink(SQLiteSink(_os.environ["VERITAS_DB_PATH"]))

"""Core tracking decorator and event handling."""

import inspect
import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional, TypeVar

from veritas.pricing import compute_cost
from veritas.sinks import BaseSink, ConsoleSink
from veritas.utils import get_current_commit_hash, utc_now_iso

F = TypeVar("F")


@dataclass
class CostEvent:
    """Structured record of a single AI API call for cost attribution."""

    feature: str
    model: str
    tokens_in: int
    tokens_out: int
    cache_creation_tokens: int
    cache_read_tokens: int
    latency_ms: float
    cost_usd: float
    code_version: Optional[str]
    timestamp: str
    status: str = "ok"  # "ok" | "error"
    estimated: bool = False  # True when usage/cost was inferred (e.g. missing from API)

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict for sinks."""
        return {
            "feature": self.feature,
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "code_version": self.code_version,
            "timestamp": self.timestamp,
            "status": self.status,
            "estimated": self.estimated,
        }


# Default sink for tracking. ConsoleSink prints to stdout.
_default_sink: BaseSink = ConsoleSink()


def _extract_usage(response) -> tuple[str, int, int, int, int]:
    """Extract (model, tokens_in, tokens_out, cache_creation_tokens, cache_read_tokens) from a Claude API response.

    Supports Anthropic SDK response objects (with .usage, .model) and dict-like
    structures. Returns ("unknown", 0, 0, 0, 0) when extraction fails.
    """
    model = "unknown"
    tokens_in = 0
    tokens_out = 0
    cache_creation_tokens = 0
    cache_read_tokens = 0

    # Attribute access (Anthropic SDK Message object)
    if hasattr(response, "model") and response.model:
        model = str(response.model)
    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        tokens_in = getattr(usage, "input_tokens", 0) or 0
        tokens_out = getattr(usage, "output_tokens", 0) or 0
        cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

    # Dict-style fallback (e.g. OpenAI-style)
    if tokens_in == 0 and tokens_out == 0:
        if isinstance(response, dict):
            model = response.get("model", model)
            usage = response.get("usage") or {}
            tokens_in = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
            tokens_out = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
            cache_creation_tokens = usage.get("cache_creation_input_tokens", 0) or 0
            cache_read_tokens = usage.get("cache_read_input_tokens", 0) or 0

    return (str(model), int(tokens_in), int(tokens_out), int(cache_creation_tokens), int(cache_read_tokens))


def track(
    feature: str,
    sink: Optional[BaseSink] = None,
) -> Callable[[F], F]:
    """Decorator to track AI API calls and emit cost events.

    Args:
        feature: Name of the feature (e.g. "chat_search", "doc_summary").
        sink: Where to emit events. Defaults to ConsoleSink.

    Example:
        @track(feature="chat_search")
        def call_claude(prompt: str):
            return anthropic.messages.create(...)
    """

    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                used_sink = sink if sink is not None else _default_sink
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    latency_ms = (time.perf_counter() - start) * 1000

                    model, tokens_in, tokens_out, cache_creation_tokens, cache_read_tokens = _extract_usage(result)
                    cost_usd, estimated = compute_cost(
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        model=model,
                        cache_creation_tokens=cache_creation_tokens,
                        cache_read_tokens=cache_read_tokens,
                    )
                    event = CostEvent(
                        feature=feature,
                        model=model,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        cache_creation_tokens=cache_creation_tokens,
                        cache_read_tokens=cache_read_tokens,
                        latency_ms=round(latency_ms, 2),
                        cost_usd=cost_usd,
                        code_version=get_current_commit_hash(),
                        timestamp=utc_now_iso(),
                        status="ok",
                        estimated=estimated,
                    )
                    used_sink.emit(event)
                    return result

                except Exception:
                    latency_ms = (time.perf_counter() - start) * 1000
                    event = CostEvent(
                        feature=feature,
                        model="unknown",
                        tokens_in=0,
                        tokens_out=0,
                        cache_creation_tokens=0,
                        cache_read_tokens=0,
                        latency_ms=round(latency_ms, 2),
                        cost_usd=0.0,
                        code_version=get_current_commit_hash(),
                        timestamp=utc_now_iso(),
                        status="error",
                        estimated=True,
                    )
                    used_sink.emit(event)
                    raise

            return async_wrapper  # type: ignore

        @wraps(func)
        def wrapper(*args, **kwargs):
            used_sink = sink if sink is not None else _default_sink
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                latency_ms = (time.perf_counter() - start) * 1000

                model, tokens_in, tokens_out, cache_creation_tokens, cache_read_tokens = _extract_usage(result)
                cost_usd, estimated = compute_cost(
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    model=model,
                    cache_creation_tokens=cache_creation_tokens,
                    cache_read_tokens=cache_read_tokens,
                )
                event = CostEvent(
                    feature=feature,
                    model=model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cache_creation_tokens=cache_creation_tokens,
                    cache_read_tokens=cache_read_tokens,
                    latency_ms=round(latency_ms, 2),
                    cost_usd=cost_usd,
                    code_version=get_current_commit_hash(),
                    timestamp=utc_now_iso(),
                    status="ok",
                    estimated=estimated,
                )
                used_sink.emit(event)
                return result

            except Exception:
                latency_ms = (time.perf_counter() - start) * 1000
                event = CostEvent(
                    feature=feature,
                    model="unknown",
                    tokens_in=0,
                    tokens_out=0,
                    cache_creation_tokens=0,
                    cache_read_tokens=0,
                    latency_ms=round(latency_ms, 2),
                    cost_usd=0.0,
                    code_version=get_current_commit_hash(),
                    timestamp=utc_now_iso(),
                    status="error",
                    estimated=True,
                )
                used_sink.emit(event)
                raise

        return wrapper  # type: ignore

    return decorator


def set_default_sink(sink: BaseSink) -> None:
    """Set the default sink for @track when no sink is passed."""
    global _default_sink
    _default_sink = sink

def get_default_sink() -> BaseSink:
    """Return the currently configured sink."""
    return _default_sink

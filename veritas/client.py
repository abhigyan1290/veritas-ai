"""
veritas/client.py — Proxy wrappers for Anthropic and OpenAI clients.

These classes are drop-in replacements that intercept every API call,
capture cost/latency/token metrics, and emit a CostEvent — silently.
The host application's code never changes.
"""
import asyncio
import time
from datetime import datetime, timezone

from veritas.core import CostEvent
from veritas.pricing import compute_cost


# ─────────────────────────────────────────────────────────────────────────────
# Shared Utility: emit a CostEvent to the configured sink
# ─────────────────────────────────────────────────────────────────────────────

def _emit_event(
    feature_name: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: float,
    commit_hash: str,
    cache_creation: int = 0,
    cache_read: int = 0,
    status: str = "ok",
):
    """Compute cost and emit a CostEvent to the configured sink. Never raises."""
    try:
        cost, estimated = compute_cost(
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
        )
        event = CostEvent(
            feature=feature_name,
            model=model,
            code_version=commit_hash,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            latency_ms=latency_ms,
            cost_usd=cost,
            estimated=estimated,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
        )
        from veritas.core import get_default_sink
        sink = get_default_sink()
        if sink:
            sink.emit(event)
    except Exception:
        # Veritas must NEVER crash the host application.
        pass


def _get_commit() -> str:
    """Resolve the current git commit hash safely.

    Delegates to ``get_current_commit_hash()`` which handles overrides,
    env vars, caching, fast-path resolution, and dirty detection.
    """
    try:
        from veritas.utils import get_current_commit_hash
        return get_current_commit_hash()
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic Proxy
# ─────────────────────────────────────────────────────────────────────────────

class Anthropic:
    """
    Veritas Proxy for the Anthropic Python SDK.

    Drop-in replacement for `anthropic.Anthropic` or `anthropic.AsyncAnthropic`.
    Intercepts `messages.create()` (both streaming and non-streaming).
    Supports sync, async, and stream=True transparently.

    Usage:
        client = veritas.Anthropic(anthropic.AsyncAnthropic(), feature_name="my_feature")
        # Then use client.messages.create(...) exactly as before.
    """
    def __init__(self, client, feature_name: str = "default_feature"):
        import anthropic as _anthropic
        self._client = client
        self._feature_name = feature_name
        is_async = isinstance(client, _anthropic.AsyncAnthropic)
        self.messages = _AnthropicMessagesProxy(client.messages, feature_name, is_async)

    def __getattr__(self, name):
        return getattr(self._client, name)


class _AnthropicMessagesProxy:
    def __init__(self, original_messages, feature_name: str, is_async: bool):
        self._original_messages = original_messages
        self._feature_name = feature_name
        self._is_async = is_async

    def create(self, *args, **kwargs):
        start_time = time.time()
        if self._is_async:
            return self._async_create(start_time, *args, **kwargs)
        else:
            return self._sync_create(start_time, *args, **kwargs)

    # ── Sync path ────────────────────────────────────────────────────────────

    def _sync_create(self, start_time: float, *args, **kwargs):
        is_streaming = kwargs.get("stream", False)
        commit = _get_commit()

        if is_streaming:
            raw_stream = self._original_messages.create(*args, **kwargs)
            return _AnthropicSyncStream(raw_stream, self._feature_name, kwargs.get("model", "unknown"), start_time, commit)

        response = self._original_messages.create(*args, **kwargs)
        latency_ms = (time.time() - start_time) * 1000
        self._track_from_response(response, kwargs.get("model", "unknown"), latency_ms, commit)
        return response

    # ── Async path ───────────────────────────────────────────────────────────

    async def _async_create(self, start_time: float, *args, **kwargs):
        commit = _get_commit()
        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            raw_stream = await self._original_messages.create(*args, **kwargs)
            return _AnthropicAsyncStream(raw_stream, self._feature_name, kwargs.get("model", "unknown"), start_time, commit)

        response = await self._original_messages.create(*args, **kwargs)
        latency_ms = (time.time() - start_time) * 1000
        # Push the blocking HTTP emit off the event loop
        await asyncio.to_thread(self._track_from_response, response, kwargs.get("model", "unknown"), latency_ms, commit)
        return response

    def _track_from_response(self, response, model: str, latency_ms: float, commit: str):
        usage = getattr(response, "usage", None)
        if not usage:
            return
        _emit_event(
            feature_name=self._feature_name,
            model=model,
            tokens_in=getattr(usage, "input_tokens", 0),
            tokens_out=getattr(usage, "output_tokens", 0),
            latency_ms=latency_ms,
            commit_hash=commit,
            cache_creation=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )


class _AnthropicSyncStream:
    """
    Wraps a sync Anthropic stream (RawMessageStreamEvent iterator).
    Passes all events through unchanged; emits a CostEvent when the stream closes.
    """
    def __init__(self, stream, feature_name, model, start_time, commit):
        self._stream = stream
        self._feature_name = feature_name
        self._model = model
        self._start_time = start_time
        self._commit = commit
        self._tokens_in = 0
        self._tokens_out = 0

    def __iter__(self):
        status = "ok"
        try:
            for event in self._stream:
                # Capture token counts from stream events
                etype = getattr(event, "type", None)
                if etype == "message_start":
                    usage = getattr(getattr(event, "message", None), "usage", None)
                    if usage:
                        self._tokens_in = getattr(usage, "input_tokens", 0)
                elif etype == "message_delta":
                    usage = getattr(event, "usage", None)
                    if usage:
                        self._tokens_out = getattr(usage, "output_tokens", 0)
                yield event
        except Exception:
            status = "error"
            raise
        finally:
            # Always emit — covers normal exhaustion, exceptions, and cancellation.
            _emit_event(
                feature_name=self._feature_name,
                model=self._model,
                tokens_in=self._tokens_in,
                tokens_out=self._tokens_out,
                latency_ms=(time.time() - self._start_time) * 1000,
                commit_hash=self._commit,
                status=status,
            )

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _AnthropicAsyncStream:
    """
    Wraps an async Anthropic stream. Same logic as sync but with __aiter__.
    """
    def __init__(self, stream, feature_name, model, start_time, commit):
        self._stream = stream
        self._feature_name = feature_name
        self._model = model
        self._start_time = start_time
        self._commit = commit

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        tokens_in = 0
        tokens_out = 0
        status = "ok"
        try:
            async for event in self._stream:
                etype = getattr(event, "type", None)
                if etype == "message_start":
                    usage = getattr(getattr(event, "message", None), "usage", None)
                    if usage:
                        tokens_in = getattr(usage, "input_tokens", 0)
                elif etype == "message_delta":
                    usage = getattr(event, "usage", None)
                    if usage:
                        tokens_out = getattr(usage, "output_tokens", 0)
                yield event
        except Exception:
            status = "error"
            raise
        finally:
            # Always emit — covers normal exhaustion, exceptions, and cancellation.
            _emit_event(
                feature_name=self._feature_name,
                model=self._model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=(time.time() - self._start_time) * 1000,
                commit_hash=self._commit,
                status=status,
            )

    def __getattr__(self, name):
        return getattr(self._stream, name)

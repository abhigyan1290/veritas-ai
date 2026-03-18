"""
veritas/openai_client.py — OpenAI proxy wrapper.

Drop-in replacement for openai.OpenAI / openai.AsyncOpenAI.
Intercepts chat.completions.create() (both streaming and non-streaming),
captures token usage, computes cost, and emits a CostEvent silently.

Data flow per call mode
──────────────────────────────────────────────────────────────────────
Non-streaming (sync/async):
  client.chat.completions.create(...)
      → API returns ChatCompletion
      → response.usage.prompt_tokens    (input tokens)
      → response.usage.completion_tokens (output tokens)
      → _emit_event() called immediately after response

Streaming (sync):
  client.chat.completions.create(..., stream=True)
      → veritas auto-injects stream_options={"include_usage": True}
      → returns _OpenAISyncStream (wraps the original generator)
      → caller iterates chunks normally
      → LAST chunk: chunk.usage.prompt_tokens / completion_tokens
      → _emit_event() called when iteration exhausts

Streaming (async):
  await client.chat.completions.create(..., stream=True)
      → veritas auto-injects stream_options={"include_usage": True}
      → returns _OpenAIAsyncStream (wraps the original async generator)
      → caller async-iterates chunks normally
      → LAST chunk: chunk.usage.prompt_tokens / completion_tokens
      → _emit_event() called when iteration exhausts
──────────────────────────────────────────────────────────────────────

Note: stream_options is injected automatically and transparently.
      The caller's code does not change at all.
"""
import asyncio
import time

from veritas.client import _emit_event, _get_commit


class OpenAI:
    """
    Veritas Proxy for the OpenAI Python SDK.

    Drop-in replacement for `openai.OpenAI` or `openai.AsyncOpenAI`.
    Intercepts `chat.completions.create()` (streaming and non-streaming).

    Usage:
        # Sync
        client = veritas.OpenAI(openai.OpenAI(), feature_name="my_feature")

        # Async
        client = veritas.OpenAI(openai.AsyncOpenAI(), feature_name="my_feature")

        # Then use client.chat.completions.create(...) exactly as before.
        # Streaming works too — just pass stream=True as normal.
    """
    def __init__(self, client, feature_name: str = "default_feature"):
        import openai as _openai
        self._client = client
        self._feature_name = feature_name
        is_async = isinstance(client, _openai.AsyncOpenAI)
        self.chat = _ChatProxy(client.chat, feature_name, is_async)

    def __getattr__(self, name):
        # Forward .models, .files, .embeddings, etc. directly.
        return getattr(self._client, name)


class _ChatProxy:
    """Mirrors client.chat so that client.chat.completions.create() is interceptable."""

    def __init__(self, original_chat, feature_name: str, is_async: bool):
        self._original_chat = original_chat
        self.completions = _CompletionsProxy(original_chat.completions, feature_name, is_async)

    def __getattr__(self, name):
        return getattr(self._original_chat, name)


class _CompletionsProxy:
    """Intercepts chat.completions.create()."""

    def __init__(self, original_completions, feature_name: str, is_async: bool):
        self._original_completions = original_completions
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
        model = kwargs.get("model", "unknown")
        commit = _get_commit()

        if is_streaming:
            # Inject stream_options so the final chunk includes usage.
            # We use setdefault to avoid overwriting if users set their own options.
            kwargs.setdefault("stream_options", {"include_usage": True})
            raw = self._original_completions.create(*args, **kwargs)
            return _OpenAISyncStream(raw, self._feature_name, model, start_time, commit)

        response = self._original_completions.create(*args, **kwargs)
        latency_ms = (time.time() - start_time) * 1000
        _track_from_completion(response, self._feature_name, model, latency_ms, commit)
        return response

    # ── Async path ───────────────────────────────────────────────────────────

    async def _async_create(self, start_time: float, *args, **kwargs):
        is_streaming = kwargs.get("stream", False)
        model = kwargs.get("model", "unknown")
        commit = _get_commit()

        if is_streaming:
            kwargs.setdefault("stream_options", {"include_usage": True})
            raw = await self._original_completions.create(*args, **kwargs)
            return _OpenAIAsyncStream(raw, self._feature_name, model, start_time, commit)

        response = await self._original_completions.create(*args, **kwargs)
        latency_ms = (time.time() - start_time) * 1000
        await asyncio.to_thread(_track_from_completion, response, self._feature_name, model, latency_ms, commit)
        return response


# ─────────────────────────────────────────────────────────────────────────────
# Stream Wrappers
# ─────────────────────────────────────────────────────────────────────────────

class _OpenAISyncStream:
    """
    Transparent wrapper around an OpenAI sync stream.
    Passes every chunk through unchanged; emits CostEvent when the stream ends.

    The final chunk (injected by stream_options) has chunk.usage populated
    with prompt_tokens and completion_tokens. All other chunks have usage=None.
    """
    def __init__(self, stream, feature_name, model, start_time, commit):
        self._stream = stream
        self._feature_name = feature_name
        self._model = model
        self._start_time = start_time
        self._commit = commit

    def __iter__(self):
        tokens_in = 0
        tokens_out = 0
        status = "ok"
        try:
            for chunk in self._stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    tokens_in = getattr(usage, "prompt_tokens", 0) or 0
                    tokens_out = getattr(usage, "completion_tokens", 0) or 0
                yield chunk
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


class _OpenAIAsyncStream:
    """
    Transparent wrapper around an OpenAI async stream.
    Same logic as the sync version but supports `async for`.
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
            async for chunk in self._stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    tokens_in = getattr(usage, "prompt_tokens", 0) or 0
                    tokens_out = getattr(usage, "completion_tokens", 0) or 0
                yield chunk
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


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _track_from_completion(response, feature_name, model, latency_ms, commit):
    """Extract token usage from a non-streaming ChatCompletion and emit."""
    usage = getattr(response, "usage", None)
    if not usage:
        return
    _emit_event(
        feature_name=feature_name,
        model=model,
        tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
        tokens_out=getattr(usage, "completion_tokens", 0) or 0,
        latency_ms=latency_ms,
        commit_hash=commit,
    )

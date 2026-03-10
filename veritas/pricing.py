"""Token cost estimation and pricing tables."""

from typing import Tuple

# Pricing per 1M tokens (input, output) in USD.
# Sources:
#   Anthropic: https://docs.anthropic.com/en/docs/about-claude/pricing
#   OpenAI: https://openai.com/api/pricing
# Keys match API model identifiers; versioned names resolve via longest-prefix match.
PRICING_TABLE: dict[str, Tuple[float, float, float, float]] = {
    # (input_per_1m, output_per_1m, cache_write_per_1m, cache_read_per_1m)

    # ── Anthropic Claude 4.x ──────────────────────────────────────────────
    "claude-opus-4":   (5.00,  25.00,  6.25, 0.50),
    "claude-sonnet-4": (3.00,  15.00,  3.75, 0.30),
    "claude-haiku-4":  (1.00,   5.00,  1.25, 0.10),
    # Claude 3.5
    "claude-3-5-sonnet": (3.00, 15.00, 3.75, 0.30),
    "claude-3-5-haiku":  (0.80,  4.00, 1.00, 0.08),
    # Claude 3
    "claude-3-opus":   (15.00, 75.00, 18.75, 1.50),
    "claude-3-sonnet":  (3.00, 15.00,  3.75, 0.30),
    "claude-3-haiku":   (0.25,  1.25,  0.30, 0.03),

    # ── OpenAI GPT ────────────────────────────────────────────────────────
    # o1 series
    "o1":                (15.00, 60.00, 0.0, 7.50),
    "o1-mini":            (3.00, 12.00, 0.0, 1.50),
    "o3-mini":            (1.10,  4.40, 0.0, 0.55),
    # GPT-4o
    "gpt-4o":             (2.50, 10.00, 0.0, 1.25),
    "gpt-4o-mini":        (0.15,  0.60, 0.0, 0.075),
    # GPT-4 Turbo / legacy
    "gpt-4-turbo":       (10.00, 30.00, 0.0, 0.0),
    "gpt-4":             (30.00, 60.00, 0.0, 0.0),
    # GPT-3.5
    "gpt-3.5-turbo":      (0.50,  1.50, 0.0, 0.0),
}


def compute_cost(
    tokens_in: int,
    tokens_out: int,
    model: str,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> Tuple[float, bool]:
    """Compute cost in USD for a model call.

    Args:
        tokens_in: Input token count (uncached).
        tokens_out: Output token count.
        model: Model identifier (e.g. "claude-3-5-sonnet-20241022"). Used for lookup;
               partial matches fall back to the base model if configured.
        cache_creation_tokens: Tokens written to the cache.
        cache_read_tokens: Tokens read from the cache.

    Returns:
        (cost_usd, estimated):
        - cost_usd: Computed cost in USD.
        - estimated: True if model was not in the pricing table (fallback rate used).
    """
    key = _resolve_model_key(model)
    if key in PRICING_TABLE:
        input_per_1m, output_per_1m, cache_write_per_1m, cache_read_per_1m = PRICING_TABLE[key]
        estimated = False
    else:
        # Unknown model: use claude-3-5-sonnet as mid-range default
        input_per_1m, output_per_1m, cache_write_per_1m, cache_read_per_1m = PRICING_TABLE["claude-3-5-sonnet"]
        estimated = True

    cost = (
        tokens_in * input_per_1m +
        tokens_out * output_per_1m +
        cache_creation_tokens * cache_write_per_1m +
        cache_read_tokens * cache_read_per_1m
    ) / 1_000_000
    return (round(cost, 6), estimated)


def _resolve_model_key(model: str) -> str:
    """Map API model string to a pricing table key.

    Handles versioned names (e.g., "claude-3-5-sonnet-20241022" -> "claude-3-5-sonnet")
    and dashboard CSV names (e.g., "Claude Sonnet 4" -> "claude-sonnet-4").
    Prefers longest matching key to avoid claude-3-5-sonnet -> claude-3-5.
    """
    normalized_model = model.lower().replace(" ", "-")
    matches = [
        key
        for key in PRICING_TABLE
        if normalized_model == key or normalized_model.startswith(key + "-")
    ]
    return max(matches, key=len) if matches else model

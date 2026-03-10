"""Core change detection engine for comparing cost events."""

from typing import Any

# Regression thresholds
REGRESSION_ABSOLUTE_THRESHOLD_USD = 0.01
REGRESSION_PERCENT_THRESHOLD = 0.10


def _compute_averages(events: list[dict]) -> dict[str, Any]:
    """Compute average metrics across a list of events."""
    count = len(events)
    if count == 0:
        return {
            "count": 0,
            "avg_cost_usd": 0.0,
            "avg_tokens_in": 0.0,
            "avg_tokens_out": 0.0,
            "avg_latency_ms": 0.0,
        }

    total_cost = sum(e["cost_usd"] for e in events)
    total_tokens_in = sum(e["tokens_in"] for e in events)
    total_tokens_out = sum(e["tokens_out"] for e in events)
    total_latency = sum(e["latency_ms"] for e in events)

    return {
        "count": count,
        "avg_cost_usd": total_cost / count,
        "avg_tokens_in": total_tokens_in / count,
        "avg_tokens_out": total_tokens_out / count,
        "avg_latency_ms": total_latency / count,
    }


def compare_commits(sink, feature: str, commit_a: str, commit_b: str) -> dict[str, Any]:
    """Compare performance metrics between two code commits for a specific feature.

    Args:
        sink: An initialized sink (e.g. SQLiteSink) to retrieve events from.
        feature: The feature name to filter on.
        commit_a: The base commit hash "before".
        commit_b: The target commit hash "after".

    Returns:
        dict containing the computed averages for both commits, the deltas,
        and whether a regression was detected.
    """
    events_a = sink.get_events(feature, commit=commit_a)
    events_b = sink.get_events(feature, commit=commit_b)

    if not events_a:
        raise ValueError(f"No data found for commit_a: {commit_a}")
    if not events_b:
        raise ValueError(f"No data found for commit_b: {commit_b}")

    avg_a = _compute_averages(events_a)
    avg_b = _compute_averages(events_b)

    cost_delta = avg_b["avg_cost_usd"] - avg_a["avg_cost_usd"]
    
    # Avoid zero division
    if avg_a["avg_cost_usd"] > 0:
        cost_percent_change = cost_delta / avg_a["avg_cost_usd"]
    elif cost_delta > 0:
        # It went from 0.0 to something > 0.0 -> Infinite % increase
        cost_percent_change = float('inf')
    else:
        # 0.0 to 0.0
        cost_percent_change = 0.0

    is_regression = (
        cost_percent_change >= REGRESSION_PERCENT_THRESHOLD and
        cost_delta >= REGRESSION_ABSOLUTE_THRESHOLD_USD
    )

    return {
        "commit_a_stats": avg_a,
        "commit_b_stats": avg_b,
        "delta_cost_usd": cost_delta,
        "percent_change": cost_percent_change,
        "is_regression": is_regression,
    }

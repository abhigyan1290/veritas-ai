"""Core change detection engine for comparing cost events."""

from typing import Any, Optional

# Regression thresholds
REGRESSION_ABSOLUTE_THRESHOLD_USD = 0.01
REGRESSION_PERCENT_THRESHOLD = 0.10


def strip_dirty_suffix(commit: str) -> str:
    """Remove the ``+dirty`` suffix from a commit hash, if present.

    >>> strip_dirty_suffix("abc1234+dirty")
    'abc1234'
    >>> strip_dirty_suffix("abc1234")
    'abc1234'
    """
    if commit.endswith("+dirty"):
        return commit[: -len("+dirty")]
    return commit


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


def filter_events_by_tags(events: list[dict], tags: Optional[dict[str, str]]) -> list[dict]:
    """Filter a list of events down to those matching all specified tags."""
    if not tags:
        return events
        
    filtered = []
    for e in events:
        e_tags = e.get("tags")
        if isinstance(e_tags, str):
            import json
            try:
                e_tags = json.loads(e_tags)
            except json.JSONDecodeError:
                e_tags = {}
        
        if not isinstance(e_tags, dict):
            e_tags = {}
            
        # Check if EVERY tag filter is present in the event's tags
        if all(str(e_tags.get(k)) == str(v) for k, v in tags.items()):
            filtered.append(e)
            
    return filtered


def compare_commits(
    sink,
    feature: str,
    commit_a: str,
    commit_b: str,
    include_dirty: bool = False,
    tags: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Compare performance metrics between two code commits for a specific feature.

    Args:
        sink: An initialized sink (e.g. SQLiteSink) to retrieve events from.
        feature: The feature name to filter on.
        commit_a: The base commit hash "before".
        commit_b: The target commit hash "after".
        include_dirty: If False (default), events whose ``code_version``
            ends with ``+dirty`` are excluded from the comparison because
            they represent untested, in-progress code.
        tags: Optional dict of key-value tags to filter events by.

    Returns:
        dict containing the computed averages for both commits, the deltas,
        and whether a regression was detected.
    """
    if commit_a == "unknown" or commit_b == "unknown":
        raise ValueError(
            f"Cannot compare commits: commit_{'a' if commit_a == 'unknown' else 'b'} is 'unknown'. "
            "Git version could not be resolved. Set the VERITAS_CODE_VERSION environment variable "
            "to the deployed commit SHA, or pass code_version to veritas.init()."
        )

    events_a = sink.get_events(feature, commit=commit_a)
    events_b = sink.get_events(feature, commit=commit_b)

    if include_dirty:
        # Also fetch +dirty variants so dirty events are included in comparison
        if not commit_a.endswith("+dirty"):
            events_a = events_a + sink.get_events(feature, commit=f"{commit_a}+dirty")
        if not commit_b.endswith("+dirty"):
            events_b = events_b + sink.get_events(feature, commit=f"{commit_b}+dirty")
    else:
        # Also fetch events for the clean base hash when the user passes a dirty hash
        clean_a = strip_dirty_suffix(commit_a)
        clean_b = strip_dirty_suffix(commit_b)
        if clean_a != commit_a:
            events_a = events_a + sink.get_events(feature, commit=clean_a)
        if clean_b != commit_b:
            events_b = events_b + sink.get_events(feature, commit=clean_b)

        # Filter out +dirty events
        events_a = [e for e in events_a if not str(e.get("code_version", "")).endswith("+dirty")]
        events_b = [e for e in events_b if not str(e.get("code_version", "")).endswith("+dirty")]

    if tags:
        events_a = filter_events_by_tags(events_a, tags)
        events_b = filter_events_by_tags(events_b, tags)

    if not events_a:
        raise ValueError(f"No data found for commit_a: {commit_a} matching tags {tags}")
    if not events_b:
        raise ValueError(f"No data found for commit_b: {commit_b} matching tags {tags}")

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


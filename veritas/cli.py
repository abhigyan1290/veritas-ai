"""Command-line interface for Veritas."""

import argparse
import sys
from veritas.sinks import SQLiteSink
from veritas.engine import compare_commits, _compute_averages, strip_dirty_suffix


def _format_money(usd: float) -> str:
    """Format float to USD."""
    return f"${usd:.6f}"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a simple, dependency-free ASCII table."""
    if not rows:
        return "No data."

    # Find max width for each column
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    # padding
    col_widths = [w + 2 for w in col_widths]

    def render_row(row_data):
        return "|".join(str(val).ljust(w) for val, w in zip(row_data, col_widths))

    lines = []
    lines.append("-" * (sum(col_widths) + len(col_widths) - 1))
    lines.append(render_row(headers))
    lines.append("-" * (sum(col_widths) + len(col_widths) - 1))
    for row in rows:
        lines.append(render_row(row))
    lines.append("-" * (sum(col_widths) + len(col_widths) - 1))

    return "\n".join(lines)


def run_diff(args):
    """Run the diff command to compare two commits."""
    # Note: args.commit_a and args.commit_b are bounded by explicit `dest` overrides
    include_dirty = getattr(args, "include_dirty", False)
    dirty_note = " (including +dirty events)" if include_dirty else ""
    print(f"Comparing feature '{args.feature}' between ({args.commit_a}) and ({args.commit_b}){dirty_note}...\\n")

    sink = SQLiteSink()
    # Parse tags list back into a dict
    tags_dict = {}
    if getattr(args, "tag", None):
        for t in args.tag:
            if "=" in t:
                k, v = t.split("=", 1)
                tags_dict[k] = v

    try:
        res = compare_commits(sink, args.feature, args.commit_a, args.commit_b, include_dirty=include_dirty, tags=tags_dict or None)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        sink.close()

    headers = ["Metric", "Commit A (Base)", "Commit B (Target)", "Delta"]
    
    a_cost = _format_money(res['commit_a_stats']['avg_cost_usd'])
    b_cost = _format_money(res['commit_b_stats']['avg_cost_usd'])
    delta_val = res['delta_cost_usd']
    delta_str = f"{_format_money(delta_val)} ({res['percent_change'] * 100:.2f}%)"

    rows = [
        ["Samples", str(res['commit_a_stats']['count']), str(res['commit_b_stats']['count']), "-"],
        ["Avg Cost/Req", a_cost, b_cost, delta_str],
        [
            "Avg Tokens In",
            f"{res['commit_a_stats']['avg_tokens_in']:.1f}",
            f"{res['commit_b_stats']['avg_tokens_in']:.1f}",
            f"{res['commit_b_stats']['avg_tokens_in'] - res['commit_a_stats']['avg_tokens_in']:.1f}"
        ],
        [
            "Avg Tokens Out",
            f"{res['commit_a_stats']['avg_tokens_out']:.1f}",
            f"{res['commit_b_stats']['avg_tokens_out']:.1f}",
            f"{res['commit_b_stats']['avg_tokens_out'] - res['commit_a_stats']['avg_tokens_out']:.1f}"
        ],
    ]

    print(_render_table(headers, rows))

    print("\\nVerdict:")
    if res["is_regression"]:
        print("❌ REGRESSION DETECTED: Cost increased beyond acceptable thresholds.")
        sys.exit(1)
    else:
        print("✅ OK: No significant cost regression detected.")
        sys.exit(0)


def run_stats(args):
    """Run the stats command."""
    print(f"Aggregating stats for feature '{args.feature}' since {args.since}...\\n")

    sink = SQLiteSink()
    try:
        events = sink.get_events(args.feature, since_iso=args.since)
    finally:
        sink.close()

    if not events:
        print("No data found.")
        sys.exit(0)

    avg = _compute_averages(events)
    
    headers = ["Metric", "Value"]
    rows = [
        ["Total Samples", str(avg["count"])],
        ["Avg Cost/Req", _format_money(avg["avg_cost_usd"])],
        ["Avg Tokens In", f"{avg['avg_tokens_in']:.1f}"],
        ["Avg Tokens Out", f"{avg['avg_tokens_out']:.1f}"],
        ["Avg Latency (ms)", f"{avg['avg_latency_ms']:.1f}"]
    ]
    
    print(_render_table(headers, rows))


def main():
    parser = argparse.ArgumentParser(description="Veritas - AI Cost Attribution & Change Detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Diff command
    diff_parser = subparsers.add_parser("diff", help="Compare cost metrics between two commits.")
    diff_parser.add_argument("--feature", required=True, help="Feature name to filter by.")
    diff_parser.add_argument("--from", dest="commit_a", required=True, help="Base commit hash.")
    diff_parser.add_argument("--to", dest="commit_b", required=True, help="Target commit hash.")
    diff_parser.add_argument(
        "--include-dirty",
        dest="include_dirty",
        action="store_true",
        default=False,
        help="Include events from +dirty (uncommitted) code versions in comparison.",
    )
    diff_parser.add_argument(
        "--tag",
        "-t",
        action="append",
        help="Filter by tag (format: key=value). Can be used multiple times.",
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="View aggregated metrics for a feature.")
    stats_parser.add_argument("--feature", required=True, help="Feature name to filter by.")
    stats_parser.add_argument(
        "--since", 
        required=True, 
        help="ISO 8601 timestamp string (e.g. 2025-06-01T00:00:00Z)"
    )

    args = parser.parse_args()

    if args.command == "diff":
        run_diff(args)
    elif args.command == "stats":
        run_stats(args)


if __name__ == "__main__":
    main()

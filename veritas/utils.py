"""Environment detection, git version resolution, timestamps."""

import subprocess
import os
from datetime import datetime, timezone


def get_current_commit_hash() -> str:
    """
    Return the current git commit hash, or 'unknown' if not in a repo or git unavailable.
    Supports a VERITAS_MOCK_COMMIT environment variable override for UI demonstrations.
    """
    # 1. Check for the mock override first (used in the YC Demo narrative)
    mock_commit = os.environ.get("VERITAS_MOCK_COMMIT")
    if mock_commit:
        return mock_commit
        
    # 2. Extract real git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
            
        return "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        # FileNotFoundError means git is not installed on the system
        return "unknown"


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

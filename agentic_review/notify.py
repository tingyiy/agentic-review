"""Where a failure goes when nobody is watching the run page.

A review that fails silently is worse than one that fails: the PR shows no
finding and no red check, which is indistinguishable from a clean review. But
alerting is the most site-specific thing this tool does, so it is one function
and one environment variable rather than an integration.
"""
import os
import subprocess
import sys

#: A shell command that receives the alert text on stdin. Empty means stderr
#: only, which is the right default: on GitHub Actions the red check IS the
#: alert, and a second channel only matters when a run can fail unwatched.
ALERT_COMMAND = os.environ.get("REVIEW_ALERT_COMMAND", "")


def alert(message):
    """Best effort, always. Failing to report a failure must not raise a second
    one on top of it — the original error is the one worth keeping."""
    print(message, file=sys.stderr, flush=True)
    if not ALERT_COMMAND:
        return
    try:
        subprocess.run(ALERT_COMMAND, shell=True, input=message, text=True,
                       timeout=30, check=False)
    except Exception as e:  # noqa: BLE001
        print(f"(alert command failed: {type(e).__name__}: {e})",
              file=sys.stderr, flush=True)

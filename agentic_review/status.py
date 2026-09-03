"""A commit status that says what the reviewer is doing, on the PR page.

The workflow's own check row is not enough. Copilot's automatic review
request lands a second later and starts a second run of the same workflow,
which takes the no-op arm and finishes in three seconds — and GitHub's merge
box shows only the NEWEST run per workflow. So the page said "All checks have
passed" while a real review was five minutes into its work, and the author
concluded it had never launched (caeli-marketing#227, 2026-09-03).

A commit status is shown independently of workflow runs, so one context —
`agentic-review` — carries the truth: pending while the review runs, the
verdict when it posts, an error when the run died. It is never made a
required check by this tool; that is the repository's decision.

Best-effort throughout. The status is a courtesy to the reader, and a token
without `statuses: write` must cost exactly one log line, not the review.
"""
import os
import urllib.error

from . import github
from .config import ORG

CONTEXT = os.environ.get("REVIEW_STATUS_CONTEXT", "agentic-review")

#: GitHub's limit for a status description.
MAX_DESCRIPTION = 140


def _run_url():
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run = os.environ.get("GITHUB_RUN_ID")
    if server and repo and run:
        return f"{server}/{repo}/actions/runs/{run}"
    return None


def set_status(repo, sha, state, description):
    """One status. True if GitHub took it; False, with one log line, if not."""
    if not (repo and sha):
        return False
    body = {"state": state, "context": CONTEXT,
            "description": description[:MAX_DESCRIPTION]}
    url = _run_url()
    if url:
        body["target_url"] = url
    try:
        github.request(f"/repos/{ORG}/{repo}/statuses/{sha}", method="POST",
                       body=body)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        code = getattr(e, "code", None)
        hint = (" — the bot token needs Commit statuses: read and write"
                if code == 403 else "")
        print(f"[status] could not set {state!r} on {sha[:7]}: "
              f"{type(e).__name__}{f' {code}' if code else ''}{hint}", flush=True)
        return False


def pending(repo, sha):
    return set_status(repo, sha, "pending", "review in progress")


def done(repo, sha, event, summary):
    """The verdict as a status. Only a blocking review is a `failure`; a
    COMMENT is a review that happened, not a broken build."""
    event = (event or "").upper()
    if event.startswith("REQUEST_CHANGES"):
        return set_status(repo, sha, "failure", f"changes requested — {summary}")
    if event.startswith("APPROVE"):
        return set_status(repo, sha, "success", f"approved — {summary}")
    return set_status(repo, sha, "success", f"commented — {summary}")


def failed(repo, sha, reason):
    return set_status(repo, sha, "error", f"review failed: {reason}")

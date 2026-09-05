"""A commit status that says what the reviewer is doing, on the PR page.

The workflow's own check row is not enough. Copilot's automatic review
request lands a second later and starts a second run of the same workflow,
which takes the no-op arm and finishes in three seconds — and GitHub's merge
box shows only the NEWEST run per workflow. So the page said "All checks have
passed" while a real review was five minutes into its work, and the author
concluded it had never launched (a private deployment, 2026-09-03).

A commit status is shown independently of workflow runs, so one context —
`agentic-review` — carries the truth: pending while the review runs, the
verdict when it posts, an error when the run died. It is never made a
required check by this tool; that is the repository's decision.

Best-effort throughout. The status is a courtesy to the reader, and a token
without `statuses: write` must cost exactly one log line, not the review —
so `set_status` catches EVERYTHING, and a dry run sets nothing at all.
"""
import os

from . import github
from .config import ORG, STATUS_CONTEXT as CONTEXT

#: GitHub's limit for a status description.
MAX_DESCRIPTION = 140

#: A status is a courtesy on the way to the review; it must not be allowed to
#: hold the review up for the full request timeout when GitHub is slow.
TIMEOUT = 15


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
    if os.environ.get("DRY"):
        # THE DRY CONTRACT IS ENFORCED HERE, not at each call site: the entry
        # point's failure path could not otherwise tell a rehearsal from a run.
        print(f"[status] DRY: would set {state!r} on {sha[:7]}: {description[:60]}",
              flush=True)
        return False
    body = {"state": state, "context": CONTEXT,
            "description": description[:MAX_DESCRIPTION]}
    url = _run_url()
    if url:
        body["target_url"] = url
    try:
        github.request(f"/repos/{ORG}/{repo}/statuses/{sha}", method="POST",
                       body=body, timeout=TIMEOUT)
        return True
    except Exception as e:  # noqa: BLE001 — a courtesy must never cost the review
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


def nothing_to_review(repo, sha, why):
    """There was no reviewable text, and the PR page has to say so.

    caeli-marketing#243 changed `public/og-image.png` and `package-lock.json`
    and nothing else: both are on the generated/binary skip list, so the run
    printed "nothing reviewable" and exited in nine seconds — correctly, since
    a text reviewer has nothing to say about a PNG. But it exited BEFORE the
    pending status, so the pull request carried no `agentic-review` status at
    all, which reads exactly like a reviewer that never ran. Deciding not to
    review is a result; a result nobody can see is the quiet death this repo
    treats as worse than a loud failure.

    `success`, not `neutral`: a commit status has no neutral state, and this is
    not a failure — there was simply nothing to read.
    """
    return set_status(repo, sha, "success", f"nothing to review — {why}")


def failed(repo, sha, reason):
    return set_status(repo, sha, "error", f"review failed: {reason}")

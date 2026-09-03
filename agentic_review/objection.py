"""Does an author's comment object to the review? If so, ask for another.

A PR comment is an `issue_comment` event, and for a long time nothing listened
to it: an author who replied "not taking this, the code does X" was read on
the NEXT run, but nothing started one. infra#161, 2026-09-03: two findings
disputed in writing, no run, the PR sat. Wiring the comment event straight to
a review would be the wrong fix — a "thanks", or a note describing the fix
just pushed, would cost a multi-minute self-hosted run and CANCEL an in-flight
one — so this sits in between.

It runs on every author comment, in its own concurrency group, and does one
cheap thing: decide whether the comment disputes the review. A mention of the
reviewer is an objection by definition. Anything else is put to the model with
the review it is replying to. Only a "yes" turns into a review, and it does so
by RE-REQUESTING the reviewer — the ordinary `review_requested` path, in the
ordinary group, with every guard that path already has. This module never
reviews anything itself.

    python -m agentic_review.objection <repo> <pr> <comment-id>

Exit 0 whatever it decides; 1 only when it could not decide AND could not
fall back. A wrong "yes" costs one cheap review. A wrong "no" is the deadlock
this exists to remove — so when the model cannot be reached, it says yes.
"""
import json
import sys

from . import llm, notify
from .errors import ReviewError
from .review import ORG, _me, _release_review_request, gh

PROMPT = """A pull-request author has replied under an automated code review.
Decide whether the reply OBJECTS to the review: it disputes, rejects, or asks
to reconsider one or more of the review's findings, or says a finding is wrong
or not applicable.

It is NOT an objection when the reply only thanks the reviewer, acknowledges a
finding, says a fix was made or pushed, explains what was changed, asks an
unrelated question, or is addressed to someone else.

THE REVIEW being replied to:
---
{review}
---

THE AUTHOR'S REPLY:
---
{comment}
---

Answer as JSON: {{"objection": true|false, "why": "<one sentence>"}}"""


def decide(comment, review, me):
    """(objection?, why). The mention is decided here; the rest is the model's.

    Fail OPEN: a model that cannot be reached returns True. The cost of a wrong
    yes is one review the author did not need; the cost of a wrong no is the
    silence this whole path exists to end.
    """
    if me and f"@{me}".lower() in (comment or "").lower():
        return True, f"mentions @{me}"
    try:
        text = llm.chat(
            [{"role": "user", "content": PROMPT.format(
                review=_clip(review, 6000, "review"),
                comment=_clip(comment, 3000, "reply"))}],
            json_mode=True, max_tokens=300, timeout=60)
        verdict = llm.parse_json_reply(text)
    except Exception as e:  # noqa: BLE001 — anything the model could not answer
        # Not only ReviewError: a proxy's HTML error page behind a 200 arrives
        # as a JSONDecodeError from the transport, and it is the same "no
        # answer". The policy is fail open, and it has to be for every shape.
        return True, f"could not ask the model ({type(e).__name__}: {str(e)[:80]}) — assuming yes"
    if not isinstance(verdict, dict):
        # Parseable but not an object (a list, a bare string). Same policy as
        # unreachable: the model did not say no.
        return True, f"unusable reply ({type(verdict).__name__}) — assuming yes"
    if verdict.get("objection") is None:
        # The field missing or null is not a "no" either.
        return True, "reply carries no verdict — assuming yes"
    return _yes(verdict["objection"]), str(verdict.get("why") or "")[:200]


def _clip(text, limit, what):
    """The first `limit` chars, SAYING SO when that is not all of them. A
    silent cut teaches the model it has seen everything — and the finding the
    author disputes may be exactly what fell off the end."""
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[{what} truncated here at {limit:,} chars]"


def _yes(value):
    """A JSON boolean, or the model's spelling of one. `bool("false")` is
    True, and a string where a boolean was asked for is a known habit even in
    JSON mode — so the string is read, not just its presence. Anything that is
    neither (a number, a list) is a malformed reply, and a malformed reply is
    not a "no": fail open."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        word = value.strip().lower()
        if word in ("false", "no"):
            return False
        # "true", "yes" — or "maybe", "1", anything else the prompt did not
        # ask for. Only a recognisable no is a no.
        return True
    return True


def _latest_review_by(reviews, login):
    ours = [r for r in reviews if (r.get("user") or {}).get("login") == login]
    return ours[-1] if ours else None


def classify_and_request(repo, pr, comment_id):
    """One verdict line, and a re-request when the verdict is yes."""
    me = _me()
    comment = json.loads(gh(f"/repos/{ORG}/{repo}/issues/comments/{comment_id}"))
    who = (comment.get("user") or {}).get("login") or ""
    if who == me:
        return f"our own comment — nothing to do"
    meta = json.loads(gh(f"/repos/{ORG}/{repo}/pulls/{pr}"))
    author = (meta.get("user") or {}).get("login") or ""
    if who != author:
        return f"comment by {who}, not the author ({author}) — nothing to do"
    if meta.get("draft") or meta.get("state") != "open":
        return "PR is not open for review — nothing to do"
    reviews = json.loads(gh(f"/repos/{ORG}/{repo}/pulls/{pr}/reviews?per_page=100"))
    last = _latest_review_by(reviews, me)
    if not last:
        return "no review of ours to object to — nothing to do"
    head = (meta.get("head") or {}).get("sha") or ""
    if last.get("state") == "APPROVED" and last.get("commit_id") == head:
        # An approval is final for its commit: only a new commit reviews again.
        return "our review at this head is an approval — only a new commit reviews again"
    yes, why = decide(comment.get("body") or "", last.get("body") or "", me)
    if not yes:
        return f"not an objection ({why}) — no review"
    # DELETE then POST: GitHub emits no `review_requested` when the reviewer is
    # already in the list, and a failed run leaves us there. If we ARE listed
    # and the release failed, the POST would be a silent no-op — say so
    # loudly instead of reporting a re-review that will never start.
    listed = me in [(u or {}).get("login") for u in meta.get("requested_reviewers") or []]
    if not _release_review_request(repo, pr) and listed:
        notify.alert(f"🚨 pr-review: {repo}#{pr} — the author objected ({why}) but "
                     f"the stale review request could not be released, so a "
                     f"re-request cannot fire. Re-request {me} by hand.")
        return f"objection ({why}) — could NOT clear the stale request; alerted"
    gh(f"/repos/{ORG}/{repo}/pulls/{pr}/requested_reviewers", method="POST",
       body={"reviewers": [me]})
    return f"objection ({why}) — re-review requested"


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        raise SystemExit(2)
    repo, pr, comment_id = sys.argv[1:4]
    try:
        print(f"[objection] {repo}#{pr} comment {comment_id}: "
              f"{classify_and_request(repo, pr, comment_id)}", flush=True)
    except Exception as e:  # noqa: BLE001 — last-resort guard
        notify.alert(f"🚨 pr-review objection check crashed on {repo}#{pr}: "
                     f"{type(e).__name__}: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

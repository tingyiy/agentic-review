"""GitHub credentials and the raw REST call.

Kept apart from `review` because the identity question — WHICH token posts the
review — is the one piece of this tool a new adopter must get right before
anything else works, and burying it in a 2,000-line module hides it.
"""
import http.client
import json
import os
import time
import urllib.error
import urllib.request

from . import env
from .errors import ReviewError

API = os.environ.get("GITHUB_API_URL", "https://api.github.com")


def token(review=True):
    """A GitHub token, preferring a BOT identity over a human one.

    Two tokens are typically available and they are NOT interchangeable:

      REVIEW_GITHUB_TOKEN   a bot account — what a review must be posted as
      GITHUB_TOKEN          whoever is running this

    Anything a human reads as coming from someone must use the bot. A review
    posted with a personal token appears as the author reviewing their own pull
    request, which is not a review — it is noise wearing a reviewer's name, and
    afterwards it cannot be told apart from a real one in the PR timeline.

    `review=False` asks for whichever token works, for read-only work where
    identity does not matter. It still prefers the bot: a token scoped to
    exactly the repositories it needs is the better default, and falling back
    only when the bot is absent keeps a missing grant loud rather than silently
    impersonating a person.
    """
    tok = (env.get("REVIEW_GITHUB_TOKEN")
           # Caeli's own name for it, kept so the existing box needs no edit.
           or env.get("GITHUB_REVIEW_TOKEN"))
    if tok:
        return tok
    if review:
        raise ReviewError(
            "REVIEW_GITHUB_TOKEN is not set — refusing to post a review as "
            "whoever happens to be running this")
    return env.get("GITHUB_TOKEN") or env.get("GH_TOKEN")


def request(path, method="GET", body=None, accept="application/vnd.github+json",
            token_for_review=True, timeout=60):
    """One GitHub REST call. Returns the response body as text.

    Deliberately not paginated: every caller that can exceed one page asks for
    `?per_page=100` and pages explicitly, because the API returns the OLDEST 30
    by default — so a silent single-page read of a long conversation returns
    exactly the part nobody needed.
    """
    url = path if path.startswith("http") else f"{API}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token(review=token_for_review)}")
    req.add_header("Accept", accept)
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    return _send(req, timeout=timeout)


#: Failures that mean "the connection died", not "GitHub said no". A dropped
#: connection is retried ONCE, after a pause: agentic-review#2's own reviews
#: died twice on 2026-09-03 (IncompleteRead at 06:01, RemoteDisconnected at
#: 06:12) while the box's Wi-Fi was flapping, each announced as "crashed" and
#: each posting nothing. An HTTPError is deliberately NOT here — a 4xx/5xx is
#: an answer, and the callers already reason about those.
_DROPPED = (http.client.IncompleteRead, ConnectionError)


def _send(req, attempts=2, timeout=60):
    # ONLY A GET IS RETRIED. A dropped connection says nothing about whether
    # the server acted: an `IncompleteRead` on `POST .../reviews` is a review
    # that may already be posted, and sending it again posts it twice (the
    # review's 🟡 on this PR). A dropped write raises at once, named, and the
    # caller — or the next run — decides.
    if req.get_method() != "GET":
        attempts = 1
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode()
        except _DROPPED as e:
            if attempt + 1 >= attempts:
                raise ReviewError(
                    f"GitHub connection dropped on {req.get_method()} "
                    f"{req.selector}"
                    f"{' twice' if attempts > 1 else ' (not retried: a write)'}"
                    f": {type(e).__name__}: {e}") from e
            print(f"[github] connection dropped ({type(e).__name__}) — "
                  f"retrying once", flush=True)
            time.sleep(2)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def job_log(path):
    """A job's log as text.

    `/actions/jobs/{id}/logs` answers 302 to a signed blob-storage URL. The
    redirect must be followed WITHOUT the Authorization header — the blob store
    is not GitHub, and a stray bearer token both leaks it and can make the
    signed request fail — so urllib's automatic redirect (which re-sends
    headers) is disabled and the second hop is fetched bare.
    """
    req = urllib.request.Request(f"{API}{path}")
    req.add_header("Authorization", f"Bearer {token(review=False)}")
    req.add_header("Accept", "application/vnd.github+json")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=60) as r:
            return r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 307) or not e.headers.get("Location"):
            raise
        with urllib.request.urlopen(e.headers["Location"], timeout=60) as r2:
            return r2.read().decode("utf-8", "replace")

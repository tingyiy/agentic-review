"""The commit status: what the PR page says while the reviewer works.

2026-09-03, a private deployment: five minutes into a real review the merge
box read "All checks have passed", because Copilot's automatic request had
started a second, no-op run of the same workflow and GitHub shows only the
newest run per workflow. The author concluded the review never launched.
"""
import urllib.error
import io

import pytest

from agentic_review import status

#: The real sender, taken at import — before the autouse conftest fixture
#: replaces `status.set_status` with a recorder for every other test.
_REAL_SET_STATUS = status.set_status


def _capture(monkeypatch, fail=None):
    monkeypatch.setattr(status, "set_status", _REAL_SET_STATUS)
    posts = []

    def request(path, method="GET", body=None, accept="", timeout=60):
        if fail is not None:
            raise fail
        posts.append((method, path, body))
        return "{}"
    monkeypatch.setattr(status.github, "request", request)
    monkeypatch.setattr(status, "ORG", "example-org")
    return posts


class TestWhatIsPosted:
    def test_pending_says_in_progress(self, monkeypatch):
        posts = _capture(monkeypatch)
        assert status.pending("app", "abc123") is True
        (method, path, body), = posts
        assert method == "POST" and path == "/repos/example-org/app/statuses/abc123"
        assert body["state"] == "pending" and body["context"] == "agentic-review"
        assert body["description"] == "review in progress"

    @pytest.mark.parametrize("event,state,word", [
        ("APPROVE", "success", "approved"),
        ("COMMENT", "success", "commented"),
        ("REQUEST_CHANGES", "failure", "changes requested"),
        # post_review's own wording when GitHub refused an approval on our PR.
        ("COMMENT (approve refused on own PR)", "success", "commented"),
    ])
    def test_the_verdict_maps_to_a_state(self, monkeypatch, event, state, word):
        posts = _capture(monkeypatch)
        status.done("app", "abc123", event, "2 finding(s): 1 high, 1 low")
        body = posts[0][2]
        assert body["state"] == state
        assert body["description"].startswith(word)
        assert "2 finding(s)" in body["description"]

    def test_a_failure_is_an_error_state(self, monkeypatch):
        posts = _capture(monkeypatch)
        status.failed("app", "abc123", "fireworks 503")
        assert posts[0][2]["state"] == "error"
        assert "review failed: fireworks 503" in posts[0][2]["description"]

    def test_the_description_fits_githubs_limit(self, monkeypatch):
        posts = _capture(monkeypatch)
        status.failed("app", "abc123", "x" * 500)
        assert len(posts[0][2]["description"]) <= 140

    def test_the_run_url_is_the_target_when_actions_provides_one(self, monkeypatch):
        posts = _capture(monkeypatch)
        monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
        monkeypatch.setenv("GITHUB_REPOSITORY", "example-org/app")
        monkeypatch.setenv("GITHUB_RUN_ID", "42")
        status.pending("app", "abc123")
        assert posts[0][2]["target_url"] == "https://github.com/example-org/app/actions/runs/42"

    def test_no_target_outside_actions(self, monkeypatch):
        posts = _capture(monkeypatch)
        for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"):
            monkeypatch.delenv(k, raising=False)
        status.pending("app", "abc123")
        assert "target_url" not in posts[0][2]


class TestItNeverCostsTheReview:
    def test_a_403_is_one_log_line_naming_the_permission(self, monkeypatch, capsys):
        err = urllib.error.HTTPError("u", 403, "Forbidden", {}, io.BytesIO(b""))
        _capture(monkeypatch, fail=err)
        assert status.pending("app", "abc123") is False
        out = capsys.readouterr().out
        assert "403" in out and "Commit statuses" in out

    def test_a_network_failure_is_swallowed(self, monkeypatch):
        _capture(monkeypatch, fail=urllib.error.URLError("no route"))
        assert status.done("app", "abc123", "APPROVE", "0 finding(s)") is False

    def test_a_dry_run_sets_nothing_on_any_path(self, monkeypatch, capsys):
        """Copilot on this PR: the entry point's failure path posted an error
        status from a DRY run. The contract is enforced in one place."""
        posts = _capture(monkeypatch)
        monkeypatch.setenv("DRY", "1")
        assert status.pending("app", "abc123") is False
        assert status.failed("app", "abc123", "boom") is False
        assert posts == [] and "DRY" in capsys.readouterr().out

    def test_an_unexpected_exception_is_swallowed_too(self, monkeypatch):
        """`github.token()` raises ReviewError without a bot token; that must
        not turn a status into a crash."""
        from agentic_review.errors import ReviewError
        _capture(monkeypatch, fail=ReviewError("no token"))
        assert status.pending("app", "abc123") is False

    def test_no_sha_means_nothing_is_posted(self, monkeypatch):
        posts = _capture(monkeypatch)
        assert status.failed("app", None, "x") is False and posts == []


class TestTheEntryPointClearsPending:
    def test_a_crash_marks_the_status_as_error(self, monkeypatch):
        from agentic_review import __main__ as entry
        seen = []
        monkeypatch.setattr(entry.status, "failed", lambda r, s, why: seen.append((r, s, why)))
        monkeypatch.setattr(entry.notify, "alert", lambda m: None)
        monkeypatch.setitem(entry._CURRENT, "repo", "app")
        monkeypatch.setitem(entry._CURRENT, "head", "abc123")
        monkeypatch.setattr(entry, "_main_unless_superseded",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(SystemExit):
            entry.main()
        assert seen == [("app", "abc123", "RuntimeError: boom")]


class TestTheWiringInMain:
    """Copilot's suppressed comment on this PR: the helpers were tested, the
    orchestration was not — a regression removing the calls stayed green."""

    def _drive(self, pr, monkeypatch, order, nothing_new=""):
        import json
        from agentic_review import review as pr
        monkeypatch.setattr(pr, "review_findings", lambda *a, **k: [])
        monkeypatch.setattr(pr, "_revise", lambda f, w, r: (f, []))
        monkeypatch.setattr(pr.checks, "run_all", lambda *a, **k: [])
        monkeypatch.setattr(pr, "checkout", lambda *a: order.append("checkout"))
        monkeypatch.setattr(pr, "build_context", lambda *a: "")
        monkeypatch.setattr(pr, "conversation", lambda *a: "")
        monkeypatch.setattr(pr, "commit_messages", lambda *a: [])
        monkeypatch.setattr(pr, "pr_diff", lambda *a: ("--- a/x\n+++ b/x\n@@\n", False, 0))
        monkeypatch.setattr(pr, "_already_reviewed",
                            lambda *a, **k: order.append("nothing-new check") or nothing_new)
        monkeypatch.setattr(pr, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(pr, "post_review",
                            lambda *a, **k: order.append("post") or "APPROVE")
        monkeypatch.setattr(pr.status, "pending",
                            lambda repo, sha: order.append(("pending", repo, sha)))
        monkeypatch.setattr(pr.status, "done",
                            lambda repo, sha, event, summary: order.append(("done", event, summary)))
        monkeypatch.setattr(pr, "gh", lambda *a, **k: json.dumps(
            {"draft": False, "state": "open", "merged": False, "title": "SCRUM-1 x",
             "user": {"login": "someone"}, "head": {"sha": "a" * 40}}))
        monkeypatch.setattr(pr.sys, "argv", ["pr-review", "app", "7"])
        monkeypatch.delenv("DRY", raising=False)
        pr.main()

    def test_the_head_is_recorded_as_soon_as_the_metadata_is(self, monkeypatch):
        """So a failure in pr_diff or the nothing-new check still gets its
        error status (Copilot's third round)."""
        from agentic_review import review
        order = []
        monkeypatch.setitem(review._CURRENT, "head", None)
        self._drive(None, monkeypatch, order, nothing_new="same commit")
        assert review._CURRENT["head"] == "a" * 40

    def test_pending_after_the_nothing_new_guard_and_before_the_checkout(self, monkeypatch):
        order = []
        self._drive(None, monkeypatch, order)
        assert order[:3] == ["nothing-new check", ("pending", "app", "a" * 40), "checkout"]

    def test_the_verdict_is_set_after_the_post(self, monkeypatch):
        order = []
        self._drive(None, monkeypatch, order)
        assert order[-2:] == ["post", ("done", "APPROVE", "0 finding(s): none")]

    def test_a_skipped_review_sets_no_status(self, monkeypatch):
        """Nothing new to review means the earlier status still tells the truth."""
        order = []
        self._drive(None, monkeypatch, order, nothing_new="same commit")
        assert order == ["nothing-new check"]

    def test_the_timeout_reaches_urlopen(self, monkeypatch):
        """Copilot's third round: the previous test mocked github.request, so
        it stayed green if the forwarding to _send and to urlopen was lost."""
        import io
        from agentic_review import github as gh_mod
        seen = {}
        class _R(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False
        def urlopen(req, timeout=None):
            seen["timeout"] = timeout; return _R(b"{}")
        monkeypatch.setattr(gh_mod.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(gh_mod, "token", lambda review=True: "t")
        gh_mod.request("/x", method="POST", body={"a": 1}, timeout=15)
        assert seen["timeout"] == 15

    def test_the_status_uses_a_short_timeout(self, monkeypatch):
        seen = {}
        def request(path, method="GET", body=None, accept="", timeout=60):
            seen["timeout"] = timeout; return "{}"
        monkeypatch.setattr(status, "set_status", _REAL_SET_STATUS)
        monkeypatch.setattr(status.github, "request", request)
        monkeypatch.setattr(status, "ORG", "example-org")
        status.pending("app", "abc123")
        assert seen["timeout"] < 60


class TestNothingReviewableIsStillAResult:
    """caeli-marketing#243: two files, `public/og-image.png` and
    `package-lock.json`, both on the generated/binary skip list. The run
    printed "nothing reviewable" and exited in nine seconds — correctly — but
    before the pending status, so the PR carried no `agentic-review` status at
    all and read exactly like a reviewer that never ran."""

    def test_it_says_so_on_the_commit(self, monkeypatch):
        posts = _capture(monkeypatch)
        status.nothing_to_review("app", "abc123", "2 generated/binary file(s)")
        (method, path, body), = posts
        assert method == "POST" and path.endswith("/statuses/abc123")
        assert body["state"] == "success"
        assert body["description"].startswith("nothing to review")
        assert "2 generated/binary file(s)" in body["description"]

    def test_a_skipped_review_no_longer_leaves_the_page_silent(self, monkeypatch):
        """Driven through `main`, because the bug was the ORDER of the exit and
        the status, not the wording of either."""
        import json
        from agentic_review import review as pr
        said = []
        monkeypatch.setattr(pr, "pr_diff", lambda *a: ("", [], pr._Skipped(["a.png", "b.lock"])))
        monkeypatch.setattr(pr, "gh", lambda *a, **k: json.dumps(
            {"draft": False, "state": "open", "merged": False, "title": "SCRUM-1 x",
             "user": {"login": "someone"}, "head": {"sha": "d" * 40}}))
        monkeypatch.setattr(pr, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(pr.status, "nothing_to_review",
                            lambda repo, sha, why: said.append((repo, sha, why)))
        monkeypatch.setattr(pr.sys, "argv", ["pr-review", "app", "7"])
        monkeypatch.delenv("DRY", raising=False)
        pr.main()
        assert said == [("app", "d" * 40, "2 generated/binary file(s), nothing else changed")]

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

    def request(path, method="GET", body=None, accept=""):
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
        entry._CURRENT.update(repo="app", head="abc123")
        monkeypatch.setattr(entry, "_main_unless_superseded",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(SystemExit):
            entry.main()
        assert seen == [("app", "abc123", "RuntimeError: boom")]

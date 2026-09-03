"""When an author's comment turns into another review — and when it must not.

infra#161, 2026-09-03: the author replied "Not taking either of these; both
are contradicted by the code" and nothing happened, because nothing listened
to comments. The fix must not swing the other way: a "thanks" is not a
request for a multi-minute run.
"""
import json

import pytest

from agentic_review import objection
from agentic_review.errors import ReviewError

ME = "review-bot"
AUTHOR = "octocat"
HEAD = "85827e7deadbeef"


def _wire(monkeypatch, *, comment_by=AUTHOR, comment_body="not taking this",
          reviews=None, draft=False, state="open", model=None,
          listed=False, release_ok=True):
    """Stub GitHub and the model; record every write and every model call."""
    calls = {"posts": [], "released": 0, "asked": [], "alerts": []}
    if reviews is None:
        reviews = [{"user": {"login": ME}, "state": "COMMENTED",
                    "commit_id": HEAD, "body": "🔵 finding one"}]

    def gh(path, method="GET", body=None, accept=""):
        if method == "POST":
            calls["posts"].append((path, body))
            return "{}"
        if "/issues/comments/" in path:
            return json.dumps({"user": {"login": comment_by}, "body": comment_body})
        if path.endswith("/reviews?per_page=100"):
            return json.dumps(reviews)
        if "/pulls/" in path:
            return json.dumps({"user": {"login": AUTHOR}, "draft": draft,
                               "state": state, "head": {"sha": HEAD},
                               "requested_reviewers": [{"login": ME}] if listed else []})
        raise AssertionError(f"unexpected call {method} {path}")

    def chat(messages, **kw):
        calls["asked"].append(messages[0]["content"])
        if isinstance(model, Exception):
            raise model
        return json.dumps(model if model is not None
                          else {"objection": True, "why": "disputes finding one"})

    monkeypatch.setattr(objection, "gh", gh)
    monkeypatch.setattr(objection, "_me", lambda: ME)
    def release(r, p):
        calls["released"] += 1
        return release_ok
    monkeypatch.setattr(objection, "_release_review_request", release)
    monkeypatch.setattr(objection.notify, "alert", calls["alerts"].append)
    monkeypatch.setattr(objection.llm, "chat", chat)
    return calls


def _requested(calls):
    return [b for p, b in calls["posts"] if p.endswith("/requested_reviewers")]


class TestAnObjectionAsksForAnotherReview:
    def test_the_model_saying_yes_re_requests_us(self, monkeypatch):
        calls = _wire(monkeypatch)
        out = objection.classify_and_request("infra", 161, 1)
        assert "re-review requested" in out
        assert _requested(calls) == [{"reviewers": [ME]}]

    def test_the_request_is_released_first(self, monkeypatch):
        """GitHub emits no `review_requested` when we are already in the list,
        and a failed run leaves us there — so a POST alone can be a silent no-op."""
        calls = _wire(monkeypatch)
        objection.classify_and_request("infra", 161, 1)
        assert calls["released"] == 1 and len(_requested(calls)) == 1

    def test_a_failed_release_while_listed_alerts_instead_of_posting(self, monkeypatch):
        """The re-review's 🔵: with us still in `requested_reviewers`, a POST
        emits no event, so "re-review requested" would have been a lie."""
        calls = _wire(monkeypatch, listed=True, release_ok=False)
        out = objection.classify_and_request("infra", 161, 1)
        assert "could NOT clear" in out and _requested(calls) == []
        assert calls["alerts"] and "by hand" in calls["alerts"][0]

    def test_a_failed_release_while_NOT_listed_still_posts(self, monkeypatch):
        """Nothing stale to clear: the POST alone fires the event."""
        calls = _wire(monkeypatch, listed=False, release_ok=False)
        objection.classify_and_request("infra", 161, 1)
        assert len(_requested(calls)) == 1 and calls["alerts"] == []

    def test_a_mention_is_an_objection_without_asking(self, monkeypatch):
        calls = _wire(monkeypatch, comment_body="@review-bot please look again")
        out = objection.classify_and_request("infra", 161, 1)
        assert "re-review requested" in out and "mentions @review-bot" in out
        assert calls["asked"] == []

    def test_the_model_sees_the_review_and_the_reply(self, monkeypatch):
        calls = _wire(monkeypatch, comment_body="the code already handles that")
        objection.classify_and_request("infra", 161, 1)
        prompt, = calls["asked"]
        assert "🔵 finding one" in prompt
        assert "the code already handles that" in prompt

    def test_a_truncated_input_says_so_to_the_model(self, monkeypatch):
        """The repo rule: every truncation says so in the text the model reads."""
        calls = _wire(monkeypatch, comment_body="x" * 3001,
                      reviews=[{"user": {"login": ME}, "state": "COMMENTED",
                                "commit_id": HEAD, "body": "r" * 6001}])
        objection.classify_and_request("infra", 161, 1)
        prompt, = calls["asked"]
        assert "[review truncated here at 6,000 chars]" in prompt
        assert "[reply truncated here at 3,000 chars]" in prompt

    def test_a_short_input_is_not_labelled_truncated(self, monkeypatch):
        calls = _wire(monkeypatch, comment_body="short")
        objection.classify_and_request("infra", 161, 1)
        assert "truncated" not in calls["asked"][0]

    def test_a_non_review_error_from_the_transport_also_fails_open(self, monkeypatch):
        """A proxy's HTML page behind a 200 is a JSONDecodeError, not a
        ReviewError — the review's 🔵 on this PR."""
        calls = _wire(monkeypatch, model=json.JSONDecodeError("bad", "<html>", 0))
        out = objection.classify_and_request("infra", 161, 1)
        assert "assuming yes" in out and len(_requested(calls)) == 1

    def test_an_unreachable_model_fails_OPEN(self, monkeypatch):
        """A wrong yes costs one cheap review. A wrong no is the deadlock."""
        calls = _wire(monkeypatch, model=ReviewError("fireworks 503"))
        out = objection.classify_and_request("infra", 161, 1)
        assert "assuming yes" in out and len(_requested(calls)) == 1


class TestTheModelsSpellingOfNo:
    """The review's two 🔵s on this PR."""

    def test_a_STRING_false_is_a_no(self, monkeypatch):
        calls = _wire(monkeypatch, model={"objection": "false", "why": "thanks only"})
        assert "not an objection" in objection.classify_and_request("infra", 161, 1)
        assert calls["posts"] == []

    def test_a_string_true_is_a_yes(self, monkeypatch):
        calls = _wire(monkeypatch, model={"objection": "true", "why": "disputes"})
        objection.classify_and_request("infra", 161, 1)
        assert len(_requested(calls)) == 1

    @pytest.mark.parametrize("reply", [{"why": "no field"}, {"objection": None}])
    def test_a_missing_or_null_verdict_fails_open(self, monkeypatch, reply):
        """The re-review's 🟡: str(None) is "none", which read as a no."""
        calls = _wire(monkeypatch, model=reply)
        out = objection.classify_and_request("infra", 161, 1)
        assert "assuming yes" in out and len(_requested(calls)) == 1

    @pytest.mark.parametrize("word", ["no", "False", " NO "])
    def test_the_known_spellings_of_no_are_a_no(self, monkeypatch, word):
        calls = _wire(monkeypatch, model={"objection": word})
        assert "not an objection" in objection.classify_and_request("infra", 161, 1)
        assert calls["posts"] == []

    @pytest.mark.parametrize("value", [1, [True], {"x": 1}, "maybe", "1", ""])
    def test_a_wrong_typed_verdict_fails_open(self, monkeypatch, value):
        calls = _wire(monkeypatch, model={"objection": value})
        objection.classify_and_request("infra", 161, 1)
        assert len(_requested(calls)) == 1

    def test_a_non_object_reply_fails_open_instead_of_crashing(self, monkeypatch):
        calls = _wire(monkeypatch, model=["objection"])
        out = objection.classify_and_request("infra", 161, 1)
        assert "assuming yes" in out and len(_requested(calls)) == 1


class TestWhatIsNotAnObjection:
    def test_the_model_saying_no_reviews_nothing(self, monkeypatch):
        calls = _wire(monkeypatch, comment_body="thanks, fixed in the last push",
                      model={"objection": False, "why": "acknowledges the fix"})
        out = objection.classify_and_request("infra", 161, 1)
        assert "not an objection" in out and calls["posts"] == []
        assert calls["released"] == 0

    def test_someone_elses_comment_is_not_even_asked_about(self, monkeypatch):
        calls = _wire(monkeypatch, comment_by="reviewer-human")
        out = objection.classify_and_request("infra", 161, 1)
        assert "not the author" in out and calls["asked"] == [] and calls["posts"] == []

    def test_our_own_comment_does_nothing(self, monkeypatch):
        """The failure-release comment is ours; it must not re-request us."""
        calls = _wire(monkeypatch, comment_by=ME)
        assert "our own" in objection.classify_and_request("infra", 161, 1)
        assert calls["posts"] == []

    def test_no_review_of_ours_means_nothing_to_object_to(self, monkeypatch):
        calls = _wire(monkeypatch, reviews=[
            {"user": {"login": "Copilot"}, "state": "COMMENTED", "commit_id": HEAD,
             "body": "x"}])
        assert "no review of ours" in objection.classify_and_request("infra", 161, 1)
        assert calls["asked"] == [] and calls["posts"] == []

    def test_an_approval_at_this_head_is_final(self, monkeypatch):
        """Tingyi's rule: once approved, only a new commit reviews again."""
        calls = _wire(monkeypatch, reviews=[
            {"user": {"login": ME}, "state": "APPROVED", "commit_id": HEAD, "body": "ok"}])
        assert "approval" in objection.classify_and_request("infra", 161, 1)
        assert calls["asked"] == [] and calls["posts"] == []

    def test_an_approval_at_an_OLDER_head_does_not_block(self, monkeypatch):
        calls = _wire(monkeypatch, reviews=[
            {"user": {"login": ME}, "state": "APPROVED", "commit_id": "0ld", "body": "ok"},
            {"user": {"login": ME}, "state": "COMMENTED", "commit_id": HEAD, "body": "f"}])
        objection.classify_and_request("infra", 161, 1)
        assert len(_requested(calls)) == 1

    @pytest.mark.parametrize("draft,state", [(True, "open"), (False, "closed")])
    def test_a_draft_or_closed_pr_is_left_alone(self, monkeypatch, draft, state):
        calls = _wire(monkeypatch, draft=draft, state=state)
        assert "not open" in objection.classify_and_request("infra", 161, 1)
        assert calls["posts"] == []


class TestTheEntryPoint:
    def test_a_crash_alerts_and_exits_1(self, monkeypatch):
        monkeypatch.setattr(objection, "classify_and_request",
                            lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
        alerts = []
        monkeypatch.setattr(objection.notify, "alert", alerts.append)
        monkeypatch.setattr(objection.sys, "argv", ["x", "infra", "161", "1"])
        with pytest.raises(SystemExit) as e:
            objection.main()
        assert e.value.code == 1 and "boom" in alerts[0]

    def test_a_verdict_exits_0(self, monkeypatch, capsys):
        monkeypatch.setattr(objection, "classify_and_request", lambda *a: "not an objection")
        monkeypatch.setattr(objection.sys, "argv", ["x", "infra", "161", "1"])
        objection.main()
        assert "[objection] infra#161" in capsys.readouterr().out

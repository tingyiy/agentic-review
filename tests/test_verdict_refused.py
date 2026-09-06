"""When GitHub refuses the VERDICT but not the review.

Measured on this repository's own first hosted self-review: a clean diff, 0
findings, an APPROVE — and a 422, because the workflow's own GITHUB_TOKEN is a
GitHub App token and an App may not approve a pull request. The fallback only
knew the "your own pull request" wording, so the whole review was lost to a
crash on the one shape a clean PR always takes.
"""
import io
import urllib.error

import pytest

from agentic_review import review as pr


def _wire(monkeypatch, refusal_body):
    posted = []

    def gh(path, method="GET", body=None, accept=""):
        if method == "POST" and path.endswith("/reviews"):
            ev = (body or {}).get("event")
            posted.append((ev, (body or {}).get("body", "")))
            if ev != "COMMENT":
                raise urllib.error.HTTPError(
                    path, 422, "Unprocessable Entity", {},
                    io.BytesIO(refusal_body.encode()))
        return "[]"
    monkeypatch.setattr(pr, "gh", gh)
    # Returns the ids it withdrew; the fallback iterates them, as a real
    # COMMENT does — a refused approval leaves an older block standing
    # otherwise.
    monkeypatch.setattr(pr, "_withdraw_stale_approval", lambda *a: [])
    return posted


class TestTheFindingsSurviveARefusedVerdict:
    @pytest.mark.parametrize("refusal", [
        # The real App response: the generic wrapper is IGNORED and the nested
        # reason is what matches, so an unrelated 422 wearing the same wrapper
        # cannot take this path.
        '{"message":"Review cannot be submitted","errors":['
        '"GitHub Apps are not permitted to approve pull requests"]}',
        '{"message":"Can not approve your own pull request"}',
        '{"message":"Unprocessable Entity","errors":["cannot approve"]}',
    ])
    def test_it_falls_back_to_a_comment(self, monkeypatch, refusal):
        posted = _wire(monkeypatch, refusal)
        out = pr.post_review("app", 1, "APPROVE", "body")
        assert [ev for ev, _ in posted] == ["APPROVE", "COMMENT"]
        assert out.startswith("COMMENT (approve refused")

    def test_it_names_which_refusal_it_hit(self, monkeypatch):
        """A clean review that silently became a comment reads as an approval
        that never happened; the log has to say which verdict was withheld."""
        _wire(monkeypatch, '{"errors":["GitHub Apps are not permitted to approve"]}')
        assert "this token may not set a verdict" in pr.post_review(
            "app", 1, "APPROVE", "body")

    def test_the_own_pr_wording_is_still_recognised(self, monkeypatch):
        _wire(monkeypatch, '{"message":"Can not approve your own pull request"}')
        assert "own PR" in pr.post_review("app", 1, "APPROVE", "body")

    def test_the_generic_wrapper_alone_is_not_a_refusal(self, monkeypatch):
        """`Review cannot be submitted` is GitHub's catch-all wrapper. Matching
        it would downgrade a stale head or an over-long body to a comment —
        the silent downgrade this guard exists to prevent. Copilot's finding
        on the PR that added it."""
        _wire(monkeypatch, '{"message":"Review cannot be submitted",'
                           '"errors":["head sha can\'t be blank"]}')
        with pytest.raises(urllib.error.HTTPError):
            pr.post_review("app", 1, "REQUEST_CHANGES", "body")

    def test_an_unrelated_422_still_raises(self, monkeypatch):
        """422 is GitHub's catch-all. Swallowing every one of them downgraded a
        blocking verdict AND printed a fabricated reason for it."""
        _wire(monkeypatch, '{"message":"body is too long"}')
        with pytest.raises(urllib.error.HTTPError):
            pr.post_review("app", 1, "REQUEST_CHANGES", "body")

    def test_a_comment_that_is_refused_still_raises(self, monkeypatch):
        """There is no verdict left to drop, so the failure is real."""
        def gh(path, method="GET", body=None, accept=""):
            if method == "POST":
                raise urllib.error.HTTPError(
                    path, 422, "x", {}, io.BytesIO(b'{"message":"cannot approve"}'))
            return "[]"
        monkeypatch.setattr(pr, "gh", gh)
        with pytest.raises(urllib.error.HTTPError):
            pr.post_review("app", 1, "COMMENT", "body")


class TestTheFallbackCommentDoesNotClaimAnApproval:
    """The approval body opens "**What this approval is.**". Falling back with
    it unchanged posted a comment claiming an approval GitHub had just refused
    to record — the false-clean verdict this module keeps being bitten by,
    arriving through the one door nobody had checked. Copilot's finding."""

    APPROVAL = ("### AI review — no findings\n\nReviewed for correctness.\n\n"
                "**What this approval is.** An agent read the change at `abc1234`.")

    def _fallback_body(self, monkeypatch):
        posted = _wire(monkeypatch,
                       '{"errors":["GitHub Apps are not permitted to approve"]}')
        pr.post_review("app", 1, "APPROVE", self.APPROVAL)
        return next(text for ev, text in posted if ev == "COMMENT")

    def test_it_says_the_verdict_was_not_recorded(self, monkeypatch):
        text = self._fallback_body(monkeypatch)
        # "an", because the verdict is interpolated and the article has to
        # follow it — the banner read "a `approve`" on this repo's own PRs,
        # which is the wording every self-review shows.
        assert "would not record an `approve`" in text
        assert "Nothing here has been recorded as an approval" in text

    def test_the_approval_prose_is_taken_back_out(self, monkeypatch):
        text = self._fallback_body(monkeypatch)
        assert "**What this approval is.**" not in text
        assert "### AI review — no findings\n" not in text

    def test_the_findings_themselves_survive(self, monkeypatch):
        posted = _wire(monkeypatch, '{"message":"Can not approve your own pull request"}')
        pr.post_review("app", 1, "REQUEST_CHANGES",
                       "### AI review\n\n🔴 **a real defect** — detail")
        text = next(t for ev, t in posted if ev == "COMMENT")
        assert "a real defect" in text and "would not record a `request changes`" in text
        # And not "recorded as an approval": under a refused REQUEST_CHANGES
        # what the reader needs is that nothing is blocking them.
        assert "Nothing here blocks the merge." in text
        assert "recorded as an approval" not in text


class TestTheFallbackReconcilesLikeARealComment:
    """A refused APPROVE becomes a COMMENT, and a COMMENT is exactly the state
    that leaves an older verdict of ours standing. The fallback returned early
    and skipped the reconciliation a real comment gets."""

    def test_it_withdraws_our_own_stale_verdict(self, monkeypatch):
        withdrawn = []
        posted = _wire(monkeypatch,
                       '{"errors":["GitHub Apps are not permitted to approve"]}')
        monkeypatch.setattr(pr, "_withdraw_stale_approval",
                            lambda repo, n, sha: withdrawn.append(sha) or ["r1"])
        pr.post_review("app", 1, "APPROVE", "body", head_sha="a" * 40)
        assert [ev for ev, _ in posted] == ["APPROVE", "COMMENT"]
        assert withdrawn == ["a" * 40]


class TestARefusedApprovalStillClearsOurOwnBlock:
    """tingyiy/agentic-review#10, 2026-09-05: round one blocked with a 🔴,
    round two was clean — and the PR stayed CHANGES_REQUESTED. The fallback
    withdrew a stale approval but never dismissed a stale block, and
    `_dismiss_stale_block` returns early on any event that is not "COMMENT", so
    handing it the refused "APPROVE" did nothing. On a token that may not
    approve — every GitHub-hosted run — nothing could ever clear the block."""

    def _wire_dismissal(self, monkeypatch, refusal):
        seen = {}
        monkeypatch.setattr(pr, "_withdraw_stale_approval", lambda *a: [])
        monkeypatch.setattr(
            pr, "_dismiss_stale_block",
            lambda repo, prn, event, head, trunc, unread=(), pr_files=():
            seen.update(event=event, unread=list(unread),
                        pr_files=list(pr_files)) or [])
        _wire(monkeypatch, refusal)
        return seen

    def test_the_fallback_asks_to_dismiss_as_a_COMMENT(self, monkeypatch):
        seen = self._wire_dismissal(
            monkeypatch, '{"errors":["GitHub Apps are not permitted to approve"]}')
        pr.post_review("app", 1, "APPROVE", "body")
        assert seen.get("event") == "COMMENT", (
            "the refused verdict was passed through, and the dismissal declines "
            "every event that is not COMMENT")

    def test_the_fallback_forwards_what_was_not_read(self, monkeypatch):
        """The seam that moved: `post_review` takes `unread` and hands it on.
        Without this, dropping the argument leaves every test green while a
        truncated review clears a block about a file it never opened."""
        seen = self._wire_dismissal(
            monkeypatch, '{"errors":["GitHub Apps are not permitted to approve"]}')
        pr.post_review("app", 1, "APPROVE", "body", unread=["data/huge.jsonl"],
                       pr_files=["Makefile"])
        assert seen.get("unread") == ["data/huge.jsonl"]
        # BOTH arguments: `_cited_files` needs the second to know a cited token
        # is a file at all, and without it the shape test comes back.
        assert seen.get("pr_files") == ["Makefile"]

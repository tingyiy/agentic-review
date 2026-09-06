"""SCRUM-1265: what a truncated page walk is allowed to conclude.

`_paged` stops at a fuse, and these endpoints are oldest-first, so what is
missing is the NEWEST items — which is where a rebuttal lands and which every
skip below reasons from. `_since_note` already declined to anchor on a
truncated list; the readers that can SKIP A REVIEW ENTIRELY did not.

The rule these tests fix in place: when the history is incomplete, pay for a
review rather than skip one, and do not tell the model it has already said
something it may never have been shown.
"""
import json

import pytest

from agentic_review import review as pr


def _paged_result(items, truncated):
    out = pr._Paged(items)
    out.truncated = truncated
    return out


class TestAReplyMayBeHidingInWhatWasNotFetched:
    """`_someone_replied_since` returning False makes `_already_reviewed` skip
    the review. Answering "nobody replied" from a list missing its newest items
    is a review that never happens — the worst outcome this module has."""

    def _wire(self, monkeypatch, truncated):
        monkeypatch.setattr(pr, "_me", lambda: "bot")
        monkeypatch.setattr(pr, "_paged",
                            lambda path, **kw: _paged_result([], truncated))

    def test_a_truncated_thread_counts_as_a_reply(self, monkeypatch):
        self._wire(monkeypatch, truncated=True)
        assert pr._someone_replied_since("r", 1, "2026-01-01T00:00:00Z") is True

    def test_a_complete_thread_still_answers_honestly(self, monkeypatch):
        self._wire(monkeypatch, truncated=False)
        assert pr._someone_replied_since("r", 1, "2026-01-01T00:00:00Z") is False


class TestTheSkipWillNotRunOnAStaleNewestReview:
    """Every skip in `_already_reviewed` reasons from our reviews — the
    approved-at-this-commit test, `last_at`, and the diff fingerprint on
    `last`. A truncated list is missing the newest of them, so all three are
    answering about a review that may not be the last."""

    REVIEWS = [{"user": {"login": "bot"}, "commit_id": "a" * 40,
                "submitted_at": "2026-01-01T00:00:00Z",
                "body": "It read `aaaaaaa`."}]

    def test_a_truncated_review_list_reviews_again(self, monkeypatch):
        monkeypatch.setattr(pr, "_me", lambda: "bot")
        assert pr._already_reviewed(
            "r", 1, "a" * 40, "diff",
            revs=_paged_result(self.REVIEWS, True)) is None

    def test_a_complete_one_still_skips(self, monkeypatch):
        """The half that must not break: the skip is what stops a re-request
        re-running minutes of work for the same answer."""
        monkeypatch.setattr(pr, "_me", lambda: "bot")
        monkeypatch.setattr(pr, "_someone_replied_since", lambda *a: False)
        why = pr._already_reviewed("r", 1, "a" * 40, "diff",
                                   revs=_paged_result(self.REVIEWS, False))
        assert why and "already has a review" in why


class TestTheModelIsNotScoldedOverAHistoryItNeverSaw:
    def _conversation(self, monkeypatch, truncated):
        items = [{"user": {"login": "octocat"}, "body": "a point",
                  "created_at": "2026-01-01T00:00:00Z"}]
        monkeypatch.setattr(pr, "_paged",
                            lambda path, **kw: _paged_result(
                                items if "issues" in path else [], truncated))
        return pr.conversation("r", 1)

    def test_an_incomplete_history_is_not_a_full_record(self, monkeypatch):
        out = self._conversation(monkeypatch, truncated=True)
        assert "INCOMPLETE" in out
        assert "do not repeat yourself" not in out

    def test_a_complete_one_keeps_the_instruction(self, monkeypatch):
        out = self._conversation(monkeypatch, truncated=False)
        assert "do not repeat yourself" in out and "INCOMPLETE" not in out


class TestADeletionKeepsItsPath:
    """`pr_diff` recorded the `diff --git a/x b/x` header as the path when
    `+++ b/` is `/dev/null`, so a deletion large enough to be excluded could
    not be matched against a real path by anything downstream — it fell out of
    the since-list and the touched-paths set."""

    DELETION = ("diff --git a/gone.ts b/gone.ts\n--- a/gone.ts\n"
                "+++ /dev/null\n@@ -1,2 +0,0 @@\n-one\n-two\n")

    def test_an_excluded_deletion_is_named_by_its_path(self, monkeypatch):
        big = ("diff --git a/src/big.ts b/src/big.ts\n--- a/src/big.ts\n"
               "+++ b/src/big.ts\n@@\n" + "+x\n" * 400)
        monkeypatch.setattr(pr, "gh", lambda *a, **k: big + self.DELETION)
        monkeypatch.setattr(pr, "MAX_DIFF", len(big) - 100)
        monkeypatch.setattr(pr, "MAX_PASSES", 1)
        _, excluded, _ = pr.pr_diff("repo", 1)
        assert excluded == ["gone.ts"], "the header was recorded, not the path"

    def test_the_path_is_usable_downstream(self, monkeypatch):
        monkeypatch.setattr(pr, "gh", lambda *a, **k: self.DELETION)
        diff, _, _ = pr.pr_diff("repo", 1)
        assert "gone.ts" in pr._diff_paths_with_deletions(diff.full)


class TestAnEmptyTruncatedHistoryStillSaysSo:
    """The `if not out: return ""` guard ran before the incomplete branch, so a
    thread whose surviving items all had empty bodies — bare APPROVEs — showed
    the model nothing and read as a clean empty conversation. That is the
    silent truncation this change exists to end, one branch earlier."""

    def test_nothing_survived_but_the_history_is_incomplete(self, monkeypatch):
        monkeypatch.setattr(pr, "_paged", lambda path, **kw: _paged_result(
            [{"user": {"login": "octocat"}, "body": "",
              "created_at": "2026-01-01T00:00:00Z"}], True))
        out = pr.conversation("r", 1)
        assert "INCOMPLETE" in out and "unknown rather than as" in out

    def test_a_genuinely_empty_thread_still_says_nothing(self, monkeypatch):
        monkeypatch.setattr(pr, "_paged", lambda path, **kw: _paged_result([], False))
        assert pr.conversation("r", 1) == ""


class TestWhichEndWentMissing:
    """The budget fills NEWEST-first, so a dropped item is an OLD one — the
    opposite of a truncated page walk, where the newest are gone. One message
    for both told the model its newest exchange was missing when it was present,
    inviting it to hedge about a rebuttal in front of it."""

    def _items(self, n, body="a point"):
        return [{"user": {"login": "octocat"}, "body": f"{body} {i}",
                 "created_at": f"2026-01-{i + 1:02d}T00:00:00Z"} for i in range(n)]

    def test_a_budget_drop_says_the_OLDEST_went(self, monkeypatch):
        monkeypatch.setattr(pr, "CONVERSATION_BUDGET", 40)
        monkeypatch.setattr(pr, "_paged", lambda path, **kw: _paged_result(
            self._items(6) if "issues" in path else [], False))
        out = pr.conversation("r", 1)
        assert "oldest\ndropped for length" in out
        assert "NEWEST items that did not fit" not in out
        # And it keeps the full instruction: the newest exchange IS present.
        assert "Do not repeat yourself" in out

    def test_a_truncated_walk_still_says_the_NEWEST_went(self, monkeypatch):
        monkeypatch.setattr(pr, "_paged", lambda path, **kw: _paged_result(
            self._items(2) if "issues" in path else [], True))
        out = pr.conversation("r", 1)
        assert "NEWEST items that did not fit" in out
        assert "Do not repeat yourself" not in out

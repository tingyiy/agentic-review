"""What a re-review is told about the round before it.

The diff is always the WHOLE pull request, so round two is handed the files it
already reviewed and the files that arrived afterwards in one undifferentiated
block. `changed_since_last_review` is the line that separates them.
"""
import json

import pytest

from agentic_review import review as pr


REVIEWED = ("### AI review\n\nsomething\n\n_Automated review — agentic-review "
            "(model) with read access. It read `aaaaaaa`. It did not run the tests._")
APPROVED = ("### AI review — no findings\n\n**What this approval is.** An agent "
            "read the change at `aaaaaaa`, explored the code around it.")


def _wire(monkeypatch, reviews, files, *, me="bot"):
    """GitHub answering the two calls the function makes, and nothing else."""
    calls = []

    def gh(path, method="GET", body=None, accept=""):
        calls.append(path)
        if "/reviews" in path:
            return json.dumps(reviews)
        if "/compare/" in path:
            return json.dumps({"files": files})
        raise AssertionError(f"unexpected call: {path}")

    monkeypatch.setattr(pr, "gh", gh)
    monkeypatch.setattr(pr, "_me", lambda: me)
    return calls


class TestItNamesWhatIsNew:
    def test_only_the_files_that_moved_since_the_last_review(self, monkeypatch):
        _wire(monkeypatch,
              [{"user": {"login": "bot"}, "submitted_at": "2026-01-01T00:00:00Z",
                "body": REVIEWED, "commit_id": "a" * 40}],
              [{"filename": "c.ts", "status": "added"},
               {"filename": "d.ts", "status": "added"}])
        out = pr.changed_since_last_review("repo", 1, "b" * 40,
                                           ["a.ts", "b.ts", "c.ts", "d.ts"])
        assert "`c.ts` (added)" in out and "`d.ts` (added)" in out
        # The files of round one are NOT listed as new — that is the whole point.
        assert "`a.ts`" not in out and "`b.ts`" not in out
        assert "NEW SINCE YOUR LAST REVIEW at `aaaaaaa`" in out

    def test_it_does_not_tell_the_model_to_skip_the_rest(self, monkeypatch):
        """A later commit can break code that was correct when it was read, so
        the round still reviews everything; this line only orders attention."""
        _wire(monkeypatch,
              [{"user": {"login": "bot"}, "submitted_at": "2026-01-01T00:00:00Z",
                "body": REVIEWED}],
              [{"filename": "c.ts", "status": "added"}])
        out = pr.changed_since_last_review("repo", 1, "b" * 40, ["c.ts"])
        assert "re-check what it could have broken" in out

    def test_a_base_merge_is_not_the_authors_new_work(self, monkeypatch):
        """`compare` between two heads of the branch also carries what arrived
        from main. Those files are absent from the PR's own diff, so the
        intersection with the PR's paths is what drops them."""
        _wire(monkeypatch,
              [{"user": {"login": "bot"}, "submitted_at": "2026-01-01T00:00:00Z",
                "body": REVIEWED}],
              [{"filename": "c.ts", "status": "modified"},
               {"filename": "vendor/from-main.ts", "status": "modified"}])
        out = pr.changed_since_last_review("repo", 1, "b" * 40, ["a.ts", "c.ts"])
        assert "`c.ts`" in out
        assert "from-main" not in out

    def test_nothing_of_this_prs_own_moved_says_nothing(self, monkeypatch):
        _wire(monkeypatch,
              [{"user": {"login": "bot"}, "submitted_at": "2026-01-01T00:00:00Z",
                "body": REVIEWED}],
              [{"filename": "vendor/from-main.ts", "status": "modified"}])
        assert pr.changed_since_last_review("repo", 1, "b" * 40, ["a.ts"]) == ""

    def test_the_list_is_capped_and_says_how_many_it_cut(self, monkeypatch):
        names = [f"f{i:03d}.ts" for i in range(pr.MAX_SINCE_PATHS + 5)]
        _wire(monkeypatch,
              [{"user": {"login": "bot"}, "submitted_at": "2026-01-01T00:00:00Z",
                "body": REVIEWED}],
              [{"filename": n, "status": "modified"} for n in names])
        out = pr.changed_since_last_review("repo", 1, "b" * 40, names)
        assert "and 5 more" in out
        assert out.count("(modified)") == pr.MAX_SINCE_PATHS


class TestWhichShaItComparesFrom:
    def test_the_body_wins_over_commit_id(self, monkeypatch):
        """GitHub stamps `commit_id` with the head at POST time, so a push that
        lands mid-run re-attributes a review to code it never read. The body
        says which snapshot was actually looked at."""
        calls = _wire(monkeypatch,
                      [{"user": {"login": "bot"},
                        "submitted_at": "2026-01-01T00:00:00Z",
                        "body": REVIEWED, "commit_id": "f" * 40}],
                      [{"filename": "c.ts", "status": "added"}])
        pr.changed_since_last_review("repo", 1, "b" * 40, ["c.ts"])
        compare = [c for c in calls if "/compare/" in c][0]
        assert "aaaaaaa..." in compare
        assert "f" * 40 not in compare

    def test_an_approval_body_carries_the_sha_too(self, monkeypatch):
        calls = _wire(monkeypatch,
                      [{"user": {"login": "bot"},
                        "submitted_at": "2026-01-01T00:00:00Z", "body": APPROVED}],
                      [{"filename": "c.ts", "status": "added"}])
        pr.changed_since_last_review("repo", 1, "b" * 40, ["c.ts"])
        assert "aaaaaaa..." in [c for c in calls if "/compare/" in c][0]

    def test_commit_id_is_the_fallback_for_an_older_review(self, monkeypatch):
        calls = _wire(monkeypatch,
                      [{"user": {"login": "bot"},
                        "submitted_at": "2026-01-01T00:00:00Z",
                        "body": "a review from before the read-at line",
                        "commit_id": "c" * 40}],
                      [{"filename": "c.ts", "status": "added"}])
        pr.changed_since_last_review("repo", 1, "b" * 40, ["c.ts"])
        assert "c" * 40 in [c for c in calls if "/compare/" in c][0]

    def test_the_newest_of_our_reviews_is_the_anchor(self, monkeypatch):
        calls = _wire(monkeypatch, [
            {"user": {"login": "bot"}, "submitted_at": "2026-01-01T00:00:00Z",
             "body": "It read `1111111`."},
            {"user": {"login": "bot"}, "submitted_at": "2026-01-03T00:00:00Z",
             "body": "It read `3333333`."},
            {"user": {"login": "bot"}, "submitted_at": "2026-01-02T00:00:00Z",
             "body": "It read `2222222`."},
        ], [{"filename": "c.ts", "status": "added"}])
        pr.changed_since_last_review("repo", 1, "b" * 40, ["c.ts"])
        assert "3333333..." in [c for c in calls if "/compare/" in c][0]

    def test_somebody_elses_review_is_not_ours(self, monkeypatch):
        """Copilot reviews these PRs too. Comparing from ITS commit would
        describe a round this reviewer never had."""
        calls = _wire(monkeypatch,
                      [{"user": {"login": "copilot"},
                        "submitted_at": "2026-01-01T00:00:00Z", "body": REVIEWED}],
                      [{"filename": "c.ts", "status": "added"}])
        assert pr.changed_since_last_review("repo", 1, "b" * 40, ["c.ts"]) == ""
        assert not [c for c in calls if "/compare/" in c]


class TestWhenThereIsNothingToSay:
    def test_a_first_review_has_no_since(self, monkeypatch):
        calls = _wire(monkeypatch, [], [])
        assert pr.changed_since_last_review("repo", 1, "b" * 40, ["a.ts"]) == ""
        assert not [c for c in calls if "/compare/" in c]

    def test_the_last_review_read_this_very_head(self, monkeypatch):
        """The abbreviated sha prefixes the head — there is no interval."""
        calls = _wire(monkeypatch,
                      [{"user": {"login": "bot"},
                        "submitted_at": "2026-01-01T00:00:00Z",
                        "body": "It read `aaaaaaa`."}], [])
        assert pr.changed_since_last_review("repo", 1, "a" * 40, ["a.ts"]) == ""
        assert not [c for c in calls if "/compare/" in c]

    @pytest.mark.parametrize("failing", ["/reviews", "/compare/"])
    def test_it_never_raises_when_github_does(self, monkeypatch, failing):
        """A force-push makes the old sha unreachable and `compare` 404s. Less
        context is worth having; a review that did not happen is not."""
        def gh(path, method="GET", body=None, accept=""):
            if failing in path:
                raise RuntimeError("boom")
            return json.dumps([{"user": {"login": "bot"},
                                "submitted_at": "2026-01-01T00:00:00Z",
                                "body": REVIEWED}])
        monkeypatch.setattr(pr, "gh", gh)
        monkeypatch.setattr(pr, "_me", lambda: "bot")
        assert pr.changed_since_last_review("repo", 1, "b" * 40, ["a.ts"]) == ""


    def test_a_payload_that_is_not_a_list_of_objects_is_not_fatal(self, monkeypatch):
        """The `except` only wraps the fetch, so a shape surprise inside the
        loop would escape it and take the whole review down."""
        monkeypatch.setattr(pr, "_me", lambda: "bot")
        monkeypatch.setattr(pr, "gh", lambda path, **kw:
                            json.dumps(["not an object"]) if "/reviews" in path
                            else json.dumps("not an object"))
        assert pr.changed_since_last_review("repo", 1, "b" * 40, ["a.ts"]) == ""

    def test_a_compare_whose_files_are_not_objects_is_not_fatal(self, monkeypatch):
        _wire(monkeypatch,
              [{"user": {"login": "bot"}, "submitted_at": "2026-01-01T00:00:00Z",
                "body": REVIEWED}],
              ["c.ts"])
        assert pr.changed_since_last_review("repo", 1, "b" * 40, ["c.ts"]) == ""

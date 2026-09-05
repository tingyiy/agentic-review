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


class TestTheWordingIsHonest:
    def test_it_does_not_claim_the_rest_was_reviewed(self, monkeypatch):
        """A partial review leaves changed files unopened and says so in its
        own caveat. "Everything else you have already reviewed" would
        contradict that warning on the very next round."""
        _wire(monkeypatch,
              [{"user": {"login": "bot"}, "submitted_at": "2026-01-01T00:00:00Z",
                "body": REVIEWED}],
              [{"filename": "c.ts", "status": "added"}])
        out = pr.changed_since_last_review("repo", 1, "b" * 40, ["c.ts"])
        assert "already present at that review" in out
        assert "already reviewed" not in out

    def test_a_compare_at_the_api_cap_stops_saying_only(self, monkeypatch):
        """GitHub's compare caps `files` at 300 with no flag saying it did.
        "the only parts that have moved" would then point the reviewer AWAY
        from real new work."""
        files = [{"filename": f"f{i:04d}.ts", "status": "modified"}
                 for i in range(pr.COMPARE_FILE_CAP)]
        _wire(monkeypatch,
              [{"user": {"login": "bot"}, "submitted_at": "2026-01-01T00:00:00Z",
                "body": REVIEWED}],
              files)
        out = pr.changed_since_last_review(
            "repo", 1, "b" * 40, [f["filename"] for f in files])
        assert "AT LEAST" in out and "the only parts" not in out

    def test_below_the_cap_it_still_says_only(self, monkeypatch):
        files = [{"filename": f"f{i:04d}.ts", "status": "modified"}
                 for i in range(pr.COMPARE_FILE_CAP - 1)]
        _wire(monkeypatch,
              [{"user": {"login": "bot"}, "submitted_at": "2026-01-01T00:00:00Z",
                "body": REVIEWED}],
              files)
        out = pr.changed_since_last_review(
            "repo", 1, "b" * 40, [f["filename"] for f in files])
        assert "the only parts" in out and "AT LEAST" not in out


class TestADeletedFileIsAChange:
    def test_diff_paths_with_deletions_keeps_the_removed_file(self):
        """`_diff_paths` reads `+++ b/` only, and a deletion's is `/dev/null`
        — so intersecting against it drops every removal silently."""
        diff = ("diff --git a/gone.ts b/gone.ts\n--- a/gone.ts\n+++ /dev/null\n"
                "diff --git a/kept.ts b/kept.ts\n--- a/kept.ts\n+++ b/kept.ts\n")
        assert pr._diff_paths(diff) == {"kept.ts"}
        assert pr._diff_paths_with_deletions(diff) == {"gone.ts", "kept.ts"}

    def test_a_new_file_has_no_a_side_to_confuse_it(self):
        diff = "diff --git a/new.ts b/new.ts\n--- /dev/null\n+++ b/new.ts\n"
        assert pr._diff_paths_with_deletions(diff) == {"new.ts"}

    def test_a_file_removed_since_the_last_review_is_listed(self, monkeypatch):
        _wire(monkeypatch,
              [{"user": {"login": "bot"}, "submitted_at": "2026-01-01T00:00:00Z",
                "body": REVIEWED}],
              [{"filename": "gone.ts", "status": "removed"}])
        out = pr.changed_since_last_review("repo", 1, "b" * 40, ["gone.ts"])
        assert "`gone.ts` (removed)" in out


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

    def test_the_footer_wins_over_a_finding_that_quotes_it(self, monkeypatch):
        """The footer is the LAST line of the body and the findings are above
        it — and a finding may quote the phrase, as one on this very PR did."""
        calls = _wire(monkeypatch,
                      [{"user": {"login": "bot"},
                        "submitted_at": "2026-01-01T00:00:00Z",
                        "body": "a finding quoting It read `9999999` in its "
                                "detail\n\n_Automated review. It read `aaaaaaa`._"}],
                      [{"filename": "c.ts", "status": "added"}])
        pr.changed_since_last_review("repo", 1, "b" * 40, ["c.ts"])
        assert "aaaaaaa..." in [c for c in calls if "/compare/" in c][0]

    def test_a_review_past_the_first_page_is_still_ours(self, monkeypatch):
        """These endpoints return OLDEST first, so the newest review — the one
        this anchors on — is on the LAST page. `per_page=100` is a ceiling."""
        page1 = [{"user": {"login": "bot"}, "submitted_at": "2026-01-01T00:00:00Z",
                  "body": "It read `1111111`."}] * 100
        page2 = [{"user": {"login": "bot"}, "submitted_at": "2026-02-01T00:00:00Z",
                  "body": "It read `2222222`."}]
        calls = []

        def gh(path, method="GET", body=None, accept=""):
            calls.append(path)
            if "/reviews" in path:
                return json.dumps(page1 if "&page=1" in path else page2)
            return json.dumps({"files": [{"filename": "c.ts", "status": "added"}]})
        monkeypatch.setattr(pr, "gh", gh)
        monkeypatch.setattr(pr, "_me", lambda: "bot")
        pr.changed_since_last_review("repo", 1, "b" * 40, ["c.ts"])
        assert "2222222..." in [c for c in calls if "/compare/" in c][0]

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


class TestItDoesNotRefetchWhatMainAlreadyHas:
    def test_reviews_handed_in_are_not_fetched_again(self, monkeypatch):
        """`main` reads this endpoint for the nothing-new guard minutes
        earlier; asking twice is two round-trips for identical data."""
        calls = []

        def gh(path, method="GET", body=None, accept=""):
            calls.append(path)
            return json.dumps({"files": [{"filename": "c.ts", "status": "added"}]})
        monkeypatch.setattr(pr, "gh", gh)
        monkeypatch.setattr(pr, "_me", lambda: "bot")
        out = pr.changed_since_last_review(
            "repo", 1, "b" * 40, ["c.ts"],
            revs=[{"user": {"login": "bot"},
                   "submitted_at": "2026-01-01T00:00:00Z", "body": REVIEWED}])
        assert "`c.ts` (added)" in out
        assert not [c for c in calls if "/reviews" in c]


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

    def test_a_null_payload_is_not_fatal(self, monkeypatch):
        """`main` calls this inline. A shape surprise anywhere in the parsing
        would abort a whole review to save a line of prompt."""
        monkeypatch.setattr(pr, "_me", lambda: "bot")
        monkeypatch.setattr(pr, "gh", lambda path, **kw: json.dumps(None))
        assert pr.changed_since_last_review("repo", 1, "b" * 40, ["a.ts"]) == ""

    def test_a_truthy_non_object_user_is_not_fatal(self, monkeypatch):
        monkeypatch.setattr(pr, "_me", lambda: "bot")
        monkeypatch.setattr(pr, "gh", lambda path, **kw: json.dumps(
            [{"user": "octocat", "submitted_at": "2026-01-01T00:00:00Z",
              "body": REVIEWED}]))
        assert pr.changed_since_last_review("repo", 1, "b" * 40, ["a.ts"]) == ""

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


class TestTheWiringInMain:
    """The helper was tested and the call site was not — the gap this repo's
    own reviewer named on the PR that added it, and the one the rules call
    'a test that calls below the change'."""

    DELETION = ("diff --git a/gone.ts b/gone.ts\n--- a/gone.ts\n"
                "+++ /dev/null\n@@\n-x\n")

    def _prompt(self, monkeypatch, reviews, files, diff=None):
        seen = {}

        def gh(path, method="GET", body=None, accept=""):
            if "/compare/" in path:
                return json.dumps({"files": files})
            if "/reviews" in path:
                return json.dumps(reviews)
            if path.endswith("/pulls/7"):
                return json.dumps(
                    {"draft": False, "state": "open", "merged": False,
                     "title": "SCRUM-1 x", "user": {"login": "someone"},
                     "head": {"sha": "b" * 40}})
            return json.dumps([])

        monkeypatch.setattr(pr, "gh", gh)
        monkeypatch.setattr(pr, "_me", lambda: "bot")
        monkeypatch.setattr(pr, "pr_diff", lambda *a: (
            diff or "diff --git a/c.ts b/c.ts\n--- a/c.ts\n+++ b/c.ts\n@@\n+x\n",
            False, 0))
        monkeypatch.setattr(pr, "_already_reviewed", lambda *a, **k: "")
        monkeypatch.setattr(pr, "conversation", lambda *a: "")
        monkeypatch.setattr(pr, "build_context", lambda *a: "")
        monkeypatch.setattr(pr, "commit_messages", lambda *a: [])
        monkeypatch.setattr(pr, "checkout", lambda *a: None)
        monkeypatch.setattr(pr.ctx, "expand_hunks", lambda d, w, **k: d)
        monkeypatch.setattr(pr, "_revise", lambda f, w, r: (f, []))
        monkeypatch.setattr(pr.checks, "run_all", lambda *a, **k: [])
        monkeypatch.setattr(pr, "post_review", lambda *a, **k: "APPROVE")
        monkeypatch.setattr(pr, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(
            pr, "review_findings",
            lambda prompt, work, repo="": seen.setdefault("prompt", prompt) and [])
        monkeypatch.setattr(pr.sys, "argv", ["pr-review", "app", "7"])
        monkeypatch.delenv("DRY", raising=False)
        pr.main()
        return seen.get("prompt", "")

    def test_the_block_reaches_the_model(self, monkeypatch):
        prompt = self._prompt(
            monkeypatch,
            [{"user": {"login": "bot"}, "submitted_at": "2026-01-01T00:00:00Z",
              "body": REVIEWED}],
            [{"filename": "c.ts", "status": "modified"}])
        assert "NEW SINCE YOUR LAST REVIEW at `aaaaaaa`" in prompt
        assert "`c.ts` (modified)" in prompt

    def test_a_first_review_carries_no_block(self, monkeypatch):
        prompt = self._prompt(monkeypatch, [], [])
        assert "NEW SINCE YOUR LAST REVIEW" not in prompt

    def test_a_deletion_only_push_still_gets_a_block(self, monkeypatch):
        """`_diff_paths` reads `+++ b/` alone, so intersecting against it drops
        every removal — and an all-deletion round would get no block at all."""
        prompt = self._prompt(
            monkeypatch,
            [{"user": {"login": "bot"}, "submitted_at": "2026-01-01T00:00:00Z",
              "body": REVIEWED}],
            [{"filename": "gone.ts", "status": "removed"}],
            diff=self.DELETION)
        assert "`gone.ts` (removed)" in prompt


class TestATruncatedListIsNotAnAnchor:
    """`_paged` stops at a fuse, and these endpoints are oldest-first — so a
    thread long enough to hit it returns the OLDEST items and drops the newest,
    which is the one everything here anchors on."""

    def _full_pages(self, monkeypatch, cap):
        review = {"user": {"login": "bot"}, "submitted_at": "2026-01-01T00:00:00Z",
                  "body": REVIEWED}

        def gh(path, method="GET", body=None, accept=""):
            if "/reviews" in path:
                return json.dumps([review] * 100)
            return json.dumps({"files": [{"filename": "c.ts", "status": "added"}]})
        monkeypatch.setattr(pr, "gh", gh)
        monkeypatch.setattr(pr, "_me", lambda: "bot")
        monkeypatch.setattr(pr, "MAX_PAGES", cap)

    def test_paged_says_it_stopped_short(self, monkeypatch):
        self._full_pages(monkeypatch, 3)
        out = pr._paged("/repos/x/y/pulls/1/reviews")
        assert out.truncated is True and len(out) == 300

    def test_a_complete_list_is_not_marked_truncated(self, monkeypatch):
        _wire(monkeypatch, [{"user": {"login": "bot"}}], [])
        assert pr._paged("/repos/x/y/pulls/1/reviews").truncated is False

    def test_no_since_list_rather_than_one_from_a_stale_review(self, monkeypatch):
        """Naming files as new that were reviewed long ago is worse than
        saying nothing."""
        self._full_pages(monkeypatch, 2)
        assert pr.changed_since_last_review("repo", 1, "b" * 40, ["c.ts"]) == ""


class TestTheSkipReadsTheShaTheReviewSaw:
    """GitHub stamps `commit_id` with the head at POST time. A review posted
    while a push lands is recorded against a commit it never read — and keying
    the skip on that stamp meant the new commit arrived already marked done and
    was never reviewed by anything."""

    def test_a_commit_the_review_never_read_is_not_skipped(self, monkeypatch):
        monkeypatch.setattr(pr, "_me", lambda: "bot")
        monkeypatch.setattr(pr, "gh", lambda path, **kw: json.dumps(
            [{"user": {"login": "bot"}, "commit_id": "b" * 40,
              "submitted_at": "2026-01-01T00:00:00Z",
              "body": "It read `aaaaaaa`."}]))
        assert pr._already_reviewed("repo", 1, "b" * 40, "diff") is None

    def test_an_approval_stamped_on_an_unread_commit_does_not_stand(self, monkeypatch):
        monkeypatch.setattr(pr, "_me", lambda: "bot")
        monkeypatch.setattr(pr, "gh", lambda path, **kw: json.dumps(
            [{"user": {"login": "bot"}, "commit_id": "b" * 40, "state": "APPROVED",
              "submitted_at": "2026-01-01T00:00:00Z",
              "body": "read the change at `aaaaaaa`, explored the code"}]))
        assert pr._already_reviewed("repo", 1, "b" * 40, "diff") is None

    def test_the_commit_it_did_read_is_still_skipped(self, monkeypatch):
        """The half that must not break: the skip is what stops a re-request
        re-running minutes of work for the same answer."""
        monkeypatch.setattr(pr, "_me", lambda: "bot")
        monkeypatch.setattr(pr, "gh", lambda path, **kw: json.dumps(
            [{"user": {"login": "bot"}, "commit_id": "b" * 40,
              "submitted_at": "2026-01-01T00:00:00Z",
              "body": "It read `bbbbbbb`."}]))
        why = pr._already_reviewed("repo", 1, "b" * 40, "diff")
        assert why and "already has a review" in why

    def test_an_older_review_without_the_footer_still_uses_commit_id(self, monkeypatch):
        monkeypatch.setattr(pr, "_me", lambda: "bot")
        monkeypatch.setattr(pr, "gh", lambda path, **kw: json.dumps(
            [{"user": {"login": "bot"}, "commit_id": "b" * 40,
              "submitted_at": "2026-01-01T00:00:00Z", "body": "no footer here"}]))
        why = pr._already_reviewed("repo", 1, "b" * 40, "diff")
        assert why and "already has a review" in why

"""A review already given is not worth giving again — and a review NOT given
must never be skipped.

Both halves matter and they pull opposite ways. The guard exists because
`synchronize` made two wasteful paths routine:

  * a re-request while a review already exists for the same head re-runs
    minutes of agent work to reach the same answer — and, because the
    concurrency group cancels in progress, a push plus a re-request seconds
    apart also leaves a CANCELLED check that reads as a failing one;
  * update-branch is a `synchronize`, so merging main into an APPROVED PR
    re-reviewed a diff that had not changed.

The danger is the mirror image: skip too eagerly and a real change ships
unreviewed while the PR still shows a green check from an older commit. So
every test here that asserts a skip has a partner asserting the review happens.
"""
import json
import pathlib
import sys

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
from conftest import load_script  # noqa: E402

BOT = "review-bot"
SHA = "a" * 40
OTHER = "b" * 40
DIFF = "diff --git a/x b/x\n+one\n"


@pytest.fixture
def pr(monkeypatch):
    """The reviewer with its network stubbed: prior reviews in, nothing out."""
    m = load_script("pr-review")
    monkeypatch.setattr(m, "_me", lambda: BOT)

    def drive(reviews):
        monkeypatch.setattr(m, "gh", lambda path, **k: json.dumps(reviews))
        return m
    return m, drive


def _review(sha, body="", who=BOT):
    return {"user": {"login": who}, "commit_id": sha, "body": body}


class TestTheSameCommit:
    def test_a_head_that_already_has_a_review_is_skipped(self, pr):
        m, drive = pr
        drive([_review(SHA)])
        why = m._already_reviewed("r", 1, SHA, DIFF)
        assert why and SHA[:7] in why, "re-reviewed a commit already reviewed"

    def test_a_NEW_head_is_reviewed(self, pr):
        """The half that must never break: a real push gets a real review."""
        m, drive = pr
        drive([_review(OTHER)])
        assert m._already_reviewed("r", 1, SHA, DIFF) is None, (
            "skipped a commit that has never been reviewed")


class TestTheSameDiff:
    def test_update_branch_is_skipped(self, pr):
        """A new SHA carrying an identical diff — the base moved, this PR did
        not. Fingerprint is in the prior review's own body."""
        m, drive = pr
        mark = m._DIFF_MARK.format(fp=m._diff_fp(DIFF))
        drive([_review(OTHER, body="### AI review\n\n" + mark)])
        why = m._already_reviewed("r", 1, SHA, DIFF)
        assert why and "base moved" in why

    def test_a_CHANGED_diff_on_a_new_head_is_reviewed(self, pr):
        m, drive = pr
        mark = m._DIFF_MARK.format(fp=m._diff_fp(DIFF))
        drive([_review(OTHER, body=mark)])
        assert m._already_reviewed("r", 1, SHA, DIFF + "+two\n") is None, (
            "a changed diff was mistaken for an update-branch")

    def test_only_the_LATEST_review_is_consulted_for_the_diff(self, pr):
        """An older review of the same diff does not license skipping: the head
        has moved on since, and what matters is the most recent state."""
        m, drive = pr
        mark = m._DIFF_MARK.format(fp=m._diff_fp(DIFF))
        drive([_review("c" * 40, body=mark), _review(OTHER, body="no mark")])
        assert m._already_reviewed("r", 1, SHA, DIFF) is None


class TestItFailsTowardReviewing:
    def test_a_FAILED_review_does_not_count(self, pr):
        """A failed run posts nothing, so it leaves no review at that SHA. The
        re-request arrow has to keep working — that is the whole property
        `_release_review_request` exists to protect."""
        m, drive = pr
        drive([])
        assert m._already_reviewed("r", 1, SHA, DIFF) is None

    def test_somebody_ELSES_review_does_not_count(self, pr):
        m, drive = pr
        drive([_review(SHA, who="copilot-pull-request-reviewer")])
        assert m._already_reviewed("r", 1, SHA, DIFF) is None, (
            "another reviewer's approval was treated as ours"
        )

    def test_an_unreadable_reviews_list_reviews_anyway(self, pr, monkeypatch):
        """Unanswerable means review it. Skipping on an API blip would drop a
        review silently, and silence is the failure this whole file guards."""
        m, _ = pr

        def boom(path, **k):
            raise RuntimeError("502")

        monkeypatch.setattr(m, "gh", boom)
        assert m._already_reviewed("r", 1, SHA, DIFF) is None

    def test_an_unknown_identity_reviews_anyway(self, pr, monkeypatch):
        m, drive = pr
        drive([_review(SHA)])
        monkeypatch.setattr(m, "_me", lambda: "")
        assert m._already_reviewed("r", 1, SHA, DIFF) is None


class TestTheMarkIsActuallyEmitted:
    """The guard is inert unless the bodies carry the fingerprint — and an
    inert guard looks exactly like a working one until a re-review costs
    somebody ten minutes."""

    def test_the_approval_carries_it(self, pr):
        m, _ = pr
        body = m.approval_body(SHA, repo="r", diff=DIFF)
        assert m._DIFF_MARK.format(fp=m._diff_fp(DIFF)) in body

    def test_a_findings_review_carries_it(self, pr):
        m, _ = pr
        body = m.render([{"file": "x", "line": 1, "severity": "low",
                          "title": "t", "detail": "d"}],
                        False, 0, head_sha=SHA, repo="r", diff=DIFF)
        assert m._DIFF_MARK.format(fp=m._diff_fp(DIFF)) in body

    def test_the_mark_is_invisible_in_rendered_markdown(self, pr):
        """It rides in the body a human reads, so it has to be a comment."""
        m, _ = pr
        assert m._DIFF_MARK.startswith("<!--") and m._DIFF_MARK.endswith("-->")


class TestDisagreeingWithIt:
    """An author must be able to argue, and the guard nearly removed that.

    `_already_reviewed` stops a re-request on an unchanged commit from
    re-running minutes of work for the same answer. But the only way to say
    "this finding is wrong, here is why, look again" IS a re-request on an
    unchanged commit — so the guard silently made pushing an empty commit the
    only lever, which is a terrible thing to have to know.

    A reply is new information. The commit is not the only thing that can
    change.
    """

    def _revs(self, when):
        return [{"user": {"login": BOT}, "commit_id": SHA, "body": "",
                 "submitted_at": when}]

    def test_a_reply_after_the_review_earns_another_look(self, pr, monkeypatch):
        m, _ = pr
        def gh(path, **k):
            if "/reviews" in path:
                return json.dumps(self._revs("2026-08-31T10:00:00Z"))
            if "/issues/1/comments" in path:
                return json.dumps([{"user": {"login": "octocat"},
                                    "created_at": "2026-08-31T11:00:00Z"}])
            return json.dumps([])
        monkeypatch.setattr(m, "gh", gh)
        assert m._already_reviewed("r", 1, SHA, DIFF) is None, (
            "an author who explained why a finding is wrong got silence")

    def test_a_reply_UNDER_A_DIFF_LINE_counts_too(self, pr, monkeypatch):
        """Where a rebuttal to a specific finding actually lands — the same
        endpoint `conversation()` had to be taught to read for the same reason."""
        m, _ = pr
        def gh(path, **k):
            if "/reviews" in path:
                return json.dumps(self._revs("2026-08-31T10:00:00Z"))
            if "/pulls/1/comments" in path:
                return json.dumps([{"user": {"login": "octocat"},
                                    "created_at": "2026-08-31T11:00:00Z"}])
            return json.dumps([])
        monkeypatch.setattr(m, "gh", gh)
        assert m._already_reviewed("r", 1, SHA, DIFF) is None

    def test_NOTHING_new_still_skips(self, pr, monkeypatch):
        """The guard has to survive: a bare re-request on an unchanged commit
        with no reply is still the wasteful case it was built for."""
        m, _ = pr
        def gh(path, **k):
            if "/reviews" in path:
                return json.dumps(self._revs("2026-08-31T10:00:00Z"))
            return json.dumps([{"user": {"login": "octocat"},
                                "created_at": "2026-08-31T09:00:00Z"}])
        monkeypatch.setattr(m, "gh", gh)
        assert m._already_reviewed("r", 1, SHA, DIFF) is not None

    def test_OUR_OWN_comment_does_not_count_as_a_reply(self, pr, monkeypatch):
        """Ours must not argue with us.

        Testing this with our own REVIEW is toothless — `last_at` is the max of
        our reviews, so one can never be newer than it, and the mutation that
        drops the exclusion still passes. A COMMENT by us is the case that
        actually exercises it, and it is reachable: anything the bot posts on
        the thread would otherwise make every later re-request look like a
        fresh argument and defeat the guard entirely.
        """
        m, _ = pr
        def gh(path, **k):
            if "/reviews" in path:
                return json.dumps(self._revs("2026-08-31T10:00:00Z"))
            if "/issues/1/comments" in path:
                return json.dumps([{"user": {"login": BOT},
                                    "created_at": "2026-08-31T23:00:00Z"}])
            return json.dumps([])
        monkeypatch.setattr(m, "gh", gh)
        assert m._already_reviewed("r", 1, SHA, DIFF) is not None, (
            "our own comment was treated as somebody disagreeing with us")

    def test_an_unreadable_comments_list_does_not_force_a_review(self, pr, monkeypatch):
        """Unanswerable means "no reply" here, not "review it": the opposite
        would turn one flaky endpoint into a re-review on every re-request."""
        m, _ = pr
        def gh(path, **k):
            if "/reviews" in path:
                return json.dumps(self._revs("2026-08-31T10:00:00Z"))
            raise RuntimeError("502")
        monkeypatch.setattr(m, "gh", gh)
        assert m._already_reviewed("r", 1, SHA, DIFF) is not None


class TestItReadsEverywhereARebuttalLands:
    """Four endpoints and a full page, because a rebuttal that is not read is
    the same as one that was never written."""

    def _revs(self, when):
        return [{"user": {"login": BOT}, "commit_id": SHA, "body": "",
                 "submitted_at": when}]

    def test_it_PAGES_rather_than_raising_a_ceiling(self, pr, monkeypatch):
        """`per_page=100` alone is a ceiling, not pagination.

        `gh()` does not paginate and all four endpoints return OLDEST-first —
        measured on infra#134, whose /commits page starts at 16:17 and ends at
        00:30 the next day. So the newest item, which is where a rebuttal
        lands, is on the LAST page. Going 30 to 100 moved the cliff instead of
        removing it: a PR with 101 comments would have failed exactly as one
        with 31 did.

        DRIVEN, not read: this test used to assert on the source text of the
        function, which passes while behaviour changes under it and fails when
        the code merely moves — as it did the day the page loop became a shared
        helper.
        """
        m, _ = pr
        pages = []
        full = [{"user": {"login": "octocat"},
                 "created_at": "2026-08-30T01:00:00Z"}] * 100

        def gh(path, **k):
            if "/pulls/1/comments" in path:
                pages.append(path)
                return json.dumps(full if "&page=1" in path else [])
            return json.dumps([])
        monkeypatch.setattr(m, "gh", gh)
        m._someone_replied_since("r", 1, "2026-08-31T10:00:00Z")
        assert any("&page=2" in p for p in pages), "stopped at the ceiling"

    def test_it_reads_a_reply_on_the_SECOND_page(self, pr, monkeypatch):
        """The failure the ceiling left behind, driven end to end."""
        m, _ = pr
        old = [{"user": {"login": "octocat"}, "created_at": "2026-08-30T01:00:00Z"}] * 100
        new = [{"user": {"login": "octocat"}, "created_at": "2026-08-31T11:00:00Z"}]

        # `&page=1`, NOT `page=1`: the query is `?per_page=100&page=2`, and
        # "page=1" is a substring of "per_page=100" — so a stub keyed on the
        # bare fragment hands page one back for EVERY page. This test then
        # walked to the twenty-page fuse and never saw the reply it exists to
        # find; it passed only because the review list above it was empty and
        # the caller returned before ever asking.
        def gh(path, **k):
            if "/reviews" in path:
                if "&page=1" in path:
                    return json.dumps([{"user": {"login": BOT}, "commit_id": SHA,
                                        "body": "", "submitted_at": "2026-08-31T10:00:00Z"}])
                return json.dumps([])
            if "/issues/1/comments" in path:
                return json.dumps(old if "&page=1" in path else new)
            return json.dumps([])
        monkeypatch.setattr(m, "gh", gh)
        assert m._already_reviewed("r", 1, SHA, DIFF) is None, (
            "a rebuttal past the first page was invisible — the same silence "
            "this function exists to end, one threshold higher")

    def test_paging_stops_on_a_short_page(self, pr, monkeypatch):
        """Without the short-page stop this walks to the fuse on every call —
        twenty requests per endpoint, four endpoints, every re-request."""
        m, _ = pr
        calls = []

        def gh(path, **k):
            calls.append(path)
            if "/reviews" in path and "&page=1" in path:
                return json.dumps([{"user": {"login": BOT}, "commit_id": SHA,
                                    "body": "", "submitted_at": "2026-08-31T10:00:00Z"}])
            return json.dumps([])
        monkeypatch.setattr(m, "gh", gh)
        m._already_reviewed("r", 1, SHA, DIFF)
        assert all("page=2" not in c for c in calls), "kept paging past a short page"

    def test_a_COMMIT_MESSAGE_rebuttal_counts(self, pr, monkeypatch):
        """`conversation()`'s docstring calls this "the FOURTH, and on some PRs
        the ONLY one" — measured on caeli-marketing#182 and tests#291, where
        every rebuttal across five rounds lived in a commit message. An author
        who cannot comment as the repo owner has no other way to argue."""
        m, _ = pr
        def gh(path, **k):
            if "/reviews" in path:
                return json.dumps(self._revs("2026-08-31T10:00:00Z"))
            if "/commits" in path:
                return json.dumps([{"parents": [{"sha": "x"}],
                                    "author": {"login": "octocat"},
                                    "commit": {"message": "declined, here is why",
                                               "author": {"date": "2026-08-31T11:00:00Z"}}}])
            return json.dumps([])
        monkeypatch.setattr(m, "gh", gh)
        assert m._already_reviewed("r", 1, SHA, DIFF) is None

    def test_an_UPDATE_BRANCH_merge_commit_is_not_a_reply(self, pr, monkeypatch):
        """Adding /commits naively breaks the guard it lives in: update-branch
        is a `synchronize` whose MERGE commit has two parents and a prose body,
        so counting it would make every update-branch look like an argument and
        defeat the same-diff skip."""
        m, _ = pr
        def gh(path, **k):
            if "/reviews" in path:
                return json.dumps(self._revs("2026-08-31T10:00:00Z"))
            if "/commits" in path:
                return json.dumps([{"parents": [{"sha": "a"}, {"sha": "b"}],
                                    "author": {"login": "octocat"},
                                    "commit": {"message": "Merge branch 'main' into x",
                                               "author": {"date": "2026-08-31T11:00:00Z"}}}])
            return json.dumps([])
        monkeypatch.setattr(m, "gh", gh)
        assert m._already_reviewed("r", 1, SHA, DIFF) is not None, (
            "an update-branch merge counted as a rebuttal")

    def test_OUR_OWN_commit_does_not_count(self, pr, monkeypatch):
        m, _ = pr
        def gh(path, **k):
            if "/reviews" in path:
                return json.dumps(self._revs("2026-08-31T10:00:00Z"))
            if "/commits" in path:
                return json.dumps([{"parents": [{"sha": "x"}],
                                    "author": {"login": BOT},
                                    "commit": {"message": "ours",
                                               "author": {"date": "2026-08-31T23:00:00Z"}}}])
            return json.dumps([])
        monkeypatch.setattr(m, "gh", gh)
        assert m._already_reviewed("r", 1, SHA, DIFF) is not None



@pytest.fixture
def rv():
    from agentic_review import review
    return review


class TestAFixedMetadataFindingIsNotSkipped:
    """browser-extension#362, 2026-09-02: the review raised "PR title does not
    name a ticket", the author fixed the title one minute later, three more
    runs fired, and every one was skipped as "this exact commit already has a
    review". The title is not in the diff and does not move the SHA, so nothing
    the skip logic fingerprints changed — the finding could never be cleared
    by doing what it asked."""

    LAST = {"user": {"login": "review-bot"}, "commit_id": "a" * 40,
            "submitted_at": "2026-09-02T21:20:43Z",
            "body": "### AI review\n\n🟡 **PR title does not name a ticket "
                    "(expected e.g. SCRUM-1234)** — _the pull request_\n"}

    def _arm(self, rv, monkeypatch, last=None):
        monkeypatch.setattr(rv, "gh", lambda *a, **k: json.dumps([last or self.LAST]))
        monkeypatch.setattr(rv, "_me", lambda: "review-bot")
        monkeypatch.setattr(rv, "_someone_replied_since", lambda *a: False)

    def test_a_title_fixed_since_the_review_means_review_again(self, rv, monkeypatch):
        self._arm(rv, monkeypatch)
        assert rv._already_reviewed("r", 1, "a" * 40, "d",
                                    title="SCRUM-1216: fixed") is None

    def test_a_title_still_missing_the_ticket_is_still_a_skip(self, rv, monkeypatch):
        self._arm(rv, monkeypatch)
        assert rv._already_reviewed("r", 1, "a" * 40, "d",
                                    title="still no ticket") is not None

    def test_a_session_trailer_added_since_means_review_again(self, rv, monkeypatch):
        last = dict(self.LAST, body="🟡 **agent-written commit with no session link**")
        self._arm(rv, monkeypatch, last)
        assert rv._already_reviewed(
            "r", 1, "b" * 40, "d", title="SCRUM-1 x",
            commits=["Fix\n\nCo-Authored-By: Claude Opus 5 <x>\n"
                     "Claude-Session: session_01AbCdEfGhIj"]) is None

    def test_a_review_with_no_metadata_finding_skips_as_before(self, rv, monkeypatch):
        last = dict(self.LAST, body="🔵 **something about the code**")
        self._arm(rv, monkeypatch, last)
        assert rv._already_reviewed("r", 1, "a" * 40, "d",
                                    title="SCRUM-1 x") is not None



class TestAnApprovalIsFinalForItsCommit:
    """Tingyi's rule, 2026-09-02: once a PR is approved, only a new commit
    triggers a review. infra#156 was approved and then re-reviewed because a
    comment landed — a reply is "something new" when there is a finding to
    argue with, and after an approval there is nothing to argue with."""

    APPROVED = {"user": {"login": "review-bot"}, "state": "APPROVED",
                "commit_id": "a" * 40, "submitted_at": "2026-09-02T22:26:00Z",
                "body": "### AI review — no findings"}

    def _arm(self, rv, monkeypatch, reviews, replied=True):
        monkeypatch.setattr(rv, "gh", lambda *a, **k: json.dumps(reviews))
        monkeypatch.setattr(rv, "_me", lambda: "review-bot")
        monkeypatch.setattr(rv, "_someone_replied_since", lambda *a: replied)

    def test_a_comment_after_approval_does_not_review_again(self, rv, monkeypatch):
        self._arm(rv, monkeypatch, [self.APPROVED], replied=True)
        why = rv._already_reviewed("r", 1, "a" * 40, "d", title="SCRUM-1 x")
        assert why and "approved at this commit" in why

    def test_a_new_commit_after_approval_IS_reviewed(self, rv, monkeypatch):
        self._arm(rv, monkeypatch, [self.APPROVED], replied=False)
        assert rv._already_reviewed("r", 1, "b" * 40, "different diff",
                                    title="SCRUM-1 x") is None

    def test_a_reply_after_a_COMMENT_review_still_reviews(self, rv, monkeypatch):
        """Unchanged: with a finding on the table, an argument gets a re-look."""
        commented = dict(self.APPROVED, state="COMMENTED",
                         body="🟡 **something** — x")
        self._arm(rv, monkeypatch, [commented], replied=True)
        assert rv._already_reviewed("r", 1, "a" * 40, "d",
                                    title="SCRUM-1 x") is None

    def test_a_later_comment_review_on_the_same_commit_reopens_it(self, rv,
                                                                  monkeypatch):
        """An approval withdrawn by a later COMMENTED review at the same commit
        is no longer the verdict — but the approval row still exists, so the
        rule must look at whether ANY approval stands at this head. It does
        here, so a reply still does not re-review; a push does."""
        commented = dict(self.APPROVED, state="COMMENTED",
                         submitted_at="2026-09-02T22:30:00Z",
                         body="🟡 **found later** — x")
        self._arm(rv, monkeypatch, [self.APPROVED, commented], replied=True)
        why = rv._already_reviewed("r", 1, "a" * 40, "d", title="SCRUM-1 x")
        assert why and "approved at this commit" in why

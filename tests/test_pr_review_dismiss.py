"""A stale block must not outlive the finding that caused it (2026-08-24).

GitHub does not let a COMMENTED review clear a CHANGES_REQUESTED — only an
APPROVE or an explicit dismissal does. So once this reviewer blocks, every later
review carrying even one medium finding leaves the PR blocked by an objection
that no longer exists.

Seen on slack-app#344, and it is not a corner case:

    23:44  CHANGES_REQUESTED  high on find_by_domain_match
    00:04  COMMENTED          medium, unrelated, in another file
    00:24  COMMENTED          medium, follow-up
    decision = CHANGES_REQUESTED, with nothing left to change

THE SAFETY PROPERTY IS THE LOGIN CHECK, not the severity logic. Dismissing a
human's blocking review would silently delete a colleague's objection because a
model could not see the problem. Every test here that asserts "someone else's
review is untouched" is guarding that, and it matters more than the feature.
"""
import json

import pytest

from conftest import load_script


@pytest.fixture(scope="module")
def prr():
    return load_script("pr-review")


ME = "review-bot"


@pytest.fixture
def gh_spy(prr, monkeypatch):
    """Record every call; serve /user and the reviews list."""
    calls = []
    state = {"reviews": []}

    def fake_gh(path, method="GET", body=None, **kw):
        calls.append((method, path, body))
        if path == "/user":
            return json.dumps({"login": ME})
        if path.endswith("/reviews") and method == "GET":
            return json.dumps(state["reviews"])
        return "{}"

    monkeypatch.setattr(prr, "gh", fake_gh)
    prr._me.cache_clear()
    return calls, state


def _review(rid, login, review_state, commit_id="oldsha0"):
    return {"id": rid, "user": {"login": login}, "state": review_state,
            "commit_id": commit_id}


class TestItClearsItsOwnStaleBlock:
    def test_a_comment_dismisses_our_earlier_block(self, prr, gh_spy):
        """THE case. The high was fixed; a medium remains; the PR must stop
        being blocked by a finding that is gone."""
        calls, state = gh_spy
        state["reviews"] = [_review(1, ME, "CHANGES_REQUESTED")]
        assert prr._dismiss_stale_block("slack-app", 344, "COMMENT", "newsha1", False) == [1]
        assert any(m == "PUT" and p.endswith("/reviews/1/dismissals")
                   for m, p, _ in calls)

    def test_the_dismissal_says_the_remaining_findings_still_stand(self, prr, gh_spy):
        """It is a claim about our earlier objection, not a clean bill of health.
        The mediums posted alongside it are still real."""
        _, state = gh_spy
        state["reviews"] = [_review(1, ME, "CHANGES_REQUESTED")]
        prr._dismiss_stale_block("slack-app", 344, "COMMENT", "newsha1", False)
        assert "still stand" in prr.DISMISS_MESSAGE

    def test_several_of_our_stale_blocks_are_all_cleared(self, prr, gh_spy):
        _, state = gh_spy
        state["reviews"] = [_review(1, ME, "CHANGES_REQUESTED"),
                            _review(2, ME, "COMMENTED"),
                            _review(3, ME, "CHANGES_REQUESTED")]
        assert prr._dismiss_stale_block("slack-app", 344, "COMMENT", "newsha1", False) == [1, 3]


class TestItNeverClearsSomeoneElses:
    def test_a_humans_block_is_untouched(self, prr, gh_spy):
        """The property that matters most in this file. A human's objection is
        not this tool's to overrule, whatever the model concluded."""
        calls, state = gh_spy
        state["reviews"] = [_review(9, "octocat", "CHANGES_REQUESTED")]
        assert prr._dismiss_stale_block("slack-app", 344, "COMMENT", "newsha1", False) == []
        assert not any(m == "PUT" for m, _, _ in calls)

    def test_ours_is_cleared_while_a_humans_survives(self, prr, gh_spy):
        _, state = gh_spy
        state["reviews"] = [_review(9, "octocat", "CHANGES_REQUESTED"),
                            _review(1, ME, "CHANGES_REQUESTED")]
        assert prr._dismiss_stale_block("slack-app", 344, "COMMENT", "newsha1", False) == [1]

    def test_an_unresolvable_identity_dismisses_nothing(self, prr, monkeypatch):
        """Fail closed. If we cannot prove which reviews are ours, we touch none
        of them — the stale block is the safe outcome.

        The reviews fetch SUCCEEDS here and only `/user` fails, so this reaches
        the real branch. The first version stubbed `gh` to raise on everything
        and passed for the wrong reason — the fetch failed first and returned
        early, never testing identity at all.

        Mutating `if not me: return []` away does NOT fail this, and that is
        correct rather than a gap: `_me()` returns "" on failure, and no real
        login equals "", so the per-review comparison below already fails
        closed. The early return is belt-and-braces — it saves a wasted list
        fetch and states the intent. The property is pinned here either way,
        which is what this test is for.
        """
        attempted = []

        def fake_gh(path, method="GET", body=None, **kw):
            attempted.append((method, path))
            if path == "/user":
                raise RuntimeError("502 from /user only")
            if path.endswith("/reviews") and method == "GET":
                return json.dumps([_review(1, ME, "CHANGES_REQUESTED")])
            return "{}"

        monkeypatch.setattr(prr, "gh", fake_gh)
        prr._me.cache_clear()
        assert prr._dismiss_stale_block("slack-app", 344, "COMMENT", "newsha1", False) == []
        assert not any(m == "PUT" for m, _ in attempted), \
            "dismissed a review without knowing whose it was"


class TestWhenItDoesNotFire:
    def test_a_still_blocking_review_keeps_the_block(self, prr, gh_spy):
        """REQUEST_CHANGES means we still object. Clearing our own block while
        raising it again would be incoherent."""
        calls, state = gh_spy
        state["reviews"] = [_review(1, ME, "CHANGES_REQUESTED")]
        assert prr._dismiss_stale_block("slack-app", 344, "REQUEST_CHANGES", "newsha1", False) == []
        assert not any(m == "PUT" for m, _, _ in calls)

    def test_an_approve_needs_no_help(self, prr, gh_spy):
        """GitHub supersedes a block with an APPROVE on its own side; a
        dismissal here would be a second, redundant write."""
        calls, state = gh_spy
        state["reviews"] = [_review(1, ME, "CHANGES_REQUESTED")]
        assert prr._dismiss_stale_block("slack-app", 344, "APPROVE", "newsha1", False) == []
        assert not any(m == "PUT" for m, _, _ in calls)

    def test_an_already_dismissed_review_is_not_dismissed_twice(self, prr, gh_spy):
        _, state = gh_spy
        state["reviews"] = [_review(1, ME, "DISMISSED")]
        assert prr._dismiss_stale_block("slack-app", 344, "COMMENT", "newsha1", False) == []

    def test_a_failed_dismissal_is_swallowed(self, prr, monkeypatch):
        """The review is already posted by the time this runs, so the worst case
        is the status quo — a stale block — never a lost review."""
        def fake_gh(path, method="GET", body=None, **kw):
            if path == "/user":
                return json.dumps({"login": ME})
            if path.endswith("/reviews") and method == "GET":
                return json.dumps([_review(1, ME, "CHANGES_REQUESTED")])
            raise RuntimeError("403 forbidden")
        monkeypatch.setattr(prr, "gh", fake_gh)
        prr._me.cache_clear()
        assert prr._dismiss_stale_block("slack-app", 344, "COMMENT", "newsha1", False) == []


class TestItWillNotClearABlockItCannotJustify:
    """The AI review's finding on infra#111, and it was right.

    Dismissing on `event == "COMMENT"` alone treats "this run found no high" as
    "the earlier high was fixed". Those are different claims, and the gap is
    reachable: with the diff truncated at MAX_DIFF and findings capped, run 2 can
    simply never reach the region where run 1 found the high. The old code then
    cleared a LIVE block and published "the blocking finding is no longer present
    at this head" — a fact nothing had checked. Under
    `required_approving_review_count: 1` that is a real gate drop.
    """

    def test_a_truncated_review_dismisses_nothing(self, prr, gh_spy):
        """THE reported case. A clean result from a review that did not read the
        whole diff is not evidence about the part it skipped."""
        calls, state = gh_spy
        state["reviews"] = [_review(1, ME, "CHANGES_REQUESTED", "oldsha0")]
        assert prr._dismiss_stale_block("slack-app", 344, "COMMENT",
                                        "newsha1", True) == []
        assert not any(m == "PUT" for m, _, _ in calls)

    def test_an_unmoved_head_dismisses_nothing(self, prr, gh_spy):
        """Re-reading the SAME code and getting a quieter answer is model
        nondeterminism, not a fix. Without this, re-requesting a review clears
        any block by rolling the dice until it comes up quiet."""
        calls, state = gh_spy
        state["reviews"] = [_review(1, ME, "CHANGES_REQUESTED", "samesha")]
        assert prr._dismiss_stale_block("slack-app", 344, "COMMENT",
                                        "samesha", False) == []
        assert not any(m == "PUT" for m, _, _ in calls)

    def test_a_block_with_no_recorded_commit_is_left_alone(self, prr, gh_spy):
        """Cannot prove the head moved, so cannot justify clearing it."""
        _, state = gh_spy
        state["reviews"] = [{"id": 1, "user": {"login": ME},
                             "state": "CHANGES_REQUESTED"}]
        assert prr._dismiss_stale_block("slack-app", 344, "COMMENT",
                                        "newsha1", False) == []

    def test_the_message_states_only_what_was_checked(self, prr, gh_spy):
        """It used to assert the finding was gone. It now says the code moved and
        a full re-review found nothing blocking — and says outright that this is
        not a claim the original was fixed."""
        calls, state = gh_spy
        state["reviews"] = [_review(1, ME, "CHANGES_REQUESTED", "oldsha0")]
        prr._dismiss_stale_block("slack-app", 344, "COMMENT", "newsha1", False)
        body = next(b for m, p, b in calls if m == "PUT")
        assert "oldsha" in body["message"] and "newsha1" in body["message"]
        assert "not a claim the original finding was fixed" in body["message"]


class TestItWithdrawsItsOwnNowStaleApproval:
    """An approval must not outlive the code it was given for (2026-08-27).

    Asked for, off slack-app#363:

        20:25:32  review-bot  APPROVED   @6b702ab
        20:31:03  review-bot  COMMENTED  @df3365d   <- found something

    GitHub changes a reviewer's state only on APPROVE or REQUEST_CHANGES, so a
    COMMENTED review leaves the approval standing. The PR then shows a green
    "approved these changes" beside the reviewer's own findings on newer code —
    and a human skimming the header sees the approval, not the comment.

    The mirror of `_dismiss_stale_block`, and the guards are deliberately
    LOOSER because the risk points the other way. Clearing a block wrongly
    unblocks a merge, so that path demands a moved head and a whole diff.
    Withdrawing an approval wrongly costs a re-request. That asymmetry is the
    point of this class, and every test below is one half of it.
    """

    def test_a_comment_withdraws_our_earlier_approval(self, prr, gh_spy):
        """THE case."""
        calls, state = gh_spy
        state["reviews"] = [_review(7, ME, "APPROVED", "oldsha0")]
        assert prr._withdraw_stale_approval("slack-app", 363, "newsha1") == [7]
        assert any(m == "PUT" and p.endswith("/reviews/7/dismissals")
                   for m, p, _ in calls)

    def test_a_humans_approval_is_untouched(self, prr, gh_spy):
        """The only real safety property here, same as its sibling. A
        colleague's approval is theirs to withdraw."""
        calls, state = gh_spy
        state["reviews"] = [_review(9, "octocat", "APPROVED", "oldsha0")]
        assert prr._withdraw_stale_approval("slack-app", 363, "newsha1") == []
        assert not any(m == "PUT" for m, _, _ in calls)

    def test_ours_goes_while_a_humans_survives(self, prr, gh_spy):
        _, state = gh_spy
        state["reviews"] = [_review(9, "octocat", "APPROVED", "oldsha0"),
                            _review(7, ME, "APPROVED", "oldsha0")]
        assert prr._withdraw_stale_approval("slack-app", 363, "newsha1") == [7]

    def test_an_UNMOVED_head_still_withdraws(self, prr, gh_spy):
        """The asymmetry, stated. `_dismiss_stale_block` refuses here, because
        re-reading the same code and going quiet is nondeterminism rather than a
        fix. Withdrawing is the safe direction: an approval standing next to a
        finding is incoherent whether or not the head moved."""
        _, state = gh_spy
        state["reviews"] = [_review(7, ME, "APPROVED", "samesha")]
        assert prr._withdraw_stale_approval("slack-app", 363, "samesha") == [7]

    def test_a_review_with_no_recorded_commit_is_still_withdrawn(self, prr, gh_spy):
        """Same direction. Not knowing what it approved is not a reason to leave
        an approval sitting beside a finding."""
        _, state = gh_spy
        state["reviews"] = [{"id": 7, "user": {"login": ME}, "state": "APPROVED"}]
        assert prr._withdraw_stale_approval("slack-app", 363, "newsha1") == [7]

    def test_a_dismissed_approval_is_not_withdrawn_twice(self, prr, gh_spy):
        _, state = gh_spy
        state["reviews"] = [_review(7, ME, "DISMISSED", "oldsha0")]
        assert prr._withdraw_stale_approval("slack-app", 363, "newsha1") == []

    def test_a_block_is_not_touched_by_this_path(self, prr, gh_spy):
        """Its sibling owns that, with stricter guards. Handling it here would
        route a merge-unblocking dismissal through the loose path."""
        _, state = gh_spy
        state["reviews"] = [_review(1, ME, "CHANGES_REQUESTED", "oldsha0")]
        assert prr._withdraw_stale_approval("slack-app", 363, "newsha1") == []

    def test_an_unresolvable_identity_withdraws_nothing(self, prr, monkeypatch):
        """Fail closed on the one property that matters.

        The reviews fetch SUCCEEDS and only `/user` fails, so this reaches the
        real branch rather than returning early for the wrong reason.

        Mutating `if not me: return []` away does NOT fail this, and that is
        correct rather than a gap — the same finding as its sibling. `_me()`
        returns "" on failure and no real login equals "", so the per-review
        comparison below already fails closed. The early return saves a wasted
        list fetch and states the intent; the property is pinned here either
        way, which is what this test is for.
        """
        attempted = []

        def fake_gh(path, method="GET", body=None, **kw):
            attempted.append((method, path))
            if path == "/user":
                raise RuntimeError("502 from /user only")
            if path.endswith("/reviews") and method == "GET":
                return json.dumps([_review(7, ME, "APPROVED", "oldsha0")])
            return "{}"

        monkeypatch.setattr(prr, "gh", fake_gh)
        prr._me.cache_clear()
        assert prr._withdraw_stale_approval("slack-app", 363, "newsha1") == []
        assert not any(m == "PUT" for m, _ in attempted)

    def test_a_failed_withdrawal_is_swallowed(self, prr, monkeypatch):
        """The review is already posted; the worst case is the status quo."""
        def fake_gh(path, method="GET", body=None, **kw):
            if path == "/user":
                return json.dumps({"login": ME})
            if path.endswith("/reviews") and method == "GET":
                return json.dumps([_review(7, ME, "APPROVED", "oldsha0")])
            raise RuntimeError("403 forbidden")
        monkeypatch.setattr(prr, "gh", fake_gh)
        prr._me.cache_clear()
        assert prr._withdraw_stale_approval("slack-app", 363, "newsha1") == []

    def test_the_message_says_what_it_claims_and_what_it_does_not(self, prr, gh_spy):
        """A statement about staleness, and about NOTHING else.

        The first version called the accompanying findings "not blocking", and
        the AI review caught that it cannot know: the withdrawal fires on a
        COMMENT verdict, and `review_event` also returns COMMENT for a finding
        whose severity it did not RECOGNISE — deliberately, so an unknown word
        never approves. Asserting "not blocking" there claims exactly what that
        run declined to decide. Same overclaim DISMISS_MESSAGE was already
        narrowed to remove.
        """
        calls, state = gh_spy
        state["reviews"] = [_review(7, ME, "APPROVED", "oldsha0")]
        prr._withdraw_stale_approval("slack-app", 363, "newsha1")
        body = next(b for m, p, b in calls if m == "PUT")
        assert "oldsha" in body["message"] and "newsha1" in body["message"]
        assert "not blocking" not in body["message"], "it cannot know that"
        assert "no longer describes the head" in body["message"]

    def test_an_unrecognised_severity_also_reaches_this_path(self, prr):
        """The state the message must not overclaim about, pinned at its source
        so the two cannot drift: a severity the model invented withholds
        approval and still routes to COMMENT, which is what triggers the
        withdrawal."""
        for sev in ("critical", "blocker", ""):
            finding = [{"file": "a.py", "line": 1, "severity": sev,
                        "title": "t", "detail": "d"}]
            assert prr.normalize_severity(sev) == "unknown"
            assert prr.review_event(finding) == "COMMENT"


class TestAFailedRunReleasesItsOwnRequest:
    """A failed review must not deadlock the PR (2026-08-28).

    A run that fails posts nothing, so GitHub never drops us from
    `requested_reviewers`. It also emits no `review_requested` event when the
    reviewer is ALREADY requested, so `--add-reviewer` is a silent no-op. And
    the caller deliberately ignores `synchronize`, so pushing commits triggers
    nothing. Every recovery path closes at once.

    portal-api#150 hit exactly that: a run failed at 21:22 and left the request
    in place, two later commits triggered nothing, and the last review had read
    64687d6 while the head moved to c3bfb52. Recovering took a DELETE and a POST
    by hand, and nothing in the failure hinted at it.
    """

    def test_it_releases_the_request(self, prr, gh_spy):
        calls, _ = gh_spy
        assert prr._release_review_request("portal-api", 150) is True
        assert any(m == "DELETE" and p.endswith("/requested_reviewers")
                   for m, p, _ in calls)

    def test_it_releases_only_ITSELF(self, prr, gh_spy):
        """A human reviewer on the same PR is not ours to withdraw."""
        calls, _ = gh_spy
        prr._release_review_request("portal-api", 150)
        body = next(b for m, _, b in calls if m == "DELETE")
        assert body == {"reviewers": [ME]}

    def test_an_unresolvable_identity_releases_nothing(self, prr, monkeypatch):
        """Fail closed: without knowing who we are, a DELETE could name someone
        else."""
        attempted = []

        def fake_gh(path, method="GET", body=None, **kw):
            attempted.append(method)
            if path == "/user":
                raise RuntimeError("502")
            return "{}"

        monkeypatch.setattr(prr, "gh", fake_gh)
        prr._me.cache_clear()
        assert prr._release_review_request("portal-api", 150) is False
        assert "DELETE" not in attempted

    def test_a_missing_repo_or_pr_is_a_no_op(self, prr, gh_spy):
        calls, _ = gh_spy
        assert prr._release_review_request("", 150) is False
        assert prr._release_review_request("portal-api", None) is False
        assert not any(m == "DELETE" for m, _, _ in calls)

    def test_a_failed_release_is_swallowed(self, prr, monkeypatch):
        """It runs while the real error is on its way out and must never
        replace it."""
        def fake_gh(path, method="GET", body=None, **kw):
            if path == "/user":
                return json.dumps({"login": ME})
            raise RuntimeError("403 forbidden")
        monkeypatch.setattr(prr, "gh", fake_gh)
        prr._me.cache_clear()
        assert prr._release_review_request("portal-api", 150) is False


class TestTheReleaseIsWiredToTheFailurePath:
    """The writer alone proves nothing — the old bug was that nothing called it."""

    @pytest.fixture
    def wired(self, prr, monkeypatch):
        released = []
        monkeypatch.setattr(prr, "_release_review_request",
                            lambda r, p: released.append((r, p)) or True)
        prr._CURRENT["repo"], prr._CURRENT["pr"] = "portal-api", 150
        return prr, released

    def test_a_failing_run_releases_and_still_raises(self, wired, monkeypatch):
        prr, released = wired
        monkeypatch.setattr(prr, "main",
                            lambda: (_ for _ in ()).throw(prr.ReviewError("boom")))
        with pytest.raises(prr.ReviewError):
            prr._main_unless_superseded()
        assert released == [("portal-api", 150)]

    def test_a_SUPERSEDED_run_does_NOT_release(self, wired, monkeypatch):
        """The superseding run posts the review. Dropping the request here would
        cancel a review that is about to happen."""
        prr, released = wired
        monkeypatch.setattr(prr, "main",
                            lambda: (_ for _ in ()).throw(prr.Superseded("newer")))
        with pytest.raises(SystemExit):
            prr._main_unless_superseded()
        assert released == []

    def test_a_SUCCESSFUL_run_does_NOT_release(self, wired, monkeypatch):
        """GitHub drops us on its own when a review is posted; releasing as well
        would be a redundant write on every green run."""
        prr, released = wired
        monkeypatch.setattr(prr, "main", lambda: None)
        prr._main_unless_superseded()
        assert released == []


class TestTheEarliestFailuresRelease:
    """The hole in the first version of the release, found by the AI review.

    `_CURRENT["repo"]/["pr"]` were set AFTER the meta fetch and `pr_diff`, so a
    failure in either — a 5xx, an expired token, a flaky lookup on the FIRST
    network call of the process — reached the release with an empty `_CURRENT`,
    hit the `if not repo or not pr` guard and released nothing.

    That is the deadlock this whole change exists to clear, still open for its
    likeliest trigger. The values were never unknown: they are `sys.argv[1:3]`,
    and only the assignment was late.
    """

    @pytest.fixture
    def driven(self, prr, monkeypatch):
        released = []
        monkeypatch.setattr(prr, "_release_review_request",
                            lambda r, p: released.append((r, p)) or True)
        monkeypatch.setattr(prr.sys, "argv", ["pr-review.py", "portal-api", "150"])
        prr._CURRENT["repo"], prr._CURRENT["pr"] = "", ""
        monkeypatch.delenv("DRY", raising=False)
        return prr, released

    def test_a_failure_on_the_FIRST_gh_call_still_releases(self, driven, monkeypatch):
        """The meta fetch is the first network call in the process."""
        prr, released = driven
        monkeypatch.setattr(prr, "gh",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("502")))
        with pytest.raises(Exception):
            prr._main_unless_superseded()
        assert released == [("portal-api", "150")], "released nothing on an early failure"

    def test_it_releases_the_REAL_repo_and_pr_not_blanks(self, driven, monkeypatch):
        """The guard that fails closed on empties is right; feeding it empties
        when the values are known is what was wrong."""
        prr, released = driven
        monkeypatch.setattr(prr, "gh",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("401")))
        with pytest.raises(Exception):
            prr._main_unless_superseded()
        assert released and all(x for x in released[0]), f"released {released[0]!r}"


class TestADryRunNeverWritesToARealPR:
    """`DRY` prints what it would do instead of doing it, and `CRON_DRY_RUN` is
    side-effect-free everywhere else in cronlib. A rehearsal that happened to
    FAIL was still issuing a real DELETE against a real PR's reviewers."""

    def test_a_failing_dry_run_releases_nothing(self, prr, monkeypatch, capsys):
        released = []
        monkeypatch.setattr(prr, "_release_review_request",
                            lambda r, p: released.append((r, p)) or True)
        monkeypatch.setenv("DRY", "1")
        prr._CURRENT["repo"], prr._CURRENT["pr"] = "portal-api", 150
        monkeypatch.setattr(prr, "main",
                            lambda: (_ for _ in ()).throw(prr.ReviewError("boom")))
        with pytest.raises(prr.ReviewError):
            prr._main_unless_superseded()
        assert released == []
        assert "would release" in capsys.readouterr().out

    def test_a_failing_REAL_run_still_releases(self, prr, monkeypatch):
        """Guard the guard: a DRY check that was always true would silently
        disable the fix."""
        released = []
        monkeypatch.setattr(prr, "_release_review_request",
                            lambda r, p: released.append((r, p)) or True)
        monkeypatch.delenv("DRY", raising=False)
        prr._CURRENT["repo"], prr._CURRENT["pr"] = "portal-api", 150
        monkeypatch.setattr(prr, "main",
                            lambda: (_ for _ in ()).throw(prr.ReviewError("boom")))
        with pytest.raises(prr.ReviewError):
            prr._main_unless_superseded()
        assert released == [("portal-api", 150)]


class TestTruncationOnlyMattersWhereTheBlockIs:
    """SCRUM-1293. Refusing to dismiss on ANY truncated review became a trap
    once a file could be excluded permanently: past `MAX_FILE_DIFF` it is over
    the ceiling in every future review too, so every review of that PR is
    truncated and the block can never clear itself. infra#183 sat blocked
    through three clean reviews until a human dismissed it by hand.

    The question was never "was this review truncated" but "did it read the
    file the block is ABOUT".
    """

    BLOCK = {"id": 1, "state": "CHANGES_REQUESTED", "commit_id": "oldsha",
             "user": {"login": "review-bot"},
             "body": "🔴 **the defect** — [`src/app.py:12`](http://x)"}

    def _wire(self, monkeypatch, prr, reviews):
        calls = []

        def gh(path, method="GET", body=None, accept=""):
            calls.append((method, path))
            return json.dumps(reviews)
        monkeypatch.setattr(prr, "gh", gh)
        monkeypatch.setattr(prr, "_me", lambda: "review-bot")
        return calls

    def test_a_block_whose_file_was_read_clears_even_when_truncated(
            self, monkeypatch, pr_review):
        prr = pr_review
        self._wire(monkeypatch, prr, [self.BLOCK])
        assert prr._dismiss_stale_block("app", 1, "COMMENT", "newsha", True,
                                        unread=["data/huge.jsonl"]) == [1]

    def test_a_block_whose_file_was_NOT_read_still_stands(
            self, monkeypatch, pr_review):
        """The case the blanket guard was written for, kept."""
        prr = pr_review
        self._wire(monkeypatch, prr, [self.BLOCK])
        assert prr._dismiss_stale_block("app", 1, "COMMENT", "newsha", True,
                                        unread=["src/app.py"]) == []

    def test_a_block_naming_no_file_still_refuses_under_truncation(
            self, monkeypatch, pr_review):
        """No parseable path means no way to tell, and an unparseable body under
        truncation is exactly what the old rule was right about."""
        prr = pr_review
        self._wire(monkeypatch, prr, [dict(self.BLOCK, body="🔴 **something** — nowhere")])
        assert prr._dismiss_stale_block("app", 1, "COMMENT", "newsha", True,
                                        unread=["x.py"]) == []

    def test_an_untruncated_review_is_unaffected(self, monkeypatch, pr_review):
        prr = pr_review
        self._wire(monkeypatch, prr, [self.BLOCK])
        assert prr._dismiss_stale_block("app", 1, "COMMENT", "newsha", False) == [1]

    def test_the_head_must_still_have_moved(self, monkeypatch, pr_review):
        """Re-reading the same code and reaching a different verdict is model
        nondeterminism, not a fix — that rule is untouched."""
        prr = pr_review
        self._wire(monkeypatch, prr, [dict(self.BLOCK, commit_id="newsha")])
        assert prr._dismiss_stale_block("app", 1, "COMMENT", "newsha", True,
                                        unread=[]) == []

    def test_main_hands_over_what_it_did_not_read(self, monkeypatch, pr_review):
        """The wiring, not the helper: `unopened` is what the caveat names, and
        it is what the dismissal has to be asked about."""
        prr = pr_review
        seen = {}
        monkeypatch.setattr(prr, "pr_diff", lambda *a: (
            "--- a/x\n+++ b/x\n@@\n+x\n", ["data/huge.jsonl"], 0))
        monkeypatch.setattr(prr, "_already_reviewed", lambda *a, **k: "")
        monkeypatch.setattr(prr, "checkout", lambda *a: None)
        monkeypatch.setattr(prr, "build_context", lambda *a: "")
        monkeypatch.setattr(prr.ctx, "skeletons", lambda *a: "")
        monkeypatch.setattr(prr, "conversation", lambda *a: "")
        monkeypatch.setattr(prr, "changed_since_last_review", lambda *a, **k: "")
        monkeypatch.setattr(prr, "commit_messages", lambda *a: [])
        monkeypatch.setattr(prr, "review_findings", lambda *a, **k: [])
        monkeypatch.setattr(prr, "_revise", lambda f, w, r: (f, []))
        monkeypatch.setattr(prr.checks, "run_all", lambda *a, **k: [])
        monkeypatch.setattr(prr, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(prr, "post_review",
                            lambda repo, n, ev, body, head_sha="", truncated=False,
                            unread=(), pr_files=(): (seen.update(
                                unread=list(unread), pr_files=list(pr_files)), ev)[1])
        monkeypatch.setattr(prr, "gh", lambda *a, **k: json.dumps(
            {"draft": False, "state": "open", "merged": False, "title": "SCRUM-1 x",
             "user": {"login": "someone"}, "head": {"sha": "a" * 40}}))
        monkeypatch.setattr(prr.sys, "argv", ["pr-review", "repo", "1"])
        monkeypatch.delenv("DRY", raising=False)
        prr.main()
        assert seen["unread"] == ["data/huge.jsonl"]
        # AND what the PR touches, which is how a cited token is known to be a
        # file at all. Dropping it falls back to the shape test and brings the
        # never-clears trap back — silently, until this assertion.
        assert "x" in seen["pr_files"]

    def test_the_all_oversized_exit_hands_over_every_file(self, monkeypatch, pr_review):
        """It read NOTHING, so it may not clear a block about any of it. That
        path posted without `unread`, so the dismissal saw an empty set and
        cleared precisely the blocks it could not speak for — a false-clean,
        and worse than the blanket refusal it replaced."""
        prr = pr_review
        seen = {}
        d = prr._Diff("")
        d.oversized = ["data/huge.jsonl"]
        monkeypatch.setattr(prr, "pr_diff",
                            lambda *a: (d, ["data/huge.jsonl"], prr._Skipped([])))
        monkeypatch.setattr(prr, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(prr, "post_review",
                            lambda repo, n, ev, body, head_sha="", truncated=False,
                            unread=(), pr_files=(): (seen.update(
                                unread=list(unread), pr_files=list(pr_files),
                                truncated=truncated), ev)[1])
        monkeypatch.setattr(prr.status, "done", lambda *a: None)
        monkeypatch.setattr(prr, "gh", lambda *a, **k: json.dumps(
            {"draft": False, "state": "open", "merged": False, "title": "SCRUM-1 x",
             "user": {"login": "someone"}, "head": {"sha": "e" * 40}}))
        monkeypatch.setattr(prr.sys, "argv", ["pr-review", "repo", "1"])
        monkeypatch.delenv("DRY", raising=False)
        prr.main()
        assert seen["unread"] == ["data/huge.jsonl"] and seen["truncated"] is True
        assert seen["pr_files"] == ["data/huge.jsonl"]

    MIXED = ("🔴 **read one** — [`src/app.py:12`](http://x)\n"
             "🟡 **no line** — `data/huge.jsonl`\n"
             "> **⚠️ Partial review — 1 changed file(s) were NOT opened**: "
             "`vendor/other.py`.\n"
             "🔵 **prose** — the `normalize` helper is fine\n")

    def test_a_finding_with_no_line_still_names_its_file(self, pr_review):
        """`_where_link` renders `path:line` when the finding carries a line and
        a bare `path` when it does not — `validate_findings` never required one.
        A block mixing the shapes must not look like it cited only the first."""
        assert pr_review._cited_files(self.MIXED) == {"src/app.py", "data/huge.jsonl"}

    def test_the_caveats_own_file_list_is_not_a_citation(self, pr_review):
        """The "were NOT opened" block renders bare `path` spans too. Reading
        those as cited would make EVERY truncated review's block undismissable —
        the blanket behaviour this replaced."""
        assert "vendor/other.py" not in pr_review._cited_files(self.MIXED)

    def test_prose_in_backticks_is_not_a_path(self, pr_review):
        assert "normalize" not in pr_review._cited_files(self.MIXED)

    def test_a_LINELESS_BLOCKING_finding_about_an_unread_file_keeps_the_block(
            self, monkeypatch, pr_review):
        """A 🔴 naming its file bare, with no `:line`, still protects itself.

        The unread file is on the BLOCKING line here. It used to be on a 🟡 in
        `MIXED`, which passed for the wrong reason — see
        `test_a_nit_about_an_unread_file_does_NOT_keep_the_block`.
        """
        prr = pr_review
        body = ("🔴 **no line** — `data/huge.jsonl`\n"
                "🟡 **read one** — [`src/app.py:12`](http://x)\n")
        monkeypatch.setattr(prr, "gh", lambda *a, **k: json.dumps(
            [{"id": 1, "state": "CHANGES_REQUESTED", "commit_id": "oldsha",
              "user": {"login": "review-bot"}, "body": body}]))
        monkeypatch.setattr(prr, "_me", lambda: "review-bot")
        assert prr._dismiss_stale_block("app", 1, "COMMENT", "newsha", True,
                                        unread=["data/huge.jsonl"]) == []

    def test_a_nit_about_an_unread_file_does_NOT_keep_the_block(
            self, monkeypatch, pr_review):
        """caeli-marketing#489, and the point of this whole class.

        Only `high` produces REQUEST_CHANGES, so the 🔴 is the entire reason a
        block exists. On #489 both blocks cited a `.json` the run HAD read and
        come back clean on, and the refusal was driven by a 🔵 naming an unread
        `brief.md`. With a 513,579-char diff against a 180,000 ceiling, 65% is
        unread in every run and some nit always names a file nobody reached —
        so the block could never clear itself, and a human dismissed it by
        hand. That is SCRUM-1293's trap by a different route.
        """
        prr = pr_review
        body = ("🔴 **the defect** — [`src/app.py:12`](http://x)\n"
                "🔵 **a nit** — [`docs/brief.md:3`](http://x)\n")
        calls = self._wire(monkeypatch, prr, [
            {"id": 1, "state": "CHANGES_REQUESTED", "commit_id": "oldsha",
             "user": {"login": "review-bot"}, "body": body}])
        assert prr._dismiss_stale_block("app", 1, "COMMENT", "newsha", True,
                                        unread=["docs/brief.md"]) == [1]
        assert any(m == "PUT" for m, _ in calls)

    def test_the_blocking_icons_are_derived_from_the_event_map(self, pr_review):
        """A SECOND MAP, not today's — a literal 🔴 passes every test written
        against the current one.

        `EVENT_BY_SEVERITY` owns which severities produce REQUEST_CHANGES. Its
        own comment notes `medium` also blocks under
        `required_approving_review_count: 1`, differing only in visibility, so
        a second entry is a plausible edit. If it lands and this pattern does
        not follow, a block raised by a 🟡 on an unread file yields no blind
        paths and the guard clears a LIVE block.
        """
        prr = pr_review
        both = dict(prr.EVENT_BY_SEVERITY, medium="REQUEST_CHANGES")
        pat = prr._blocking_pattern(both)
        assert pat.search(f"{prr.ICON['medium']} **x** — `a.py`")
        assert pat.search(f"{prr.ICON['high']} **x** — `a.py`")
        assert not pat.search(f"{prr.ICON['low']} **x** — `a.py`")

    def test_only_high_blocks_today_so_a_nit_icon_is_not_matched(self, pr_review):
        prr = pr_review
        pat = prr._blocking_pattern()
        assert pat.search(f"{prr.ICON['high']} **x** — `a.py`")
        for s in ("medium", "low", "unknown"):
            assert not pat.search(f"{prr.ICON[s]} **x** — `a.py`"), s

    def test_no_blocking_severity_falls_back_to_every_finding_line(self, pr_review):
        """An empty alternation compiles to a pattern matching a bare space,
        and silently narrowing to nothing is the direction that clears live
        blocks."""
        prr = pr_review
        pat = prr._blocking_pattern({"high": "COMMENT", "low": "APPROVE"})
        assert pat is prr._FINDING_LINE

    def test_a_block_with_no_parseable_high_falls_back_to_the_whole_body(
            self, monkeypatch, pr_review):
        """Narrowing to the 🔴s is only safe when a 🔴 can be found.

        A CHANGES_REQUESTED review with no recognisable blocking line is one
        this code cannot explain, and narrowing on a body it failed to read is
        the false-clean this function has been burned by four times. So the
        old whole-body check stands in that case.
        """
        prr = pr_review
        # A 🟡, so the WHOLE-BODY parser finds a citation and the blocking-only
        # parser finds none — the only shape that tells the two apart. And
        # truncated=False, because `truncated and not cited` would otherwise
        # refuse for its own reason and the test would pass either way.
        body = "🟡 **the defect** — `data/huge.jsonl`\n"
        monkeypatch.setattr(prr, "gh", lambda *a, **k: json.dumps(
            [{"id": 1, "state": "CHANGES_REQUESTED", "commit_id": "oldsha",
              "user": {"login": "review-bot"}, "body": body}]))
        monkeypatch.setattr(prr, "_me", lambda: "review-bot")
        assert prr._dismiss_stale_block("app", 1, "COMMENT", "newsha", False,
                                        unread=["data/huge.jsonl"]) == []

    def test_every_severity_icon_counts_as_a_finding_line(self, pr_review):
        """`normalize_severity` gives ⚠️ to any severity outside the vocabulary,
        and the hand-written alternation listed only 🔴🟡🔵 — so a block citing
        an unread file through an unknown-severity finding was invisible to the
        guard and got dismissed. The pattern is built from ICON now, so a fifth
        icon cannot reintroduce it."""
        prr = pr_review
        for icon in prr.ICON.values():
            body = f"{icon} **a** — `data/huge.jsonl`\n"
            assert prr._cited_files(body) == {"data/huge.jsonl"}, icon

    def test_an_unknown_severity_finding_does_not_keep_the_block(
            self, monkeypatch, pr_review):
        """⚠️ maps to COMMENT in `EVENT_BY_SEVERITY`, not REQUEST_CHANGES, so
        it withholds approval rather than blocking — it cannot be the reason a
        CHANGES_REQUESTED exists and cannot keep one alive.

        That ⚠️ is still PARSED as a finding line is the property that mattered
        here, and it is pinned directly by
        `test_every_severity_icon_counts_as_a_finding_line`.
        """
        prr = pr_review
        body = ("🔴 **read one** — [`src/app.py:12`](http://x)\n"
                "⚠️ **odd severity** — `data/huge.jsonl`\n")
        monkeypatch.setattr(prr, "gh", lambda *a, **k: json.dumps(
            [{"id": 1, "state": "CHANGES_REQUESTED", "commit_id": "oldsha",
              "user": {"login": "review-bot"}, "body": body}]))
        monkeypatch.setattr(prr, "_me", lambda: "review-bot")
        assert prr._dismiss_stale_block("app", 1, "COMMENT", "newsha", True,
                                        unread=["data/huge.jsonl"]) == [1]

    def test_a_single_word_filename_keeps_the_block(self, monkeypatch, pr_review):
        """`Makefile`, `Dockerfile`, `LICENSE` have no separator and no suffix,
        and `_where_link` renders them bare. The shape test dropped them, so a
        block citing an unread Makefile was invisible and got dismissed — the
        FOURTH shape of one false-clean on this change. The decision is an exact
        set intersection now, so no filename has to look like one."""
        prr = pr_review
        body = ("🔴 **the build** — `Makefile`\n"
                "🟡 **read one** — [`src/app.py:12`](http://x)\n")
        monkeypatch.setattr(prr, "gh", lambda *a, **k: json.dumps(
            [{"id": 1, "state": "CHANGES_REQUESTED", "commit_id": "oldsha",
              "user": {"login": "review-bot"}, "body": body}]))
        monkeypatch.setattr(prr, "_me", lambda: "review-bot")
        assert prr._dismiss_stale_block("app", 1, "COMMENT", "newsha", True,
                                        unread=["Makefile"]) == []

    def test_prose_cannot_match_a_path_that_was_not_read(self, monkeypatch, pr_review):
        """The other half: dropping the shape test must not make every
        backticked word block a dismissal. `normalize` is not a file, so it
        cannot be in the unread set."""
        prr = pr_review
        body = ("🔴 **read one** — [`src/app.py:12`](http://x)\n"
                "🔵 **prose** — the `normalize` helper is fine\n")
        monkeypatch.setattr(prr, "gh", lambda *a, **k: json.dumps(
            [{"id": 1, "state": "CHANGES_REQUESTED", "commit_id": "oldsha",
              "user": {"login": "review-bot"}, "body": body}]))
        monkeypatch.setattr(prr, "_me", lambda: "review-bot")
        assert prr._dismiss_stale_block("app", 1, "COMMENT", "newsha", True,
                                        unread=["data/huge.jsonl"]) == [1]

    def test_a_read_single_word_filename_lets_the_block_clear(
            self, monkeypatch, pr_review):
        """The MIRROR of the previous test, and the trap SCRUM-1293 exists to
        end: a block citing only `Makefile` named no file by the shape test, so
        under truncation it could never clear even once that file HAD been
        read. Membership of the PR's own files answers it exactly."""
        prr = pr_review
        monkeypatch.setattr(prr, "gh", lambda *a, **k: json.dumps(
            [{"id": 1, "state": "CHANGES_REQUESTED", "commit_id": "oldsha",
              "user": {"login": "review-bot"},
              "body": "🔴 **the build** — `Makefile`"}]))
        monkeypatch.setattr(prr, "_me", lambda: "review-bot")
        assert prr._dismiss_stale_block(
            "app", 1, "COMMENT", "newsha", True,
            unread=["data/huge.jsonl"], pr_files=["Makefile"]) == [1]

    def test_prose_only_block_under_truncation_still_refuses(
            self, monkeypatch, pr_review):
        """A block naming nothing this PR touches cannot be checked, and an
        unparseable body under truncation is what the blanket rule was right
        about."""
        prr = pr_review
        monkeypatch.setattr(prr, "gh", lambda *a, **k: json.dumps(
            [{"id": 1, "state": "CHANGES_REQUESTED", "commit_id": "oldsha",
              "user": {"login": "review-bot"},
              "body": "🔴 **vague** — the `normalize` helper"}]))
        monkeypatch.setattr(prr, "_me", lambda: "review-bot")
        assert prr._dismiss_stale_block(
            "app", 1, "COMMENT", "newsha", True,
            unread=["x.py"], pr_files=["Makefile", "src/app.py"]) == []


class TestAHighAboutAnUnreadFileCannotBlock:
    """caeli-marketing#519. The review raised:

        🔴 New products' primary images are unmeasured, so
           test/image-dims.test.ts:129 fails — `data/image-dims.json:1`

    while its OWN caveat said:

        ⚠️ Partial review — 1 changed file(s) were NOT opened:
           `data/image-dims.json`.

    Measured at that head: the suite passed 10/10 and both ids the finding
    called absent were present. The file is over `MAX_FILE_DIFF`, so no later
    pass could reach it either, and because the head never moved
    `_dismiss_stale_block` correctly refused to clear the block. A human had to.

    The contradiction is decidable from the review's own two halves, which is
    why this is arithmetic rather than something asked of the model.
    """

    def _final(self, prr, findings, excluded):
        return prr._finalize_review(findings, [], truncated=bool(excluded),
                                    excluded=excluded, head_sha="s" * 40)

    def test_the_519_shape_does_not_request_changes(self, pr_review):
        body, event = self._final(pr_review, [
            {"file": "data/image-dims.json", "line": 1, "severity": "high",
             "title": "primary images are unmeasured", "detail": "the test fails"},
        ], ["data/image-dims.json"])
        assert event != "REQUEST_CHANGES", body

    def test_the_finding_is_kept_not_dropped(self, pr_review):
        """An unread file is not proof the finding is wrong, and deleting it
        would hide a real defect inferred from a caller the agent DID read."""
        body, _ = self._final(pr_review, [
            {"file": "data/image-dims.json", "line": 1, "severity": "high",
             "title": "primary images are unmeasured", "detail": "the test fails"},
        ], ["data/image-dims.json"])
        assert "primary images are unmeasured" in body

    def test_it_says_which_file_and_why(self, pr_review):
        body, _ = self._final(pr_review, [
            {"file": "data/image-dims.json", "line": 1, "severity": "high",
             "title": "t", "detail": "d"},
        ], ["data/image-dims.json"])
        assert "did not open" in body and "data/image-dims.json" in body

    def test_a_path_backticked_in_the_TITLE_also_demotes(self, pr_review):
        """THE ONLY DIVERGENCE between this and the dismissal guard.

        `_dismiss_stale_block` intersects `_cited_tokens(body, _BLOCKING_LINE)`
        with the unread set, and the rendered 🔴 line is
        `**{title}** — [`{file}:{line}`](…)`; `detail` and `fix` are on their
        own lines and never reach it. So a path backticked in the TITLE is read
        by the dismissal and was not read by the demotion — not demoted, so it
        blocks; blind non-empty, so it can never clear. Measured before fixing:

            dismissal reads   ['data/huge.json', 'src/app.py']
            demotion keyed on  src/app.py

        Raised by the reviewer, which attributed it to `detail` rather than the
        title; the asymmetry is real, the mechanism it gave was not.
        """
        _, event = self._final(pr_review, [
            {"file": "src/app.py", "line": 12, "severity": "high",
             "title": "the `data/huge.json` map is stale", "detail": "d"},
        ], ["data/huge.json"])
        assert event != "REQUEST_CHANGES"

    def test_a_TITLELESS_finding_demotes_on_its_synthesised_title(self, pr_review):
        """`validate_findings` does not require a `title`, and `render` then
        uses the first SENTENCE OF DETAIL as the heading. So the detail text
        reaches the 🔴 line after all, the dismissal reads a path off it and
        refuses to clear — while a demotion keyed on the raw (empty) title left
        the finding blocking. The trap, through the fallback rather than
        through a title the model wrote. Measured before fixing:

            rendered  🔴 **The `data/huge.json` map is stale.** — [`src/app.py:12`]
            dismissal ['data/huge.json', 'src/app.py']
            demotion  []  -> stayed high
        """
        f = {"file": "src/app.py", "line": 12, "severity": "high",
             "detail": "The `data/huge.json` map is stale."}
        _, event = self._final(pr_review, [dict(f)], ["data/huge.json"])
        assert event != "REQUEST_CHANGES"

        # THE APPEND MUST NOT BECOME THE HEADING. `_demote_unread_claims`
        # rewrites `detail`, and `_finding_title` synthesises the heading from
        # its FIRST SENTENCE — so the token set the demotion read and the one
        # `render` draws could drift apart if the note were ever appended
        # without its blank line, or the split became a paragraph split. It
        # holds today; nothing pinned it. Raised by the reviewer, which traced
        # it and said so rather than calling it a defect.
        #
        # MEASURED WHILE PINNING IT: NEITHER named risk alone breaks this.
        # Dropping the blank line still leaves the sentence split returning
        # the original first sentence; switching to a paragraph split still
        # leaves the note in the second paragraph. Only BOTH together move the
        # heading, and this assertion goes red on that pair. Two independent
        # protections — worth knowing before either is touched, because "it
        # holds" without saying why is what lets the second one get removed.
        before = pr_review._finding_title(f)
        out, _ = pr_review._demote_unread_claims([f], ["data/huge.json"])
        assert pr_review._finding_title(out[0]) == before
        body = pr_review.render(out, True, 0, head_sha="s" * 40, repo="x",
                                excluded=["data/huge.json"])
        assert before in body

    def test_the_demotion_reads_the_title_render_will_draw(self, pr_review):
        """One definition, asserted against `render`'s own output rather than
        restated — a second copy of the synthesis rule is what this fixes."""
        prr = pr_review
        f = {"file": "src/app.py", "line": 12, "severity": "high",
             "detail": "The `data/huge.json` map is stale. And more."}
        body = prr.render([f], True, 0, head_sha="s" * 40, repo="x",
                          excluded=["data/huge.json"])
        rendered = prr._BLOCKING_LINE.findall(body)[0]
        assert prr._finding_title(f) in rendered

    def test_the_demotion_follows_the_event_map(self, pr_review, monkeypatch):
        """Which severities block is `EVENT_BY_SEVERITY`'s fact, and
        `_blocking_pattern` already derives it for the dismissal. A literal
        here puts the halves out of step the moment a `medium` entry lands."""
        prr = pr_review
        monkeypatch.setitem(prr.EVENT_BY_SEVERITY, "medium", "REQUEST_CHANGES")
        out, demoted = prr._demote_unread_claims(
            [{"file": "a.json", "severity": "medium", "title": "t", "detail": "d"}],
            ["a.json"])
        assert demoted == ["a.json"] and out[0]["severity"] == "low"

    def test_the_demotion_announces_itself_once(self, pr_review, capsys):
        """`main` demotes before logging severities and `_finalize_review`
        demotes again so no caller can post a blocking claim about unread
        bytes. With the print at a call site the second call found nothing
        left to demote and the line fired ZERO times in a real run."""
        prr = pr_review
        f = [{"file": "a.json", "line": 1, "severity": "high",
              "title": "t", "detail": "d"}]
        once, _ = prr._demote_unread_claims(f, ["a.json"])
        prr._finalize_review(once, [], truncated=True, excluded=["a.json"],
                             head_sha="s" * 40)
        out = capsys.readouterr().out
        assert out.count("lowered to low") == 1, out

    def test_an_absolute_looking_file_still_demotes(self, pr_review):
        """`os.path.normpath` KEEPS a leading slash, and the model sometimes
        writes `/data/huge.json`. That matched nothing in the unread set, so
        the finding kept its 🔴.

        Measured: `_cited_tokens` misses it too, so the dismissal still clears
        — a demotion MISS, not the undismissable trap. (The reviewer predicted
        `blind` would be non-empty here; it is empty.) Widened only on this
        side: the dismissal guard is deployed and conservative, and making it
        stricter could keep a block it should clear.
        """
        _, event = self._final(pr_review, [
            {"file": "/data/huge.json", "line": 1, "severity": "high",
             "title": "t", "detail": "d"},
        ], ["data/huge.json"])
        assert event != "REQUEST_CHANGES"

    def test_the_demotion_is_a_superset_of_what_the_dismissal_holds(self, pr_review):
        """The asymmetry runs ONE way, and that is the safe way.

        Demoting something the dismissal would not have held costs a 🔴 that
        becomes a 🔵. The reverse — the dismissal holding a block this side
        left blocking — is the trap. `/data/huge.json` is the live case: this
        side strips the leading slash, `_cited_tokens` does not.
        """
        prr = pr_review
        import os
        f = {"file": "/data/huge.json", "line": 1, "severity": "high",
             "title": "t", "detail": "d"}
        unread = {os.path.normpath("data/huge.json")}
        body = prr.render([f], True, 0, head_sha="s" * 40, repo="x",
                          excluded=["data/huge.json"])
        held = prr._cited_tokens(body, prr._BLOCKING_LINE) & unread
        _, demoted = prr._demote_unread_claims([f], ["data/huge.json"])
        assert demoted and not held, (
            "the demotion must be the wider side; if the dismissal ever holds "
            "something this does not demote, that is the unclearable block")

    def test_the_note_names_the_findings_OWN_file_when_that_is_the_unread_one(
            self, pr_review):
        """`cited` also holds the title's backticks, so the alphabetically
        first match could name a path the finding is not about — telling the
        reader the wrong file is why the severity dropped."""
        body, _ = self._final(pr_review, [
            {"file": "zz/mine.json", "line": 1, "severity": "high",
             "title": "the `aa/other.json` map is stale", "detail": "d"},
        ], ["zz/mine.json", "aa/other.json"])
        assert "did not open `zz/mine.json`" in body, body

    def test_the_note_falls_back_when_the_findings_own_file_WAS_read(
            self, pr_review):
        body, _ = self._final(pr_review, [
            {"file": "src/app.py", "line": 1, "severity": "high",
             "title": "the `aa/other.json` map is stale", "detail": "d"},
        ], ["aa/other.json"])
        assert "did not open `aa/other.json`" in body, body

    def test_one_derivation_of_detail(self, pr_review):
        """`render` draws it, `_finding_title` falls back to its first
        sentence, and the demotion reads that heading. The demotion used to
        call `_finding_title(f)` while `render` passed its own copy —
        identical today, so a change to either would have diverged silently."""
        prr = pr_review
        f = {"file": "a.py", "line": 1, "severity": "high",
             "detail": "  First `x/y.json` sentence.  Second one.  "}
        body = prr.render([f], False, 0, head_sha="s" * 40, repo="x")
        assert prr._finding_detail(f) in body
        assert prr._finding_title(f) in prr._BLOCKING_LINE.findall(body)[0]

    def test_a_finding_with_no_file_is_untouched(self, pr_review):
        """`_where_link` renders `_the pull request_`, so a PR-level finding —
        a missing ticket id, an unsigned agent commit — puts no path on the 🔴
        line. It is not a claim about unread bytes and must keep its
        severity."""
        out, demoted = pr_review._demote_unread_claims(
            [{"file": "", "severity": "high", "title": "no ticket id",
              "detail": "d"}], ["data/huge.json"])
        assert demoted == [] and out[0]["severity"] == "high"

    def test_detail_alone_does_not_demote(self, pr_review):
        """`detail` is not on the 🔴 line, so the dismissal never reads it
        either. Demoting on it would make the two halves disagree again, in the
        other direction, and would mute real findings for mentioning a big
        file in prose."""
        _, event = self._final(pr_review, [
            {"file": "src/app.py", "line": 12, "severity": "high",
             "title": "a real defect",
             "detail": "this also loads `data/huge.json` at boot"},
        ], ["data/huge.json"])
        assert event == "REQUEST_CHANGES"

    def test_the_demoted_list_reaches_the_caller(self, monkeypatch, pr_review):
        """`_finalize_review` rebinds only its own local, so `main`'s log line
        and the commit status reported the PRE-demotion severities — a run that
        posted a 🔵 printed "1 high" and set a status to match."""
        prr = pr_review
        f = [{"file": "data/image-dims.json", "line": 1, "severity": "high",
              "title": "t", "detail": "d"}]
        out, demoted = prr._demote_unread_claims(f, ["data/image-dims.json"])
        assert demoted == ["data/image-dims.json"]
        assert prr.severity_breakdown(out) == prr.severity_breakdown(
            [dict(f[0], severity="low")])
        assert f[0]["severity"] == "high", "the input list must not be mutated"

    def test_a_high_about_a_file_that_WAS_read_still_blocks(self, pr_review):
        """The guard must not disarm the reviewer. Only the unread file's
        finding moves."""
        body, event = self._final(pr_review, [
            {"file": "src/app.py", "line": 12, "severity": "high",
             "title": "a real defect", "detail": "d"},
        ], ["data/image-dims.json"])
        assert event == "REQUEST_CHANGES", body

    def test_a_nit_about_an_unread_file_is_left_alone(self, pr_review):
        """Only `high` posts REQUEST_CHANGES, so only `high` is the authority
        this takes away. Rewriting a 🔵 would be noise."""
        body, _ = self._final(pr_review, [
            {"file": "data/image-dims.json", "line": 1, "severity": "low",
             "title": "t", "detail": "original detail"},
        ], ["data/image-dims.json"])
        assert "Severity lowered automatically" not in body

    def test_nothing_excluded_changes_nothing(self, pr_review):
        body, event = self._final(pr_review, [
            {"file": "data/image-dims.json", "line": 1, "severity": "high",
             "title": "t", "detail": "d"},
        ], [])
        assert event == "REQUEST_CHANGES"
        assert "Severity lowered automatically" not in body

    def test_the_paths_are_compared_normalised(self, pr_review):
        """`unopened` comes from the diff and the finding's `file` from the
        model; `./a/b.json` and `a/b.json` are the same file."""
        _, event = self._final(pr_review, [
            {"file": "./data/image-dims.json", "line": 1, "severity": "high",
             "title": "t", "detail": "d"},
        ], ["data/image-dims.json"])
        assert event != "REQUEST_CHANGES"

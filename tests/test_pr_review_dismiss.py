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
                            unread=(): (seen.update(unread=list(unread)), ev)[1])
        monkeypatch.setattr(prr, "gh", lambda *a, **k: json.dumps(
            {"draft": False, "state": "open", "merged": False, "title": "SCRUM-1 x",
             "user": {"login": "someone"}, "head": {"sha": "a" * 40}}))
        monkeypatch.setattr(prr.sys, "argv", ["pr-review", "repo", "1"])
        monkeypatch.delenv("DRY", raising=False)
        prr.main()
        assert seen["unread"] == ["data/huge.jsonl"]

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
                            unread=(): (seen.update(unread=list(unread),
                                                    truncated=truncated), ev)[1])
        monkeypatch.setattr(prr.status, "done", lambda *a: None)
        monkeypatch.setattr(prr, "gh", lambda *a, **k: json.dumps(
            {"draft": False, "state": "open", "merged": False, "title": "SCRUM-1 x",
             "user": {"login": "someone"}, "head": {"sha": "e" * 40}}))
        monkeypatch.setattr(prr.sys, "argv", ["pr-review", "repo", "1"])
        monkeypatch.delenv("DRY", raising=False)
        prr.main()
        assert seen["unread"] == ["data/huge.jsonl"] and seen["truncated"] is True

"""A superseded run is not a broken one (2026-08-25).

GitHub Actions cancels an in-flight job by SIGKILLing the process tree. Under
the old hermes subprocess that arrived as `hermes exited -9` — indistinguishable
at the subprocess level from a crash — was raised as `AgentFailed`, and paged a
human with "pr-review BROKEN".

That fires on the most ordinary path this tool has. `cancel-in-progress` is
deliberate: asking for a re-review while one is running should supersede it, and
opening a PR then requesting a reviewer does exactly that within seconds.
Measured on portal-api#140:

    15:17:24  agent exploring the checkout (timeout 900s)…
    15:18:05  agent finished in 39s
    15:18:05  no findings — asking it to show its work before approving
    15:18:22  ##[error]The operation was canceled.

Two healthy reviews and one page.

WHAT THE SIGNAL BECAME. The loop is in-process now, so a cancel kills this
process outright and there is nothing left to classify. What survives is the
OTHER half of the same problem: a run whose clock expires while the run that
replaced it is already going. That is still a supersession, and the successor
probe is still what tells the two apart — so every test below the fixture is
unchanged, because what counts as a successor never depended on hermes.
"""
import json
from pathlib import Path

import pytest

from conftest import load_script


@pytest.fixture(scope="module")
def prr():
    return load_script("pr-review")


@pytest.fixture
def run_with(prr, monkeypatch):
    """Drive `_run_agent` with a scripted loop outcome.

    `superseded` says whether a newer run for this PR exists — the thing that
    separates a cancellation from a failure.
    """
    def drive(outcome, superseded=True):
        def fake_run(system, user, root, **kw):
            if isinstance(outcome, Exception):
                raise outcome
            return outcome, ["turn 1: answered"]

        monkeypatch.setattr(prr.agent, "run", fake_run)
        monkeypatch.setattr(prr, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(prr, "_superseding_run_exists", lambda *a: superseded)
        return prr._run_agent("prompt", cwd="/tmp", timeout=10, repo="infra", pr="1")
    return drive


class TestACancelledRunIsNotAFailure:
    def test_expiry_with_a_successor_raises_superseded(self, prr, run_with):
        """The clock ran out while a newer run for this PR was already going:
        somebody else is posting the review."""
        with pytest.raises(prr.Superseded) as e:
            run_with(prr.agent.Timeout("deadline of 900s exhausted", ["turn 4"]),
                     superseded=True)
        assert "900s" in str(e.value)

    def test_expiry_with_NO_successor_alerts(self, prr, run_with):
        """The review's finding on infra#112, and it was right. A run that
        simply ran out of time with nothing to finish the job must not be
        demoted — that loses a review in silence."""
        with pytest.raises(prr.AgentFailed) as e:
            run_with(prr.agent.Timeout("deadline of 900s exhausted", ["turn 4"]),
                     superseded=False)
        assert "exhausted" in str(e.value)

    def test_a_PROVIDER_failure_is_never_demoted_to_superseded(self, prr, run_with):
        """The trap the signal version had: an outage and a cancellation looked
        identical, so anything that could be either was guessed at. An
        AgentError is the model failing to answer and stays an alert EVEN IF a
        successor exists — the successor will hit the same outage."""
        with pytest.raises(prr.AgentFailed) as e:
            run_with(prr.agent.AgentError("turn 2: fireworks -> 503", []),
                     superseded=True)
        assert "503" in str(e.value)

    def test_superseded_is_not_a_scanerror(self, prr):
        """THE property. `guard_main` alerts on ReviewError, so inheriting from
        it would page a human for a cancellation no matter what else we did."""
        assert not issubclass(prr.Superseded, prr.ReviewError)
        assert not issubclass(prr.Superseded, prr.AgentFailed)


class TestARealFailureStillAlerts:
    def test_a_provider_error_is_still_agentfailed(self, prr, run_with):
        """Flake protection must not become failure suppression. A provider
        outage still has to reach a human."""
        with pytest.raises(prr.AgentFailed) as e:
            run_with(prr.agent.AgentError("turn 1: fireworks -> 500", []),
                     superseded=False)
        assert "500" in str(e.value)

    def test_agentfailed_is_still_a_scanerror(self, prr):
        assert issubclass(prr.AgentFailed, prr.ReviewError)

    def test_a_clean_run_returns_the_answer(self, prr, monkeypatch):
        monkeypatch.setattr(prr.agent, "run",
                            lambda *a, **k: ("  answer  ", ["turn 1"]))
        monkeypatch.setattr(prr, "_pr_is_gone", lambda *a: None)
        assert prr._run_agent("p", cwd="/tmp", timeout=10) == "answer"

    def test_a_failure_prints_the_transcript(self, prr, monkeypatch, capsys):
        """The entire reason for the rewrite. A failure that says only "timed
        out after 900s" is the state this replaced."""
        monkeypatch.setattr(prr, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(prr, "_superseding_run_exists", lambda *a: False)

        def fake_run(*a, **k):
            raise prr.agent.Timeout("deadline exhausted", [
                "turn 3: 1 tool call(s) in 12.0s",
                "  grep(pattern='def handler') -> 900 chars in 0.30s"])
        monkeypatch.setattr(prr.agent, "run", fake_run)
        with pytest.raises(prr.AgentFailed):
            prr._run_agent("p", cwd="/tmp", timeout=10, repo="infra", pr="1")
        out = capsys.readouterr().out
        assert "turn 3" in out and "def handler" in out


class TestTheEntrypoint:
    def test_superseded_exits_nonzero_without_alerting(self, prr, monkeypatch):
        """Exit 1 so the Actions job stays red — this run produced no review, and
        saying otherwise would be dishonest. It simply must not page."""
        monkeypatch.setattr(prr, "main",
                            lambda: (_ for _ in ()).throw(prr.Superseded("signal 9")))
        with pytest.raises(SystemExit) as e:
            prr._main_unless_superseded()
        assert e.value.code == 1

    def test_a_real_error_propagates_to_guard_main(self, prr, monkeypatch):
        """Anything that is not a cancellation must still reach the alerting
        handler, unchanged."""
        monkeypatch.setattr(prr, "main",
                            lambda: (_ for _ in ()).throw(prr.AgentFailed("hermes exited 1")))
        with pytest.raises(prr.AgentFailed):
            prr._main_unless_superseded()


class TestWhatCountsAsASuccessor:
    """`_superseding_run_exists` decides whether a -9 is quiet, so every way it
    can be wrong ends with a lost review or a false page. It FAILS TOWARD
    ALERTING: an unanswerable question is a "no"."""

    def _runs(self, prr, monkeypatch, payload, run_id="999", draft=False):
        monkeypatch.setenv("GITHUB_RUN_ID", run_id)

        def fake_gh(path, *a, **k):
            if "/actions/runs" in path:
                return json.dumps(payload)
            return json.dumps({"draft": draft})     # the PR itself

        monkeypatch.setattr(prr, "gh", fake_gh)
        return prr._superseding_run_exists("infra", "112")

    def _run(self, rid, number=112, name="PR review", status="in_progress",
             path=".github/workflows/pr-review-caller.yml"):
        return {"id": rid, "name": name, "status": status, "path": path,
                "pull_requests": [{"number": number}]}

    def test_a_newer_run_on_this_pr_counts(self, prr, monkeypatch):
        assert self._runs(prr, monkeypatch, {"workflow_runs": [self._run(1000)]})

    def test_an_older_run_does_not(self, prr, monkeypatch):
        """It cannot post for us — it started first and we outlived it."""
        assert not self._runs(prr, monkeypatch, {"workflow_runs": [self._run(500)]})

    def test_a_newer_run_on_a_different_pr_does_not(self, prr, monkeypatch):
        """The busiest failure mode on a shared runner: something else is always
        starting. Only OUR successor can post OUR review."""
        assert not self._runs(prr, monkeypatch,
                              {"workflow_runs": [self._run(1000, number=999)]})

    def test_a_different_workflow_does_not(self, prr, monkeypatch):
        assert not self._runs(prr, monkeypatch, {"workflow_runs": [
            self._run(1000, name="Tests", path=".github/workflows/test.yml")]})

    def test_it_matches_on_path_even_if_the_name_is_changed(self, prr, monkeypatch):
        """`name:` is cosmetic. One repo editing it must not silently stop that
        repo's supersession from being recognised."""
        assert self._runs(prr, monkeypatch, {"workflow_runs": [
            self._run(1000, name="Renamed By Someone")]})

    def test_it_still_matches_on_name_if_the_path_moves(self, prr, monkeypatch):
        assert self._runs(prr, monkeypatch, {"workflow_runs": [
            self._run(1000, path=".github/workflows/elsewhere.yml")]})

    def test_a_finished_run_does_not(self, prr, monkeypatch):
        """Already completed, so it is not going to post anything for us."""
        assert not self._runs(prr, monkeypatch,
                              {"workflow_runs": [self._run(1000, status="completed")]})

    def test_no_run_id_in_the_environment_means_no(self, prr, monkeypatch):
        monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
        assert not prr._superseding_run_exists("infra", "112")

    def test_an_api_failure_means_no(self, prr, monkeypatch):
        """Fail toward the page. An unknown must never buy silence."""
        monkeypatch.setenv("GITHUB_RUN_ID", "999")
        monkeypatch.setattr(prr, "gh",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("403")))
        assert not prr._superseding_run_exists("infra", "112")


class TestTheWorkflowIdentityIsNotJustAHardcodedString:
    """The AI review's second point on infra#112, and it was the better half.

    Every other test here mocks `gh` with a payload that hardcodes the same
    workflow name the code looks for, so the whole suite passes whether or not
    that string matches what GitHub actually returns — the classic
    passes-with-or-without-the-fix shape.

    These read the REAL workflow file instead. Verified against the live API
    while fixing this: a run of ours comes back as
    `name=PR review, path=.github/workflows/pr-review-caller.yml`, and the
    reusable workflow (name "AI PR review") gets no run entry at all.
    """

    # Shipped as an example: this repository has no runner to call it with.
    CALLER = Path(__file__).resolve().parents[1] / "examples/pr-review-caller.yml"
    REUSABLE = Path(__file__).resolve().parents[1] / ".github/workflows/pr-review.yml"

    def test_the_path_constant_points_at_a_file_that_exists(self, prr):
        assert self.CALLER.exists(), self.CALLER
        # The constant names where a CALLER installs it, not where the example lives.
        assert prr.CALLER_PATH == ".github/workflows/" + self.CALLER.name

    def test_the_name_constant_matches_the_caller_workflow(self, prr):
        declared = next(l.split(":", 1)[1].strip()
                        for l in self.CALLER.read_text().splitlines()
                        if l.startswith("name:"))
        assert declared == prr.CALLER_NAME, (
            f"caller workflow is named {declared!r} but the successor check "
            f"looks for {prr.CALLER_NAME!r}")

    def test_it_is_not_the_reusable_workflows_name(self, prr):
        """The trap the review found: `pr-review.yml` is "AI PR review", and it
        is the file that runs the job — but it never appears in the runs list."""
        reusable = self.REUSABLE
        declared = next(l.split(":", 1)[1].strip()
                        for l in reusable.read_text().splitlines()
                        if l.startswith("name:"))
        assert declared != prr.CALLER_NAME


class TestAWillSkipSuccessorIsNotASuccessor:
    """The AI review's third finding on infra#112, and the subtlest.

    The concurrency group keys on the reviewer-request arm of the job's `if` but
    NOT on the draft arm — verified: zero mentions of draft in the group, one in
    the gate. So on a PR that goes draft mid-review:

        run N reviewing PR 12 sits in the -active group
        author converts 12 to draft
        review_requested for review-bot fires run M, same -active group
        cancel-in-progress kills N with -9
        M's job gate `draft == false` is false -> M SKIPS, posts nothing

    Counting M as a successor buys exactly the silence this predicate exists to
    prevent. It is the same will-skip-cancels-real shape the workflow's own
    comment records for caeli-marketing#181, where a skipping Copilot run took a
    real review down with it.
    """

    def _run(self, rid):
        return {"id": rid, "name": "PR review", "status": "in_progress",
                "path": ".github/workflows/pr-review-caller.yml",
                "pull_requests": [{"number": 112}]}

    def _ask(self, prr, monkeypatch, draft):
        monkeypatch.setenv("GITHUB_RUN_ID", "999")

        def fake_gh(path, *a, **k):
            if "/actions/runs" in path:
                return json.dumps({"workflow_runs": [self._run(1000)]})
            return json.dumps({"draft": draft})

        monkeypatch.setattr(prr, "gh", fake_gh)
        return prr._superseding_run_exists("infra", "112")

    def test_a_successor_on_a_draft_pr_does_not_count(self, prr, monkeypatch):
        assert not self._ask(prr, monkeypatch, draft=True)

    def test_a_successor_on_a_live_pr_still_counts(self, prr, monkeypatch):
        """The guard must not swallow the ordinary case it was built for."""
        assert self._ask(prr, monkeypatch, draft=False)

    def test_an_unanswerable_draft_state_means_no_successor(self, prr, monkeypatch):
        """Fail toward the page, like every other branch here."""
        monkeypatch.setenv("GITHUB_RUN_ID", "999")

        def fake_gh(path, *a, **k):
            if "/actions/runs" in path:
                return json.dumps({"workflow_runs": [self._run(1000)]})
            raise RuntimeError("500")

        monkeypatch.setattr(prr, "gh", fake_gh)
        assert not prr._superseding_run_exists("infra", "112")

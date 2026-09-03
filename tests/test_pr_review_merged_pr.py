"""A merged PR is not worth reviewing, and must not be alerted about.

Three windows, because the PR can merge in any of them and only one was
covered before — badly, as a 422 at POST time after the agent had already run:

  1. BEFORE the run starts (queued behind another review on the single runner).
  2. DURING the agent run — the expensive one. `subprocess.run` blocked until
     the timeout, so a PR merged two minutes into a twelve-minute review held
     the runner for the other ten.
  3. BETWEEN the agent finishing and the POST.

The opposite error is the dangerous one: aborting a review of a LIVE PR loses
it silently. So every abort test here has a partner proving an open PR is
reviewed, and the probe fails toward continuing whenever it cannot get an
answer.
"""
import json
import subprocess
import sys
import pathlib

import itertools

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
from conftest import load_script  # noqa: E402


@pytest.fixture
def prr():
    return load_script("pr-review")


def _state(monkeypatch, m, **fields):
    monkeypatch.setattr(m, "gh", lambda path, **k: json.dumps(fields))


class TestTheProbe:
    def test_a_merged_pr_is_gone(self, prr, monkeypatch):
        _state(monkeypatch, prr, merged=True, state="closed")
        assert prr._pr_is_gone("r", 1) == "merged"

    def test_a_closed_pr_is_gone(self, prr, monkeypatch):
        _state(monkeypatch, prr, merged=False, state="closed")
        assert prr._pr_is_gone("r", 1) == "closed"

    def test_an_OPEN_pr_is_not(self, prr, monkeypatch):
        _state(monkeypatch, prr, merged=False, state="open")
        assert prr._pr_is_gone("r", 1) is None

    def test_an_unreadable_state_CONTINUES_the_review(self, prr, monkeypatch):
        """Killing a good review to avoid wasted work is a worse trade than the
        waste. A 5xx must not abort anything."""
        def boom(path, **k):
            raise RuntimeError("502")
        monkeypatch.setattr(prr, "gh", boom)
        assert prr._pr_is_gone("r", 1) is None


class TestTheAgentAborts:
    """Window 2 — the one that actually costs runner minutes.

    Under the hermes subprocess this needed a poll and a kill. The loop is
    in-process now, so the check is a hook the loop calls BETWEEN TURNS: the
    review stops at the next turn boundary instead of at the next poll, and
    there is no process left running to leak.
    """

    def _fake_agent(self, prr, monkeypatch, turns=3):
        """Stand in for `agent.run`, calling `on_turn` the way the real loop does."""
        seen = {"turns": 0}

        def run(system, user, root, on_turn=None, **kw):
            for turn in range(1, turns + 1):
                seen["turns"] = turn
                if on_turn is not None:
                    on_turn(turn)
            return "answer", ["turn 1: answered"]

        monkeypatch.setattr(prr.agent, "run", run)
        return seen

    def test_a_merge_mid_run_aborts_the_agent(self, prr, monkeypatch):
        seen = self._fake_agent(prr, monkeypatch)
        monkeypatch.setattr(prr, "_pr_is_gone", lambda *a: "merged")
        monkeypatch.setattr(prr, "PR_STATE_POLL", 0)
        with pytest.raises(prr.PRClosed) as e:
            prr._run_agent("p", cwd="/tmp", timeout=5, repo="r", pr="1")
        assert "merged" in str(e.value)
        assert seen["turns"] == 1, "the loop kept going on a merged PR"

    def test_an_OPEN_pr_runs_to_completion(self, prr, monkeypatch):
        """The partner. Polling must not cost a review of a live PR."""
        seen = self._fake_agent(prr, monkeypatch)
        monkeypatch.setattr(prr, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(prr, "PR_STATE_POLL", 0)
        assert prr._run_agent("p", cwd="/tmp", timeout=5, repo="r", pr="1") == "answer"
        assert seen["turns"] == 3

    def test_the_state_probe_is_RATE_LIMITED_not_per_turn(self, prr, monkeypatch):
        """A GitHub call on every turn would be ~40 extra API calls per review
        for a state that changes at most once. `PR_STATE_POLL` bounds it.

        The clock is FAKED: this asserted against the real `time.monotonic()`
        and passed only on a box up for more than an hour — the sentinel was
        `0.0`, so a fresh runner never probed at all. The first CI run of this
        suite was the first machine to notice."""
        self._fake_agent(prr, monkeypatch, turns=6)
        probes = []
        monkeypatch.setattr(prr, "_pr_is_gone",
                            lambda *a: probes.append(1) and None)
        monkeypatch.setattr(prr, "PR_STATE_POLL", 3600)
        clock = {"now": 5.0}                      # a box up for five seconds
        monkeypatch.setattr(prr.time, "monotonic", lambda: clock["now"])
        prr._run_agent("p", cwd="/tmp", timeout=5, repo="r", pr="1")
        assert len(probes) == 1, f"probed {len(probes)} times in 6 turns"

    def test_the_probe_fires_again_once_the_interval_has_passed(self, prr, monkeypatch):
        probes = []
        # Each read of the clock is 61s later than the last; between_turns
        # reads it once per turn, so every turn is past the 60s interval.
        ticks = itertools.count(start=5.0, step=61.0)
        monkeypatch.setattr(prr.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(prr, "_pr_is_gone", lambda *a: probes.append(1) and None)
        monkeypatch.setattr(prr, "PR_STATE_POLL", 60)
        self._fake_agent(prr, monkeypatch, turns=4)
        prr._run_agent("p", cwd="/tmp", timeout=5, repo="r", pr="1")
        assert len(probes) == 4


class TestItIsNotAnAlert:
    def test_PRClosed_exits_zero(self, prr, monkeypatch):
        """A red check on a merged PR is a complaint about nothing — and this
        repo just spent a PR learning that a spurious red reads as a real one."""
        monkeypatch.setattr(prr, "main", lambda: (_ for _ in ()).throw(
            prr.PRClosed("the PR was merged while the review ran")))
        with pytest.raises(SystemExit) as e:
            prr._main_unless_superseded()
        assert e.value.code == 0

    def test_a_superseded_run_still_exits_one(self, prr, monkeypatch):
        """Unchanged, and the contrast is the point: that run owed a review and
        did not produce one, so red is honest there."""
        monkeypatch.setattr(prr, "main", lambda: (_ for _ in ()).throw(
            prr.Superseded("killed by signal 9")))
        with pytest.raises(SystemExit) as e:
            prr._main_unless_superseded()
        assert e.value.code == 1

    def test_PRClosed_is_not_a_ScanError(self, prr):
        """A ScanError reaches guard_main's alerting path. This must not."""
        assert not issubclass(prr.PRClosed, prr.ReviewError)

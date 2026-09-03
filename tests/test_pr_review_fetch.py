"""The fetch retries, and says what went wrong (2026-08-29).

slack-app#367 lost a whole review to one TCP connect:

    fatal: unable to access 'https://github.com/example/app.git/':
    Failed to connect to github.com port 443 after 405 ms: Couldn't connect to
    server

The box was fine two seconds either side — the reviewer fetch succeeded at
01:07:33 and the diff at 01:07:35, this failed at 01:07:36. Every other
network-dependent step in this module already retries a transient; the git
fetch, the MOST network-dependent of them, did not.

AND THE ALERT NAMED THE WRONG THING. `CalledProcessError` escaped raw, so the
page read "pr-review crashed: CalledProcessError … exit status 128", which
reads as a bug in the reviewer. git's stderr — the one line that explained it —
reached the run log and never the alert. Two wrong theories were chased (a
deleted SHA, then a force-push) before anybody read it.
"""
import subprocess

import pytest

from conftest import load_script


@pytest.fixture(scope="module")
def prr():
    return load_script("pr-review")


def _result(code, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=code, stdout="", stderr=stderr)


NET = ("fatal: unable to access 'https://github.com/example/app.git/': "
       "Failed to connect to github.com port 443 after 405 ms: Couldn't connect to server")


class TestItRetriesOnce:
    def test_a_transient_failure_then_success_is_a_success(self, prr, monkeypatch):
        """THE reported case: one blip must not cost the review."""
        calls = []

        def run(args, **kw):
            calls.append(args)
            return _result(0) if len(calls) > 1 else _result(128, NET)

        monkeypatch.setattr(prr.subprocess, "run", run)
        monkeypatch.setattr(prr.time, "sleep", lambda s: None)
        prr._fetch_head("/tmp/x", "c3ef86eff13c", {})
        assert len(calls) == 2

    def test_it_does_not_retry_a_success(self, prr, monkeypatch):
        calls = []
        monkeypatch.setattr(prr.subprocess, "run",
                            lambda args, **kw: calls.append(args) or _result(0))
        prr._fetch_head("/tmp/x", "abc1234", {})
        assert len(calls) == 1

    def test_it_gives_up_after_the_retry(self, prr, monkeypatch):
        """Bounded. A permanent fault fails the same way twice, and the second
        attempt costs seconds to prove it."""
        calls = []
        monkeypatch.setattr(prr.subprocess, "run",
                            lambda args, **kw: calls.append(args) or _result(128, NET))
        monkeypatch.setattr(prr.time, "sleep", lambda s: None)
        with pytest.raises(prr.ReviewError):
            prr._fetch_head("/tmp/x", "abc1234", {})
        assert len(calls) == 2


class TestTheErrorExplainsItself:
    @pytest.fixture
    def failing(self, prr, monkeypatch):
        def fail(stderr):
            monkeypatch.setattr(prr.subprocess, "run",
                                lambda args, **kw: _result(128, stderr))
            monkeypatch.setattr(prr.time, "sleep", lambda s: None)
            with pytest.raises(prr.ReviewError) as e:
                prr._fetch_head("/tmp/x", "c3ef86eff13c", {})
            return str(e.value)
        return fail

    def test_it_raises_a_ScanError_not_a_CalledProcessError(self, prr, monkeypatch):
        """The whole point. `CalledProcessError` reaching guard_main reports
        'crashed', which reads as a defect in the reviewer rather than a
        network fault, and carries none of git's stderr."""
        monkeypatch.setattr(prr.subprocess, "run", lambda args, **kw: _result(128, NET))
        monkeypatch.setattr(prr.time, "sleep", lambda s: None)
        with pytest.raises(prr.ReviewError):
            prr._fetch_head("/tmp/x", "abc1234", {})

    def test_it_carries_gits_own_words(self, failing):
        assert "Couldn't connect to server" in failing(NET)

    def test_it_names_a_network_fault_as_one(self, failing):
        """The sentence that would have saved the two wrong theories."""
        msg = failing(NET)
        assert "unreachable" in msg
        assert "rather than anything about this PR" in msg

    def test_it_names_the_sha_it_wanted(self, failing):
        assert "c3ef86e" in failing(NET)

    @pytest.mark.parametrize("stderr", [
        "fatal: could not resolve host: github.com",
        "fatal: unable to access '...': Connection timed out after 30001 ms",
        "error: RPC failed; curl 56 Recv failure: Connection reset by peer",
        "fatal: the remote end hung up unexpectedly",
    ])
    def test_other_network_shapes_are_recognised(self, failing, stderr):
        assert "unreachable" in failing(stderr)

    def test_a_REAL_git_error_is_not_called_a_network_fault(self, failing):
        """Misfiling a genuine repository problem as 'the network' would send
        the reader looking at the wrong thing — the exact cost this fixes."""
        msg = failing("fatal: couldn't find remote ref deadbeef")
        assert "unreachable" not in msg
        assert "couldn't find remote ref" in msg

    def test_an_empty_stderr_still_produces_a_usable_message(self, prr, monkeypatch):
        monkeypatch.setattr(prr.subprocess, "run", lambda args, **kw: _result(128, ""))
        monkeypatch.setattr(prr.time, "sleep", lambda s: None)
        with pytest.raises(prr.ReviewError) as e:
            prr._fetch_head("/tmp/x", "abc1234", {})
        assert "exit 128" in str(e.value)


class TestCheckoutActuallyUsesIt:
    """The helper alone proves nothing.

    Mutation-tested: replacing `_fetch_head(...)` in `checkout` with the old
    `subprocess.run(..., check=True)` — i.e. restoring the exact raw
    CalledProcessError that produced the misleading "pr-review crashed" page —
    passed all 13 tests above. Every one of them called `_fetch_head` directly.

    That is the fifth time in this suite's history the gap was the WIRING rather
    than the logic. Test the call site.
    """

    @pytest.fixture
    def stub_git(self, prr, monkeypatch):
        """Every git call succeeds except the fetch."""
        monkeypatch.setattr(prr.github, "token", lambda **k: "t")
        monkeypatch.setattr(prr.time, "sleep", lambda s: None)

        def run(args, **kw):
            if "fetch" in args:
                return _result(128, NET)
            return _result(0)

        monkeypatch.setattr(prr.subprocess, "run", run)

    def test_a_failed_fetch_from_checkout_is_a_ScanError(self, prr, stub_git, tmp_path):
        """Not a CalledProcessError. guard_main turns that into 'crashed', which
        is what sent two people down two wrong theories."""
        with pytest.raises(prr.ReviewError) as e:
            prr.checkout("slack-app", "c3ef86eff13c", str(tmp_path))
        assert "unreachable" in str(e.value)
        assert "Couldn't connect to server" in str(e.value)

    def test_it_does_not_leak_a_CalledProcessError(self, prr, stub_git, tmp_path):
        try:
            prr.checkout("slack-app", "c3ef86eff13c", str(tmp_path))
        except prr.ReviewError:
            pass
        except subprocess.CalledProcessError:  # noqa: TRY302
            pytest.fail("checkout still lets a raw CalledProcessError escape")

    def test_the_retry_happens_from_checkout_too(self, prr, monkeypatch, tmp_path):
        """The retry has to be on the path checkout takes, not only on the
        helper called directly."""
        fetches = []

        def run(args, **kw):
            if "fetch" in args:
                fetches.append(args)
                return _result(0) if len(fetches) > 1 else _result(128, NET)
            return _result(0)

        monkeypatch.setattr(prr.github, "token", lambda **k: "t")
        monkeypatch.setattr(prr.time, "sleep", lambda s: None)
        monkeypatch.setattr(prr.subprocess, "run", run)
        prr.checkout("slack-app", "c3ef86eff13c", str(tmp_path))
        assert len(fetches) == 2, "checkout did not get the retry"

    def test_the_remote_is_still_dropped_when_the_fetch_fails(self, prr, stub_git,
                                                              monkeypatch, tmp_path):
        """The `finally` must survive the new exception type: a credential in
        the directory handed to the agent is the thing that cleanup exists for."""
        seen = []
        inner = prr.subprocess.run
        monkeypatch.setattr(prr.subprocess, "run",
                            lambda args, **kw: seen.append(args) or inner(args, **kw))
        with pytest.raises(prr.ReviewError):
            prr.checkout("slack-app", "c3ef86eff13c", str(tmp_path))
        assert any("remote" in a and "remove" in a for a in seen)


class TestAHungFetchIsRetriedToo:
    """The gap the AI review found in the retry itself.

    `subprocess.run(timeout=...)` RAISES `TimeoutExpired` rather than returning
    a non-zero result, so the loop — which only inspected `returncode` — never
    saw a hang. Two consequences, both the opposite of what this change is for:

      * no retry, on the one network failure a retry is most likely to fix;
      * the exception is not a ScanError, so it reaches guard_main's generic
        handler and is announced "pr-review crashed" — the misleading page this
        function exists to remove, reached by the other door.

    The module already guards its own hermes subprocess this way. This is the
    same rule, one call over, and every test above passed with the gap present
    because they all stub a RETURN rather than a raise.
    """

    def test_a_hung_fetch_is_retried(self, prr, monkeypatch):
        calls = []

        def run(args, **kw):
            calls.append(args)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(cmd=args, timeout=prr.FETCH_TIMEOUT)
            return _result(0)

        monkeypatch.setattr(prr.subprocess, "run", run)
        monkeypatch.setattr(prr.time, "sleep", lambda s: None)
        prr._fetch_head("/tmp/x", "c3ef86eff13c", {})
        assert len(calls) == 2, "a hung fetch got no second attempt"

    def test_a_persistent_hang_raises_a_ScanError_not_TimeoutExpired(self, prr,
                                                                     monkeypatch):
        """The alert must name the network, not crash."""
        monkeypatch.setattr(prr.subprocess, "run",
                            lambda a, **k: (_ for _ in ()).throw(
                                subprocess.TimeoutExpired(cmd=a, timeout=300)))
        monkeypatch.setattr(prr.time, "sleep", lambda s: None)
        with pytest.raises(prr.ReviewError) as e:
            prr._fetch_head("/tmp/x", "c3ef86eff13c", {})
        msg = str(e.value)
        assert "timed out" in msg
        assert "unreachable" in msg, "a hang is a network fault and should say so"

    def test_it_does_not_escape_from_checkout_either(self, prr, monkeypatch,
                                                     tmp_path):
        """End to end, since a raw TimeoutExpired out of `checkout` is exactly
        the 'crashed' page this PR removes."""
        monkeypatch.setattr(prr.github, "token", lambda **k: "t")
        monkeypatch.setattr(prr.time, "sleep", lambda s: None)

        def run(args, **kw):
            if "fetch" in args:
                raise subprocess.TimeoutExpired(cmd=args, timeout=300)
            return _result(0)

        monkeypatch.setattr(prr.subprocess, "run", run)
        try:
            prr.checkout("slack-app", "c3ef86eff13c", str(tmp_path))
        except prr.ReviewError:
            pass
        except subprocess.TimeoutExpired:
            pytest.fail("a hung fetch still escapes as TimeoutExpired")

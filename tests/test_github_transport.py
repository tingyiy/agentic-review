"""One retry for a dropped connection to GitHub, and nothing else."""
import http.client
import io
import urllib.error
import urllib.request

import pytest

from agentic_review import github


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _urlopen_sequence(monkeypatch, outcomes):
    calls = []

    def urlopen(req, timeout=None):
        calls.append(req)
        out = outcomes[len(calls) - 1]
        if isinstance(out, BaseException):
            raise out
        return _Resp(out)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(github.time, "sleep", lambda s: None)
    monkeypatch.setattr(github, "token", lambda review=True: "t")
    return calls


class TestADroppedConnectionIsRetriedOnce:
    def test_a_single_drop_is_recovered(self, monkeypatch):
        calls = _urlopen_sequence(monkeypatch, [
            http.client.RemoteDisconnected("closed"), b'{"ok":1}'])
        assert github.request("/x") == '{"ok":1}'
        assert len(calls) == 2

    def test_two_drops_raise_a_named_error_not_a_crash(self, monkeypatch):
        _urlopen_sequence(monkeypatch, [
            http.client.IncompleteRead(b"x"), ConnectionResetError(54, "reset")])
        with pytest.raises(github.ReviewError) as e:
            github.request("/x")
        assert "dropped on GET /x twice" in str(e.value)

    def test_a_dropped_WRITE_is_never_re_sent(self, monkeypatch):
        """The review's 🟡: an IncompleteRead on `POST …/reviews` is a review
        that may already be posted. Re-sending it posts it twice."""
        calls = _urlopen_sequence(monkeypatch, [
            http.client.IncompleteRead(b"x"), b'{"posted":"again"}'])
        with pytest.raises(github.ReviewError) as e:
            github.request("/repos/o/r/pulls/1/reviews", method="POST",
                           body={"event": "APPROVE"})
        assert len(calls) == 1
        assert "not retried: a write" in str(e.value)

    def test_an_http_error_is_not_retried(self, monkeypatch):
        """A 404 is an answer. Asking again spends a call to get the same one."""
        err = urllib.error.HTTPError("u", 404, "nf", {}, io.BytesIO(b""))
        calls = _urlopen_sequence(monkeypatch, [err, b"never"])
        with pytest.raises(urllib.error.HTTPError):
            github.request("/x")
        assert len(calls) == 1

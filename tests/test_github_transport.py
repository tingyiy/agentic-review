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


class TestAnHttpErrorNamesItsRequest:
    """browser-extension#386, 2026-09-11: a run crashed with "HTTPError: HTTP
    Error 422: Unprocessable Entity" and the log could not say which of a dozen
    calls had failed — the method, the path and GitHub's own explanation were
    all discarded. The dropped-connection branch had named its request since
    the day it was written.
    """

    def _raise(self, monkeypatch, code=422, body=b'{"message":"Validation Failed"}'):
        def urlopen(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, code, "Unprocessable Entity",
                                         {}, io.BytesIO(body))

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        req = urllib.request.Request("https://api.github.com/repos/o/r/statuses/abc",
                                     data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as caught:
            github._send(req)
        return caught.value

    def test_the_message_names_method_path_and_the_bodys_reason(self, monkeypatch):
        e = self._raise(monkeypatch)
        assert "POST" in str(e)
        assert "/repos/o/r/statuses/abc" in str(e)
        assert "Validation Failed" in str(e)
        assert "422" in str(e)

    def test_the_type_and_code_are_unchanged(self, monkeypatch):
        """Callers switch on both: the review POST reads a 422 to tell a
        refused verdict from any other rejection, and the redirect follower
        switches on `.code`."""
        e = self._raise(monkeypatch, code=404, body=b"nope")
        assert isinstance(e, urllib.error.HTTPError)
        assert e.code == 404

    def test_the_body_can_still_be_read_by_the_caller(self, monkeypatch):
        """`addbase.__init__` binds `read` to the original stream, so putting
        the body back means rebinding the method too. Setting `.fp` alone
        leaves every later reader an empty string — which would make every 422
        look like an unknown one to the verdict-refused handler."""
        e = self._raise(monkeypatch, body=b'{"message":"Review cannot be requested"}')
        assert "Review cannot be requested" in e.read().decode()

    def test_an_empty_body_still_names_the_request(self, monkeypatch):
        e = self._raise(monkeypatch, code=500, body=b"")
        assert "POST /repos/o/r/statuses/abc" in str(e)
        assert e.read() == b""

    def test_a_write_is_not_retried_by_this_path(self, monkeypatch):
        calls = []

        def urlopen(req, timeout=None):
            calls.append(1)
            raise urllib.error.HTTPError(req.full_url, 422, "x", {}, io.BytesIO(b""))

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        req = urllib.request.Request("https://api.github.com/x", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError):
            github._send(req)
        assert len(calls) == 1, "a 4xx is an answer, not a flake"

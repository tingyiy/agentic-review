"""What `_post` calls transient.

Every clause here is a failure that was once announced as a crash. A crash
means "the reviewer has a bug"; these mean "try the other provider".
"""
import http.client
import io
import json
import urllib.error
import urllib.request

import pytest

from agentic_review import llm


def _raise(exc):
    def urlopen(req, timeout=None):
        raise exc
    return urlopen


class TestAConnectionDroppedMidBodyIsTransient:
    """agentic-review#2, 2026-09-03 06:01: a router restart cut a reply at
    341 bytes. `URLError` only wraps connect-time failures, so the read-time
    `IncompleteRead` escaped as "pr-review crashed" and the run posted nothing."""

    @pytest.mark.parametrize("exc", [
        http.client.IncompleteRead(b"x" * 341),
        ConnectionResetError(54, "Connection reset by peer"),
        # 06:12 the same morning, at main: "Remote end closed connection
        # without response". A ConnectionResetError subclass, named so the
        # sibling is pinned rather than inferred.
        http.client.RemoteDisconnected("Remote end closed connection without response"),
        BrokenPipeError(32, "Broken pipe"),
    ])
    def test_it_is_retryable_not_a_crash(self, monkeypatch, exc):
        monkeypatch.setattr(urllib.request, "urlopen", _raise(exc))
        with pytest.raises(llm._Retryable) as e:
            llm._post("https://x/v1", "k", {"messages": []}, 5, "fireworks", "m")
        assert "connection dropped" in str(e.value)
        assert type(exc).__name__ in str(e.value)

    def test_a_connect_time_failure_still_is(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen",
                            _raise(urllib.error.URLError("no route")))
        with pytest.raises(llm._Retryable):
            llm._post("https://x/v1", "k", {"messages": []}, 5, "fireworks", "m")


_GONE = (b'{"error":{"message":"Model not found, inaccessible, and/or not '
         b'deployed","param":"model","code":"NOT_FOUND","type":"error"}}')


def _http(code, body):
    return urllib.error.HTTPError("https://x/v1", code, "err", {}, io.BytesIO(body))


class TestAModelTheProviderSaysItLacksFailsOver:
    """caeli-marketing#746, 2026-11-13 19:36 UTC: Fireworks answered turn 1
    with 404 "Model not found ... not deployed" for a model its `/models`
    listed and served minutes later. Read as our payload, it ended the review
    on turn 1 with OpenRouter healthy."""

    def test_the_404_is_retryable(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", _raise(_http(404, _GONE)))
        with pytest.raises(llm._Retryable) as e:
            llm._post("https://x/v1", "k", {"messages": []}, 5, "fireworks", "m")
        assert "404" in str(e.value)

    @pytest.mark.parametrize("code,body", [
        (404, b'{"error":{"message":"route not found"}}'),
        (400, _GONE),
    ])
    def test_any_other_4xx_is_still_ours(self, monkeypatch, code, body):
        monkeypatch.setattr(urllib.request, "urlopen", _raise(_http(code, body)))
        with pytest.raises(llm.ReviewError) as e:
            llm._post("https://x/v1", "k", {"messages": []}, 5, "fireworks", "m")
        assert not isinstance(e.value, llm._Retryable)

    def _providers(self, monkeypatch):
        seen = []

        def urlopen(req, timeout=None):
            if "fireworks" in req.full_url:
                seen.append("fireworks")
                raise _http(404, _GONE)
            seen.append("openrouter")
            return _Json({"choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": "from openrouter"}}]})

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        monkeypatch.setenv("FIREWORKS_API_KEY", "f")
        monkeypatch.setenv("OPENROUTER_API_KEY", "o")
        llm.reset_usage()
        return seen

    def test_a_mapped_model_is_answered_by_openrouter(self, monkeypatch):
        seen = self._providers(monkeypatch)
        out = llm.chat([{"role": "user", "content": "x"}])
        assert out == "from openrouter"
        assert seen == ["fireworks", "openrouter"]

    def test_a_mistyped_id_still_fails_without_spending_openrouter(self, monkeypatch):
        """The same body is what a typo gets. Nobody mapped a typo, so it must
        raise on the spot rather than discover the same thing twice."""
        seen = self._providers(monkeypatch)
        with pytest.raises(llm.ReviewError) as e:
            llm.chat([{"role": "user", "content": "x"}],
                     model="accounts/fireworks/models/deepseek-v4p1-flsh")
        assert "no failover mapping" in str(e.value)
        assert seen == ["fireworks"]


class _Json:
    """A non-streaming JSON response, which `_post` accepts unchanged."""

    def __init__(self, body):
        self._raw = json.dumps(body).encode()
        self.headers = {"Content-Type": "application/json"}

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

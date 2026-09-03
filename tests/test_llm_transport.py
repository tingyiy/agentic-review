"""What `_post` calls transient.

Every clause here is a failure that was once announced as a crash. A crash
means "the reviewer has a bug"; these mean "try the other provider".
"""
import http.client
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

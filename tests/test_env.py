"""Credentials, and the one thing about them that fails silently.

A missing key is loud. A key looked for in the wrong PLACE is not: the reviewer
raises "FIREWORKS_API_KEY not set" while the file holding it sits two lines
away in the config, and the message sends whoever reads it to the wrong problem.
"""
import importlib

import pytest

from agentic_review import env


def _reload(monkeypatch, value):
    monkeypatch.setenv("REVIEW_ENV_FILES", value)
    return importlib.reload(env)


def test_the_environment_wins(monkeypatch, tmp_path):
    f = tmp_path / "a.env"
    f.write_text("K=from-file\n")
    e = _reload(monkeypatch, str(f))
    monkeypatch.setenv("K", "from-env")
    assert e.get("K") == "from-env"


def test_a_file_value_is_found_and_EXPORTED(monkeypatch, tmp_path):
    """Exported, so a subprocess sees it too — git needs the token."""
    f = tmp_path / "a.env"
    f.write_text("K=secret\n")
    e = _reload(monkeypatch, str(f))
    monkeypatch.delenv("K", raising=False)
    import os
    assert e.get("K") == "secret"
    assert os.environ["K"] == "secret"


def test_tilde_is_expanded(monkeypatch, tmp_path):
    """The workflow passes `~/.config/agentic-review/.env`, because a GitHub Actions `env:` value
    is NOT shell-expanded and `$HOME` would arrive literally. If this stops
    working, every review fails with "token not set" on a box that has it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "h.env").write_text("K=tilde\n")
    e = _reload(monkeypatch, "~/h.env")
    monkeypatch.delenv("K", raising=False)
    assert e.get("K") == "tilde"


def test_files_are_tried_in_order(monkeypatch, tmp_path):
    (tmp_path / "1.env").write_text("K=first\n")
    (tmp_path / "2.env").write_text("K=second\n")
    e = _reload(monkeypatch, f"{tmp_path}/1.env:{tmp_path}/2.env")
    monkeypatch.delenv("K", raising=False)
    assert e.get("K") == "first"


def test_a_missing_file_is_skipped_not_fatal(monkeypatch, tmp_path):
    (tmp_path / "2.env").write_text("K=second\n")
    e = _reload(monkeypatch, f"{tmp_path}/nope.env:{tmp_path}/2.env")
    monkeypatch.delenv("K", raising=False)
    assert e.get("K") == "second"


def test_quotes_and_trailing_comments_are_stripped(monkeypatch, tmp_path):
    (tmp_path / "a.env").write_text('K="quoted"  # a note\n')
    e = _reload(monkeypatch, f"{tmp_path}/a.env")
    monkeypatch.delenv("K", raising=False)
    assert e.get("K") == "quoted"


def test_a_prefix_match_is_not_a_match(monkeypatch, tmp_path):
    """`KEY=` must not be found by a lookup for `K`."""
    (tmp_path / "a.env").write_text("KEY_LONGER=no\n")
    e = _reload(monkeypatch, f"{tmp_path}/a.env")
    monkeypatch.delenv("KEY", raising=False)
    assert e.get("KEY") is None


def test_nothing_configured_returns_none(monkeypatch):
    e = _reload(monkeypatch, "")
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    assert e.get("NOT_SET_ANYWHERE") is None


class TestAnEdgeBlockIsNotOurPayload:
    """slack-app#378, turn 10: `403: error code: 1010` — a Cloudflare signature
    ban from the CDN in front of the API, after NINE turns of the same
    credentials and the same payload shape had succeeded.

    The failover rule is "a 4xx is our payload and would fail identically on the
    second provider". That is right in general and wrong here: the other
    provider sits behind a different edge, and treating this as a payload error
    threw away a review that was nine turns in.
    """

    def test_a_cloudflare_1010_is_retryable(self):
        from agentic_review import llm
        assert llm._is_edge_block(403, "error code: 1010")

    def test_a_1020_access_denial_too(self):
        from agentic_review import llm
        assert llm._is_edge_block(403, "error code: 1020")

    def test_a_named_cloudflare_page_counts(self):
        from agentic_review import llm
        assert llm._is_edge_block(403, "<title>Attention Required! | Cloudflare</title>")

    def test_a_REAL_403_from_the_api_is_not(self):
        """A revoked key or a model that is not enabled must still fail loudly,
        rather than spend a second provider's budget discovering the same."""
        from agentic_review import llm
        assert not llm._is_edge_block(
            403, '{"error":{"message":"model not enabled for this account"}}')

    def test_a_400_is_never_an_edge_block(self):
        """Our payload really is our payload."""
        from agentic_review import llm
        assert not llm._is_edge_block(400, "error code: 1010")

    def test_a_404_is_not_either(self):
        from agentic_review import llm
        assert not llm._is_edge_block(404, "cloudflare")



class TestFailoverIsSticky:
    """2026-09-02 22:00-22:20 UTC, two live reviews: Fireworks' /models answered
    in 0.27s while every chat completion hit the 180s timeout, and failover was
    per call — so every turn paid 180s on Fireworks before OpenRouter answered
    in ~40s. Turns cost ~3.5 minutes and both runs were heading for the
    25-minute ceiling. A provider that has just stalled is the one least likely
    to answer the next call in time."""

    def _arm(self, monkeypatch, fireworks_ok):
        from agentic_review import llm
        calls = []

        def fake_post(url, key, payload, timeout, provider, model):
            calls.append(provider)
            if provider == "fireworks" and not fireworks_ok:
                raise llm._Retryable("fireworks timed out")
            return {"finish_reason": "stop",
                    "message": {"role": "assistant", "content": provider}}

        monkeypatch.setattr(llm, "_post", fake_post)
        monkeypatch.setenv("FIREWORKS_API_KEY", "f")
        monkeypatch.setenv("OPENROUTER_API_KEY", "o")
        llm.reset_usage()
        return llm, calls

    def test_after_one_fireworks_failure_openrouter_goes_first(self, monkeypatch):
        llm, calls = self._arm(monkeypatch, fireworks_ok=False)
        m = [{"role": "user", "content": "x"}]
        assert llm.chat(m) == "openrouter"
        assert llm.chat(m) == "openrouter"
        assert calls == ["fireworks", "openrouter", "openrouter"], calls

    def test_a_fresh_review_starts_on_fireworks_again(self, monkeypatch):
        llm, calls = self._arm(monkeypatch, fireworks_ok=False)
        llm.chat([{"role": "user", "content": "x"}])
        llm.reset_usage()                      # main() calls this per review
        monkeypatch.setattr(llm, "_post", lambda *a: (calls.append(a[4]) or
                            {"finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}))
        llm.chat([{"role": "user", "content": "x"}])
        assert calls[-1] == "fireworks"

    def test_a_healthy_fireworks_is_never_bypassed(self, monkeypatch):
        llm, calls = self._arm(monkeypatch, fireworks_ok=True)
        m = [{"role": "user", "content": "x"}]
        llm.chat(m); llm.chat(m)
        assert calls == ["fireworks", "fireworks"]

    def test_sticky_openrouter_still_falls_back_to_fireworks(self, monkeypatch):
        """Preferring the failover must not make it the only provider."""
        from agentic_review import llm
        seq = []

        def fake_post(url, key, payload, timeout, provider, model):
            seq.append(provider)
            if len(seq) <= 1 or provider == "openrouter":
                raise llm._Retryable(f"{provider} down")
            return {"finish_reason": "stop",
                    "message": {"role": "assistant", "content": provider}}

        monkeypatch.setattr(llm, "_post", fake_post)
        monkeypatch.setenv("FIREWORKS_API_KEY", "f")
        monkeypatch.setenv("OPENROUTER_API_KEY", "o")
        llm.reset_usage()
        m = [{"role": "user", "content": "x"}]
        # call 1: fireworks fails -> openrouter fails -> error, but sticky is set
        try:
            llm.chat(m)
        except llm.ReviewError:
            pass
        # call 2: openrouter first (fails), then fireworks (now healthy)
        assert llm.chat(m) == "fireworks"

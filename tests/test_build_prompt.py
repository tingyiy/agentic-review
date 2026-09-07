"""SCRUM-1314: one prompt path, because the harness kept drifting from `main`.

Three drifts, two caught by review and one not caught for weeks:

  · `diff` omitted, so the cross-reference section measured itself as absent;
  · the pass and the revision called directly and nothing in between, so an
    18-run "gap arm" was a second baseline sample (2026-09-07);
  · hunks never expanded at all, so every eval number taken before
    2026-09-07 measured a reviewer prompted with bare hunks against a
    production one prompted with the code around them.

Each of these was invisible in its own run and rewrote the result. The tests
here drive both callers and compare what reaches the model.
"""
import json

import pytest

from agentic_review import review


DIFF = "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-x\n+y\n"


@pytest.fixture
def marked(monkeypatch):
    """`expand_hunks` marks what it touched, so a prompt built without it is
    distinguishable from one built with it."""
    monkeypatch.setattr(review.ctx, "expand_hunks",
                        lambda d, w, **k: "[EXPANDED]\n" + d)
    return "[EXPANDED]"


class TestBuildPrompt:
    def test_it_returns_the_expanded_text_it_used(self, marked):
        prompt, shown = review.build_prompt("r", ".", DIFF)
        assert shown.startswith(marked)
        assert shown in prompt

    def test_the_caller_s_caveats_context_and_prior_reach_the_prompt(self, marked):
        prompt, _ = review.build_prompt("r", ".", DIFF, caveats="CAVEAT",
                                        context="CONTEXT", prior="PRIOR")
        for piece in ("CAVEAT", "CONTEXT", "PRIOR"):
            assert piece in prompt

    def test_it_caps_what_reaches_the_model(self, monkeypatch):
        """`expand_hunks` returns the ORIGINAL diff when expanding would
        breach the cap, so `max_chars` bounds the expansion, not the prompt."""
        monkeypatch.setattr(review.ctx, "expand_hunks",
                            lambda d, w, **k: "x" * (review.MAX_DIFF * 4))
        _, shown = review.build_prompt("r", ".", DIFF)
        # The cap keeps a whole-line prefix and appends a footer saying where
        # it stopped, so the result is the limit plus that note — not the
        # four-times-over text it was handed.
        assert len(shown) <= int(review.MAX_DIFF * 1.6) + 500
        assert "truncated" in shown


class TestBothCallersUseIt:
    """Driven, not read: this repository does not assert on source text."""

    def _main(self, monkeypatch, marked):
        seen = {}
        monkeypatch.setattr(review, "gh",
                            lambda path, method="GET", body=None, accept="":
                            json.dumps({"draft": False, "state": "open",
                                        "merged": False, "title": "SCRUM-1 x",
                                        "user": {"login": "someone"},
                                        "head": {"sha": "a" * 40}})
                            if path.endswith("/pulls/7") else json.dumps([]))
        diff = review._Diff(DIFF)
        diff.overflow = []
        diff.full = DIFF
        monkeypatch.setattr(review, "pr_diff", lambda *a: (diff, [], 0))
        monkeypatch.setattr(review, "_already_reviewed", lambda *a, **k: "")
        monkeypatch.setattr(review, "conversation", lambda *a: "")
        monkeypatch.setattr(review, "changed_since_last_review", lambda *a, **k: "")
        monkeypatch.setattr(review, "build_context", lambda *a: "")
        monkeypatch.setattr(review, "commit_messages", lambda *a: [])
        monkeypatch.setattr(review, "checkout", lambda *a: None)
        monkeypatch.setattr(review.ctx, "skeletons", lambda w, paths: "")
        monkeypatch.setattr(review, "review_findings",
                            lambda prompt, work, repo="": seen.setdefault(
                                "prompt", prompt) and [])
        monkeypatch.setattr(review, "_look_again",
                            lambda f, w, r, shown: seen.setdefault("shown", shown) and [])
        monkeypatch.setattr(review, "_revise", lambda f, w, r: (f, []))
        monkeypatch.setattr(review.checks, "run_all", lambda *a, **k: [])
        monkeypatch.setattr(review, "post_review", lambda *a, **k: "COMMENT")
        monkeypatch.setattr(review, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(review.sys, "argv", ["pr-review", "app", "7"])
        monkeypatch.delenv("DRY", raising=False)
        review.main()
        return seen

    def _harness(self, monkeypatch, marked):
        from eval import compare

        seen = {}
        monkeypatch.setattr(compare.review, "gh", lambda *a, **k: json.dumps(
            {"head": {"sha": "a" * 40}, "title": "t", "body": ""}))
        monkeypatch.setattr(compare, "_diff_at", lambda *a: (DIFF, [], 0))
        monkeypatch.setattr(compare.review, "checkout", lambda *a: None)
        monkeypatch.setattr(compare.review, "build_context", lambda *a: "")
        monkeypatch.setattr(compare.review, "commit_messages", lambda *a: [])
        monkeypatch.setattr(compare.review, "review_findings",
                            lambda prompt, work, repo="": seen.setdefault(
                                "prompt", prompt) and [])
        monkeypatch.setattr(compare.review, "_look_again",
                            lambda f, w, r, shown: seen.setdefault("shown", shown) and [])
        monkeypatch.setattr(compare.review, "_revise", lambda f, w, r: (f, []))
        monkeypatch.setattr(compare.checks, "run_all", lambda *a, **k: [])
        compare.run_ours("app", 7)
        return seen

    def test_main_prompts_with_the_expanded_diff(self, monkeypatch, marked):
        assert marked in self._main(monkeypatch, marked)["prompt"]

    def test_the_harness_prompts_with_the_expanded_diff(self, monkeypatch, marked):
        """The drift that went uncaught: bare hunks in the harness, the code
        around them in production, and every measured number in between."""
        assert marked in self._harness(monkeypatch, marked)["prompt"]

    def test_both_hand_the_second_look_what_the_model_was_shown(self, monkeypatch, marked):
        assert marked in self._main(monkeypatch, marked)["shown"]
        assert marked in self._harness(monkeypatch, marked)["shown"]

    def test_the_two_prompts_agree_given_the_same_inputs(self, monkeypatch, marked):
        """Not byte-identical — `main` adds its own caveats and prior — but
        the diff each was built from must be the same text."""
        assert (self._main(monkeypatch, marked)["shown"]
                == self._harness(monkeypatch, marked)["shown"])

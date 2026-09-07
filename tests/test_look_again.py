"""SCRUM-1241: the second look is aimed, not a generic "what else do you find?".

The generic form was measured on 2026-09-02 (rounds 7 and 8, three PRs, n=3):
28 findings became 39. It cannot reach the misses that matter, because a
defect in a file the agent never opened is not recoverable by asking the same
conversation again. `read_ranges` already records which lines of which file
were shown, so the gap is arithmetic rather than a guess.
"""
import json

import pytest

from agentic_review import review


DIFF = ("--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        "--- a/src/b.py\n+++ b/src/b.py\n@@ -1 +1 @@\n-x\n+y\n"
        "--- a/src/c.py\n+++ b/src/c.py\n@@ -1 +1 @@\n-x\n+y\n")


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.setitem(review._CURRENT, "opened", set())
    monkeypatch.setitem(review._CURRENT, "read_ranges", {})
    monkeypatch.setitem(review._CURRENT, "stats", {"messages": [{"role": "user"}]})
    monkeypatch.setitem(review._CURRENT, "answer_shortened", False)
    monkeypatch.setenv("REVIEW_GAP_PASS", "1")
    return review


def _finding(path, title="t"):
    return {"file": path, "line": 1, "severity": "medium", "title": title,
            "detail": "d"}


def _armed(monkeypatch, reply, tool_calls=3, stats_extra=None):
    seen = {}

    def resume(messages, question, root, **kw):
        seen["question"] = question
        seen["kw"] = kw
        kw["stats"].update(tool_calls=tool_calls, turns=1, **(stats_extra or {}))
        if isinstance(reply, Exception):
            raise reply
        return reply, []

    monkeypatch.setattr(review.agent, "resume", resume)
    return seen


class TestTheGapIsArithmetic:
    def test_a_fully_read_file_is_not_in_it(self, clean):
        clean._CURRENT["opened"] = {"src/a.py"}
        assert [p for p, _ in clean._unread_changed(DIFF)] == ["src/b.py", "src/c.py"]

    def test_never_opened_comes_before_partly_read(self, clean):
        clean._CURRENT["read_ranges"] = {
            "src/b.py": {"total": 100, "covered": [(1, 40)]},
            "src/c.py": {"total": 100, "covered": [(1, 90)]}}
        assert clean._unread_changed(DIFF) == [
            ("src/a.py", None), ("src/b.py", 60), ("src/c.py", 10)]

    def test_overlapping_windows_are_not_counted_twice(self, clean):
        clean._CURRENT["read_ranges"] = {
            "src/a.py": {"total": 100, "covered": [(1, 50), (40, 60)]}}
        assert dict(clean._unread_changed(DIFF))["src/a.py"] == 40

    def test_nothing_unread_means_no_call(self, clean, monkeypatch):
        clean._CURRENT["opened"] = {"src/a.py", "src/b.py", "src/c.py"}
        seen = _armed(monkeypatch, "unused")
        assert clean._look_again([], ".", "r", DIFF) == []
        assert not seen, "it asked anyway"


class TestItIsOffByDefault:
    def test_no_flag_no_call(self, clean, monkeypatch):
        monkeypatch.delenv("REVIEW_GAP_PASS", raising=False)
        seen = _armed(monkeypatch, "unused")
        assert clean._look_again([], ".", "r", DIFF) == []
        assert not seen


class TestWhatItAsksFor:
    def test_it_names_the_unread_paths_and_the_answer_shape(self, clean, monkeypatch):
        seen = _armed(monkeypatch, json.dumps({"findings": []}))
        clean._look_again([], ".", "r", DIFF)
        q = seen["question"]
        for path in ("src/a.py", "src/b.py", "src/c.py"):
            assert path in q
        assert "never opened" in q
        assert "NOT already in your answer" in q, "a plain 'what else' plateaus"
        assert '"findings"' in q
        assert seen["kw"]["answer_schema"] is clean.ANSWER_SCHEMA

    def test_a_long_list_is_capped_and_says_so(self, clean, monkeypatch):
        big = "".join(f"--- a/f{i}.py\n+++ b/f{i}.py\n@@ -1 +1 @@\n-x\n+y\n"
                      for i in range(30))
        seen = _armed(monkeypatch, json.dumps({"findings": []}))
        clean._look_again([], ".", "r", big)
        assert "…and 18 more" in seen["question"]
        assert seen["question"].count("  · ") == clean.MAX_GAP_PATHS + 1


class TestWhatItAcceptsBack:
    def test_a_finding_about_an_unread_file_is_kept(self, clean, monkeypatch):
        _armed(monkeypatch, json.dumps({"findings": [_finding("src/b.py")]}))
        got = clean._look_again([], ".", "r", DIFF)
        assert [f["file"] for f in got] == ["src/b.py"]

    def test_a_finding_about_a_file_it_had_already_read_is_dropped(self, clean, monkeypatch):
        """Otherwise this is a second 'what else', which is the thing measured
        to plateau: asked for more, the model restates."""
        clean._CURRENT["opened"] = {"src/a.py"}
        _armed(monkeypatch, json.dumps({"findings": [
            _finding("src/a.py"), _finding("src/b.py")]}))
        got = clean._look_again([], ".", "r", DIFF)
        assert [f["file"] for f in got] == ["src/b.py"]

    def test_an_unusable_reply_adds_nothing(self, clean, monkeypatch):
        _armed(monkeypatch, "I looked and everything seems fine.")
        assert clean._look_again([], ".", "r", DIFF) == []

    def test_a_provider_failure_adds_nothing(self, clean, monkeypatch):
        _armed(monkeypatch, review.ReviewError("fireworks -> 500"))
        assert clean._look_again([], ".", "r", DIFF) == []

    def test_a_closed_pr_still_aborts(self, clean, monkeypatch):
        _armed(monkeypatch, review.PRClosed("merged"))
        with pytest.raises(review.PRClosed):
            clean._look_again([], ".", "r", DIFF)

    def test_what_it_read_is_merged_even_when_the_reply_is_unusable(self, clean, monkeypatch):
        _armed(monkeypatch, "prose", stats_extra={"opened": {"src/b.py"}})
        clean._look_again([], ".", "r", DIFF)
        assert "src/b.py" in clean._CURRENT["opened"], (
            "a file it opened must not be reported as never opened")

    def test_a_shortened_answer_marks_the_review(self, clean, monkeypatch):
        _armed(monkeypatch, json.dumps({"findings": []}),
               stats_extra={"shortened": True})
        clean._look_again([], ".", "r", DIFF)
        assert clean._CURRENT["answer_shortened"] is True


class TestMainActuallyCallsIt:
    """The guard this class exists for: removing the call site in `main` left
    every test above green. Correct code behind a call nobody makes is the
    failure shape this repository produces most often."""

    def _run(self, monkeypatch, gap_findings):
        seen = {"diffs": [], "revised": []}

        def look_again(findings, work, repo, shown):
            seen["diffs"].append(shown)
            return list(gap_findings)

        monkeypatch.setattr(review, "_look_again", look_again)
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
        monkeypatch.setattr(review.ctx, "expand_hunks", lambda d, w, **k: d)
        monkeypatch.setattr(review.ctx, "skeletons", lambda w, paths: "")
        monkeypatch.setattr(review, "review_findings",
                            lambda *a, **k: [_finding("src/a.py", "from the pass")])
        monkeypatch.setattr(review, "_revise",
                            lambda f, w, r: (seen["revised"].append(list(f)) or (f, [])))
        monkeypatch.setattr(review.checks, "run_all", lambda *a, **k: [])
        monkeypatch.setattr(review, "post_review",
                            lambda *a, **k: seen.setdefault("posted", a[3]) and "COMMENT")
        monkeypatch.setattr(review, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(review.sys, "argv", ["pr-review", "app", "7"])
        monkeypatch.delenv("DRY", raising=False)
        review.main()
        return seen

    def test_it_runs_on_the_diff_the_pass_was_shown(self, monkeypatch):
        seen = self._run(monkeypatch, [])
        assert seen["diffs"], "main never called the second look"
        assert "src/a.py" in seen["diffs"][0]

    def test_its_findings_are_posted_and_revised_with_the_rest(self, monkeypatch):
        seen = self._run(monkeypatch, [_finding("src/b.py", "from the gap")])
        assert [f["title"] for f in seen["revised"][0]] == [
            "from the pass", "from the gap"], "the revision never saw it"
        assert "from the gap" in seen["posted"]


class TestTheHarnessRunsWhatMainRuns:
    """A feature reachable only from `main` measures itself as absent. On
    2026-09-07 an 18-run "gap arm" was in fact a second baseline sample,
    because the eval harness called the pass and the revision directly and
    nothing in between. Driven, not read: this repository does not assert on
    source text."""

    def test_the_second_look_runs_between_the_pass_and_the_revision(self, monkeypatch, tmp_path):
        from eval import compare

        order = []

        def pass_(prompt, work, repo=""):
            order.append("pass")
            return [_finding("src/a.py", "from the pass")]

        def look(findings, work, repo, shown):
            order.append("look")
            assert "src/b.py" in shown, "the second look was not given the diff"
            return [_finding("src/b.py", "from the gap")]

        def revise(findings, work, repo):
            order.append("revise")
            return list(findings), []

        monkeypatch.setattr(compare.review, "review_findings", pass_)
        monkeypatch.setattr(compare.review, "_look_again", look)
        monkeypatch.setattr(compare.review, "_revise", revise)
        monkeypatch.setattr(compare.review, "gh", lambda *a, **k: json.dumps(
            {"head": {"sha": "a" * 40}, "title": "t", "body": ""}))
        monkeypatch.setattr(compare, "_diff_at", lambda *a: (DIFF, [], 0))
        monkeypatch.setattr(compare.review, "checkout", lambda *a: None)
        monkeypatch.setattr(compare.review, "build_context", lambda *a: "")
        monkeypatch.setattr(compare.review, "commit_messages", lambda *a: [])
        monkeypatch.setattr(compare.checks, "run_all", lambda *a, **k: [])

        result = compare.run_ours("app", 7)
        assert order == ["pass", "look", "revise"], order
        assert [f["title"] for f in result["findings"]] == [
            "from the pass", "from the gap"]


class TestTheRangeArithmeticSurvivesRealRecords:
    """`_record_opened` writes `(offset, None)` for a window with no footer —
    the read ran to the end of the file. `hi + 1` on that raised a TypeError
    that killed 6 of 18 eval runs on 2026-09-07, from outside the guard, so in
    production it would have been reported as a broken reviewer."""

    def test_an_open_ended_window_is_clamped_to_the_total(self, clean):
        clean._CURRENT["read_ranges"] = {
            "src/a.py": {"total": 100, "covered": [(60, None)]}}
        assert dict(clean._unread_changed(DIFF))["src/a.py"] == 59

    def test_a_window_past_the_end_does_not_go_negative(self, clean):
        clean._CURRENT["read_ranges"] = {
            "src/a.py": {"total": 10, "covered": [(1, 40)]}}
        assert dict(clean._unread_changed(DIFF))["src/a.py"] == 0

    def test_a_broken_record_costs_no_findings(self, clean, monkeypatch):
        def boom(_diff):
            raise TypeError("unsupported operand type(s)")

        monkeypatch.setattr(review, "_unread_changed", boom)
        seen = _armed(monkeypatch, json.dumps({"findings": []}))
        assert clean._look_again([_finding("src/a.py")], ".", "r", DIFF) == []
        assert not seen, "it asked the model with a broken list"


class TestItLeavesTheRevisionItsBudget:
    """Both passes resume from the same pool and the optional one goes first.
    An extra look that leaves the mandatory revision on its 60s floor — forced
    on turn one with no tool calls — has made the review worse."""

    def test_it_asks_for_less_than_the_revision_reserves(self, clean, monkeypatch):
        monkeypatch.setattr(review, "_remaining_budget", lambda: 1000)
        seen = _armed(monkeypatch, json.dumps({"findings": []}))
        clean._look_again([], ".", "r", DIFF)
        assert seen["kw"]["deadline"] == review.GAP_DEADLINE
        assert review.GAP_DEADLINE < review.REVISE_DEADLINE

    def test_a_tight_budget_shrinks_the_look_not_the_revision(self, clean, monkeypatch):
        monkeypatch.setattr(review, "_remaining_budget",
                            lambda: review.REVISE_DEADLINE + 90)
        seen = _armed(monkeypatch, json.dumps({"findings": []}))
        clean._look_again([], ".", "r", DIFF)
        assert seen["kw"]["deadline"] == 90

    def test_no_room_for_both_means_no_second_look(self, clean, monkeypatch):
        monkeypatch.setattr(review, "_remaining_budget",
                            lambda: review.REVISE_DEADLINE + 30)
        seen = _armed(monkeypatch, json.dumps({"findings": []}))
        assert clean._look_again([], ".", "r", DIFF) == []
        assert not seen, "it spent the revision's budget"


class TestOneSpellingOfAPath:
    """`read_ranges` and `opened` are normalised; a diff header is raw. The
    normalisation exists because `./src/big.py` once never matched
    `src/big.py` — the same mismatch here reports a read file as unread and
    then drops the finding it asked for."""

    def test_a_dot_slash_diff_path_matches_a_normalised_record(self, clean):
        diff = "--- a/./src/a.py\n+++ b/./src/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        clean._CURRENT["opened"] = {"src/a.py"}
        assert clean._unread_changed(diff) == []

    def test_a_finding_is_kept_however_the_model_spells_the_path(self, clean, monkeypatch):
        _armed(monkeypatch, json.dumps({"findings": [_finding("./src/b.py")]}))
        got = clean._look_again([], ".", "r", DIFF)
        assert [f["file"] for f in got] == ["./src/b.py"], (
            "a legitimate finding was dropped on spelling alone")

"""Reviewing the whole pull request in several passes.

caeli-marketing#240: 172,511 chars of diff against a 60,000 budget, and
`components/shop/product-pdp.tsx` alone was 69,366 — larger than the whole
budget, so it could never fit at any packing order. Five files went unreviewed,
both test files among them, while Copilot reported 17/17.
"""
import json

import pytest

from agentic_review import review as pr


def _blob(path, lines=3):
    body = "".join(f"+line {i} of {path}\n" for i in range(lines))
    return (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
            f"@@ -0,0 +1,{lines} @@\n{body}")


class _Harness:
    """`main` with everything but the pass loop stubbed out."""

    def run(self, monkeypatch, first, overflow, findings_per_pass=(),
            excluded=(), deadline=None):
        diff = pr._Diff(first)
        diff.overflow = list(overflow)
        diff.full = "\n".join([first] + list(overflow))
        seen = {"prompts": [], "revised": []}
        calls = iter(findings_per_pass or [[] for _ in range(1 + len(overflow))])

        def review_findings(prompt, work, repo=""):
            seen["prompts"].append(prompt)
            return list(next(calls, []))

        monkeypatch.setattr(pr, "gh", lambda path, method="GET", body=None, accept="":
                            json.dumps({"draft": False, "state": "open",
                                        "merged": False, "title": "SCRUM-1 x",
                                        "user": {"login": "someone"},
                                        "head": {"sha": "a" * 40}})
                            if path.endswith("/pulls/7") else json.dumps([]))
        monkeypatch.setattr(pr, "pr_diff", lambda *a: (diff, list(excluded), 0))
        monkeypatch.setattr(pr, "_already_reviewed",
                            lambda *a, **k: seen.setdefault("fingerprinted", a[3]) and "")
        monkeypatch.setattr(pr, "conversation", lambda *a: "")
        monkeypatch.setattr(pr, "changed_since_last_review", lambda *a, **k: "")
        monkeypatch.setattr(pr, "build_context", lambda *a: "")
        monkeypatch.setattr(pr, "commit_messages", lambda *a: [])
        monkeypatch.setattr(pr, "checkout", lambda *a: None)
        monkeypatch.setattr(pr.ctx, "expand_hunks", lambda d, w, **k: d)
        monkeypatch.setattr(pr.ctx, "skeletons", lambda w, paths: "")
        monkeypatch.setattr(pr, "review_findings", review_findings)
        monkeypatch.setattr(pr, "_revise",
                            lambda f, w, r: (seen["revised"].append(list(f)) or (f, [])))
        monkeypatch.setattr(pr.checks, "run_all", lambda *a, **k: [])
        monkeypatch.setattr(pr, "post_review",
                            lambda *a, **k: seen.setdefault("posted", a[3]) and "COMMENT")
        monkeypatch.setattr(pr, "_pr_is_gone", lambda *a: None)
        if deadline is not None:
            monkeypatch.setattr(pr, "PASS_DEADLINE", deadline)
        monkeypatch.setattr(pr.sys, "argv", ["pr-review", "app", "7"])
        monkeypatch.delenv("DRY", raising=False)
        pr.main()
        return seen


class TestEveryPassIsReviewed:
    def test_each_pass_gets_its_own_call(self, monkeypatch):
        seen = _Harness().run(monkeypatch, _blob("a.py"),
                              [_blob("b.py"), _blob("c.py")])
        assert len(seen["prompts"]) == 3
        # The DIFF LINES, not the path: every prompt names the other passes'
        # paths on purpose, so a path alone proves nothing about which pass
        # actually carries the change.
        assert "line 1 of a.py" in seen["prompts"][0]
        assert "line 1 of b.py" not in seen["prompts"][0]
        assert "line 1 of b.py" in seen["prompts"][1]
        assert "line 1 of c.py" in seen["prompts"][2]

    def test_a_pr_that_fits_still_costs_one_call(self, monkeypatch):
        seen = _Harness().run(monkeypatch, _blob("a.py"), [])
        assert len(seen["prompts"]) == 1

    def test_findings_from_every_pass_survive(self, monkeypatch):
        seen = _Harness().run(
            monkeypatch, _blob("a.py"), [_blob("b.py")],
            findings_per_pass=[[{"file": "a.py", "line": 1, "title": "one",
                                 "severity": "medium", "detail": "d"}],
                               [{"file": "b.py", "line": 2, "title": "two",
                                 "severity": "medium", "detail": "d"}]])
        assert "one" in seen["posted"] and "two" in seen["posted"]

    def test_each_pass_is_revised_in_its_own_conversation(self, monkeypatch):
        """`_revise` RESUMES the conversation that read the code. Revising the
        first pass's findings inside the third pass's would ask a model about
        code it never saw."""
        seen = _Harness().run(
            monkeypatch, _blob("a.py"), [_blob("b.py")],
            findings_per_pass=[[{"file": "a.py", "line": 1, "title": "one",
                                 "severity": "low", "detail": "d"}],
                               [{"file": "b.py", "line": 2, "title": "two",
                                 "severity": "low", "detail": "d"}]])
        assert [[f["title"] for f in call] for call in seen["revised"]] \
            == [["one"], ["two"]]


class TestAPassKnowsItIsOneOfSeveral:
    def test_it_is_told_what_the_others_hold(self, monkeypatch):
        """A reviewer shown two of eighteen files, with no word that the rest
        exist, reads the two as the whole change."""
        seen = _Harness().run(monkeypatch, _blob("a.py"), [_blob("b.py")])
        assert "part 1" in seen["prompts"][0] and "`b.py`" in seen["prompts"][0]
        assert "part 2" in seen["prompts"][1] and "`a.py`" in seen["prompts"][1]
        assert "do not report them as missing" in seen["prompts"][0]

    def test_a_single_pass_is_told_nothing_of_the_kind(self, monkeypatch):
        seen = _Harness().run(monkeypatch, _blob("a.py"), [])
        assert "part 1" not in seen["prompts"][0]


class TestTheSameDefectIsPostedOnce:
    def test_two_passes_reaching_one_line_post_one_finding(self, monkeypatch):
        same = {"file": "a.py", "line": 7, "title": "The same defect",
                "severity": "medium", "detail": "d"}
        seen = _Harness().run(monkeypatch, _blob("a.py"), [_blob("b.py")],
                              findings_per_pass=[[same], [dict(same)]])
        assert seen["posted"].count("The same defect") == 1

    def test_different_lines_are_different_findings(self, monkeypatch):
        a = {"file": "a.py", "line": 7, "title": "defect", "severity": "medium",
             "detail": "d"}
        b = dict(a, line=9)
        seen = _Harness().run(monkeypatch, _blob("a.py"), [_blob("b.py")],
                              findings_per_pass=[[a], [b]])
        assert seen["posted"].count("defect") == 2

    def test_wording_that_differs_only_in_punctuation_is_one(self):
        assert pr._finding_key({"file": "a.py", "line": 1, "title": "Off-by-one!"}) \
            == pr._finding_key({"file": "A.py", "line": "1", "title": "off by one"})


class TestTheClockStopsIt:
    def test_a_pass_is_not_started_past_the_deadline(self, monkeypatch):
        """The job is `timeout-minutes: 25`, and a killed job posts NOTHING —
        worse than an honest partial review."""
        seen = _Harness().run(monkeypatch, _blob("a.py"),
                              [_blob("b.py"), _blob("c.py")], deadline=-1)
        assert len(seen["prompts"]) == 1

    def test_what_it_did_not_reach_is_named_to_the_reader(self, monkeypatch):
        seen = _Harness().run(monkeypatch, _blob("a.py"),
                              [_blob("b.py"), _blob("c.py")], deadline=-1)
        assert "b.py" in seen["posted"] and "c.py" in seen["posted"]
        assert "NOT opened" in seen["posted"] or "not opened" in seen["posted"]

    def test_a_generous_deadline_reaches_them_all(self, monkeypatch):
        seen = _Harness().run(monkeypatch, _blob("a.py"),
                              [_blob("b.py"), _blob("c.py")], deadline=10_000)
        assert len(seen["prompts"]) == 3


class TestTheFingerprintCoversTheWholePR:
    """`_diff_fp` hashes what it is given, and the mark it writes is what
    decides whether anything changed since the last review. Hashing only the
    first pass meant a push confined to a file past the budget left the
    fingerprint untouched and was skipped as "the base moved, this PR's own
    changes did not" — a commit reviewed by nothing."""

    def test_the_nothing_new_guard_sees_every_file(self, monkeypatch):
        seen = _Harness().run(monkeypatch, _blob("a.py"), [_blob("b.py")])
        assert "line 1 of b.py" in seen["fingerprinted"]

    def test_a_change_in_an_overflow_file_moves_the_fingerprint(self):
        one = _blob("a.py") + "\n" + _blob("b.py")
        two = _blob("a.py") + "\n" + _blob("b.py", lines=4)
        assert pr._diff_fp(one) != pr._diff_fp(two)


class TestTheCaveatGivesAdviceThatWorks:
    """caeli-marketing#240, 2026-09-05: five files were cut on every round and
    the caveat said "request a follow-up review naming them" — which cuts the
    same five files, because it is the same budget. The author asked for one,
    got the identical caveat, and the loop had no exit. Advice a reader cannot
    act on is worse than no advice."""

    def test_it_does_not_send_the_author_round_the_loop(self):
        note = pr._unreviewed_files_note(["a.py", "b.py"])
        assert "follow-up review naming them" not in note
        assert "cuts the same files again" in note

    def test_it_names_what_actually_clears_it(self):
        note = pr._unreviewed_files_note(["a.py"])
        assert "split the PR" in note and "REVIEW_MAX_PASSES" in note

    def test_nothing_excluded_says_nothing(self):
        assert pr._unreviewed_files_note([]) == ""

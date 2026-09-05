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


class TestTheMarkMatchesWhatTheGuardRecomputes:
    """The `<!-- caeli-review diff:… -->` mark in the body is what
    `_already_reviewed` recomputes and compares on the next run. It hashes the
    WHOLE PR — so a first-pass fingerprint in the body made the two disagree on
    every multi-pass PR, and the "the base moved, this PR's own changes did
    not" skip could never fire: every update-branch paid for a full multi-pass
    review. Found by this reviewer on the PR that added the passes."""

    def test_the_body_carries_the_whole_prs_fingerprint(self, monkeypatch):
        first, over = _blob("a.py"), _blob("b.py")
        seen = _Harness().run(monkeypatch, first, [over])
        whole = "\n".join([first, over])
        assert pr._DIFF_MARK.format(fp=pr._diff_fp(whole)) in seen["posted"]

    def test_and_it_is_not_the_first_pass_alone(self, monkeypatch):
        first, over = _blob("a.py"), _blob("b.py")
        seen = _Harness().run(monkeypatch, first, [over])
        assert pr._DIFF_MARK.format(fp=pr._diff_fp(first)) not in seen["posted"]


class TestItPromisesOnlyWhatTheClockCanKeep:
    def test_the_note_does_not_guarantee_a_later_pass(self, monkeypatch):
        """`PASS_DEADLINE` can cancel one, and the model was told not to report
        those files as missing — a promise the run may not keep."""
        seen = _Harness().run(monkeypatch, _blob("a.py"), [_blob("b.py")])
        assert "ARE being reviewed" not in seen["prompts"][0]
        assert "SCHEDULED for their own passes" in seen["prompts"][0]

    def test_it_still_says_they_are_not_missing(self, monkeypatch):
        seen = _Harness().run(monkeypatch, _blob("a.py"), [_blob("b.py")])
        assert "do not report them as missing" in seen["prompts"][0]


class TestAFileTooBigForAnyPass:
    """infra#180 added a 2 MB JSONL corpus. "The first file always fits" made
    it `pass 2 of 2: 2,084,684 chars`; the transcript hit its budget on turn
    one, the model answered in 32 characters having made no tool calls, and the
    evidence guard correctly refused the whole review. Nobody was ever going to
    read a truncated slab of that file."""

    def _diff(self, monkeypatch, blobs, cap, ceiling):
        monkeypatch.setattr(pr, "gh", lambda *a, **k: "".join(blobs))
        monkeypatch.setattr(pr, "MAX_DIFF", cap)
        monkeypatch.setattr(pr, "MAX_FILE_DIFF", ceiling)
        return pr.pr_diff("repo", 1)

    def test_it_gets_a_skeleton_not_a_pass(self, monkeypatch):
        small, huge = _blob("a.py"), _blob("data.jsonl", lines=4000)
        diff, excluded, _ = self._diff(monkeypatch, [small, huge],
                                       cap=10_000, ceiling=20_000)
        assert "data.jsonl" in excluded
        assert all("data.jsonl" not in p for p in [str(diff)] + list(diff.overflow))
        assert "a.py" in diff

    def test_the_fingerprint_still_sees_it(self, monkeypatch):
        """`full` answers "what does this PR touch" and feeds the mark. Dropping
        it there would make a push that changes only that file invisible to the
        nothing-new guard — the bug the fingerprint was widened to fix."""
        small, huge = _blob("a.py"), _blob("data.jsonl", lines=4000)
        diff, _, _ = self._diff(monkeypatch, [small, huge],
                                cap=10_000, ceiling=20_000)
        assert "data.jsonl" in diff.full

    def test_a_file_under_the_ceiling_still_gets_its_own_pass(self, monkeypatch):
        """The point of multi-pass: a big SOURCE file is still reviewed."""
        small, big = _blob("a.py"), _blob("src/big.py", lines=300)
        diff, excluded, _ = self._diff(monkeypatch, [small, big],
                                       cap=len(small) + 10, ceiling=1_000_000)
        assert excluded == [] and "src/big.py" in "".join(diff.overflow)


class TestNothingUnboundedReachesTheModel:
    """`expand_hunks` returns the ORIGINAL diff when expanding would breach its
    cap — `max_chars` bounds the expansion, not the prompt — and `pr_diff`
    keeps the first file of a pass at any size. Before this there was no point
    at which the text handed to the model was bounded."""

    def test_it_truncates_at_a_line_and_says_so(self):
        text = "\n".join(f"+line {i}" for i in range(2000))
        out = pr._capped(text, 500)
        assert len(out) < 900 and out.count("\n") > 5
        assert "diff truncated here" in out and "open it" in out
        assert not out.split("[diff truncated")[0].endswith("+line")

    def test_it_leaves_a_diff_under_the_limit_alone(self):
        text = "+one\n+two\n"
        assert pr._capped(text, 10_000) == text

    def test_the_prompt_itself_is_capped(self, monkeypatch):
        """The wiring, not the helper: `expand_hunks` hands back an
        un-expanded diff of any size, so the cap has to sit at the call site or
        a single big file reaches the model whole — which is what emptied the
        transcript on infra#180."""
        monkeypatch.setattr(pr, "MAX_DIFF", 1_000)
        seen = _Harness().run(monkeypatch, _blob("big.py", lines=600), [])
        assert "diff truncated here" in seen["prompts"][0]
        assert len(seen["prompts"][0]) < 20_000


class TestAPRThatIsAllOversized:
    """The 🟡 this reviewer raised on the PR that added the ceiling: when every
    file is over it, the diff is empty and the run took the generated-files
    exit — posting "nothing to review — no reviewable text in this change"
    about a 2 MB file somebody deliberately committed. "Nothing to review" and
    "I did not read it" are different sentences."""

    def _run(self, monkeypatch, excluded):
        seen = {}
        monkeypatch.setattr(pr, "pr_diff",
                            lambda *a: (pr._Diff(""), list(excluded), pr._Skipped([])))
        monkeypatch.setattr(pr, "gh", lambda *a, **k: json.dumps(
            {"draft": False, "state": "open", "merged": False, "title": "SCRUM-1 x",
             "user": {"login": "someone"}, "head": {"sha": "e" * 40}}))
        monkeypatch.setattr(pr, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(pr, "post_review",
                            lambda repo, prn, event, body, **k:
                            seen.setdefault("posted", (event, body)) and "COMMENT")
        monkeypatch.setattr(pr.status, "done",
                            lambda repo, sha, event, summary:
                            seen.setdefault("status", summary))
        monkeypatch.setattr(pr.status, "nothing_to_review",
                            lambda repo, sha, why: seen.setdefault("nothing", why))
        monkeypatch.setattr(pr.sys, "argv", ["pr-review", "app", "7"])
        monkeypatch.delenv("DRY", raising=False)
        pr.main()
        return seen

    def test_it_does_not_claim_there_was_nothing_to_review(self, monkeypatch):
        seen = self._run(monkeypatch, ["data.jsonl"])
        assert "nothing" not in seen, "said 'nothing to review' about a real change"

    def test_it_names_the_file_it_did_not_read(self, monkeypatch):
        seen = self._run(monkeypatch, ["data.jsonl"])
        event, body = seen["posted"]
        assert event == "COMMENT" and "data.jsonl" in body
        assert "NOT" in body and "1 file(s) too large to review" in seen["status"]

    def test_a_generated_only_pr_still_takes_the_quiet_exit(self, monkeypatch):
        """The other half must not regress: a PNG-only PR has genuinely nothing
        to read, and a comment on every one of those would be noise."""
        seen = self._run(monkeypatch, [])
        assert "posted" not in seen and "nothing" in seen

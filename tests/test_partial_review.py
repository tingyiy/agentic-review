"""A big PR is reviewed in part, and the part that was not is NAMED.

Tingyi's call, 2026-09-02: rather than grow the cap or summarise, review what
fits, tell both the model and the reader exactly which files did not, and never
let a partial review approve. A separate review can be launched for the rest;
the goal is a fast, reliable reviewer for day-to-day changes.

Measured before this: caeli-marketing#212 had 25 changed files, 10 reached the
model, the caveat said "the diff was truncated" and named nothing, and the last
file that fit arrived as half a hunk.
"""
import json

import pytest


@pytest.fixture
def pr():
    from agentic_review import review
    return review


def _file(path, body_lines=20):
    body = "\n".join(f"+line {i} of {path}" for i in range(body_lines))
    return (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
            f"@@ -1,0 +1,{body_lines} @@\n{body}\n")


class TestPacking:
    def _diff(self, pr, monkeypatch, blobs, cap):
        monkeypatch.setattr(pr, "gh", lambda *a, **k: "".join(blobs))
        monkeypatch.setattr(pr, "MAX_DIFF", cap)
        return pr.pr_diff("repo", 1)

    def test_whole_files_only(self, pr, monkeypatch):
        """A half hunk is worse than a named omission."""
        a, b = _file("src/a.py"), _file("src/b.py")
        diff, excluded, _ = self._diff(pr, monkeypatch, [a, b], cap=len(a) + 10)
        assert "src/a.py" in diff and "src/b.py" not in diff
        assert excluded == ["src/b.py"]
        assert diff.rstrip().endswith("+line 19 of src/a.py")

    def test_source_is_packed_before_tests_and_docs(self, pr, monkeypatch):
        """If only some of the PR fits, the source is where the defects are — a
        test that fits while its subject does not reviews the proof and never
        sees the claim."""
        t, d, s = _file("tests/test_a.py"), _file("docs/x.md"), _file("src/a.py")
        diff, excluded, _ = self._diff(pr, monkeypatch, [t, d, s], cap=len(s) + 10)
        assert "src/a.py" in diff
        assert set(excluded) == {"tests/test_a.py", "docs/x.md"}

    def test_claude_md_is_low_priority(self, pr, monkeypatch):
        c, s = _file("CLAUDE.md"), _file("src/a.py")
        diff, excluded, _ = self._diff(pr, monkeypatch, [c, s], cap=len(s) + 10)
        assert excluded == ["CLAUDE.md"]

    def test_everything_fits_means_nothing_excluded(self, pr, monkeypatch):
        a, b = _file("src/a.py"), _file("src/b.py")
        diff, excluded, _ = self._diff(pr, monkeypatch, [a, b], cap=10_000)
        assert excluded == [] and "src/b.py" in diff

    def test_the_first_file_always_fits_even_over_the_cap(self, pr, monkeypatch):
        """A cap smaller than one file must not produce an empty review."""
        a = _file("src/a.py", body_lines=200)
        diff, excluded, _ = self._diff(pr, monkeypatch, [a], cap=100)
        assert "src/a.py" in diff and excluded == []

    def test_generated_files_are_still_skipped_not_excluded(self, pr, monkeypatch):
        """Skipped is "not worth reviewing"; excluded is "worth it, did not
        fit". They must not be conflated in the note to the reader."""
        lock, s = _file("package-lock.json"), _file("src/a.py")
        diff, excluded, skipped = self._diff(pr, monkeypatch, [lock, s], cap=10_000)
        assert skipped == 1 and excluded == []


class TestTheReaderIsTold:
    FINDING = [{"file": "src/a.py", "line": 1, "severity": "low",
                "title": "t", "detail": "d"}]

    def test_the_note_names_the_files_and_comes_first(self, pr):
        body = pr.render(list(self.FINDING), True, 0, excluded=["src/b.py", "src/c.py"])
        assert "NOT reviewed" in body
        assert "src/b.py" in body and "src/c.py" in body
        assert body.index("NOT reviewed") < body.index("**t**")

    def test_an_approval_body_carries_the_note_too(self, pr):
        body = pr.approval_body(head_sha="a" * 40, repo="r", excluded=["src/b.py"])
        assert "NOT reviewed" in body and "src/b.py" in body

    def test_no_exclusions_means_no_note(self, pr):
        assert "NOT reviewed" not in pr.render(list(self.FINDING), False, 0)
        assert "NOT reviewed" not in pr.approval_body(head_sha="a" * 40, repo="r")

    def test_a_long_list_is_capped_with_a_count(self, pr):
        body = pr.render(list(self.FINDING), True, 0,
                         excluded=[f"src/f{i}.py" for i in range(40)])
        assert "and 10 more" in body


class TestAPartialReviewNeverApproves:
    """An approval is the one outcome that carries authority, and one that
    covered 10 of 25 files reads exactly like one that covered all of them."""

    def test_no_findings_with_exclusions_is_a_comment(self, pr):
        body, event = pr._finalize_review([], [], excluded=["src/b.py"])
        assert event == "COMMENT"
        assert "src/b.py" in body

    def test_no_findings_without_exclusions_still_approves(self, pr):
        _, event = pr._finalize_review([], [])
        assert event == "APPROVE"

    def test_a_blocking_finding_is_not_softened_by_exclusions(self, pr):
        f = [{"file": "a", "line": 1, "severity": "high", "title": "t", "detail": "d"}]
        _, event = pr._finalize_review(f, [], excluded=["src/b.py"])
        assert event == "REQUEST_CHANGES"


class TestTheModelIsToldByName:
    def test_the_caveat_lists_the_paths_and_says_they_are_readable(
            self, pr, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr(pr, "gh", lambda *a, **k: json.dumps(
            {"draft": False, "state": "open", "merged": False, "title": "SCRUM-1 x",
             "user": {"login": "u"}, "head": {"sha": "a" * 40}, "body": ""}))
        monkeypatch.setattr(pr, "pr_diff",
                            lambda *a: ("--- a/x\n+++ b/x\n@@\n", ["src/big.py"], 0))
        monkeypatch.setattr(pr, "_already_reviewed", lambda *a, **k: "")
        monkeypatch.setattr(pr, "checkout", lambda *a: None)
        monkeypatch.setattr(pr, "build_context", lambda *a: "")
        monkeypatch.setattr(pr, "conversation", lambda *a: "")
        monkeypatch.setattr(pr, "commit_messages", lambda *a: [])
        monkeypatch.setattr(pr, "review_findings",
                            lambda prompt, work, repo="": seen.update(prompt=prompt) or [])
        monkeypatch.setattr(pr, "_revise", lambda f, w, r: (f, []))
        monkeypatch.setattr(pr.checks, "run_all", lambda *a, **k: [])
        monkeypatch.setattr(pr, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(pr, "post_review", lambda *a, **k: "COMMENT")
        monkeypatch.setattr(pr.sys, "argv", ["pr-review", "repo", "1"])
        pr.main()
        assert "NOT INCLUDED" in seen["prompt"]
        assert "src/big.py" in seen["prompt"]
        assert "read_file works on them" in seen["prompt"]


class TestTheFingerprintIgnoresPackingOrder:
    def test_same_files_in_a_different_order_fingerprint_the_same(self, pr):
        """Packing order is the reviewer's, not the author's. A reviewer that
        reorders its input must not read that as a push — it would re-review
        every open PR on its next trigger for a change nobody made."""
        a, b = _file("src/a.py"), _file("tests/test_a.py")
        assert pr._diff_fp(a + b) == pr._diff_fp(b + a)

    def test_a_real_change_still_moves_it(self, pr):
        a = _file("src/a.py")
        assert pr._diff_fp(a) != pr._diff_fp(a.replace("line 3", "line 3 changed"))

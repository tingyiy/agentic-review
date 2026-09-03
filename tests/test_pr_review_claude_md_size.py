"""The one class of finding the LLM reviewer structurally misses.

On slack-app#378 (2026-09-01) GitHub Copilot reported a 112,848-char CLAUDE.md
against the "hard cap ~35k chars" written at line 5 of that same file. Our
reviewer ran six rounds and produced nine findings on that PR — including the
real cross-tenant attribution bug Copilot also found, plus three Copilot
missed — and never mentioned size once.

It is arithmetic against a fixed threshold, so it does not belong in a prompt:
no tokens, no sampling, nothing to hallucinate. Every CLAUDE.md in the
workspace was over 40k the morning this was written.
"""
import pathlib
import sys

import json

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from conftest import load_script  # noqa: E402


@pytest.fixture
def pr():
    return load_script("pr-review")


TOUCHES_MD = "--- a/CLAUDE.md\n+++ b/CLAUDE.md\n@@ -1 +1 @@\n-x\n+y\n"


def _repo(tmp_path, body, name="CLAUDE.md"):
    f = tmp_path / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(body)
    return str(tmp_path)


class TestItFires:
    def test_the_slack_app_378_case(self, pr, tmp_path):
        """The file states ~35k and is 3.2x that. Both numbers must appear."""
        w = _repo(tmp_path, "# CLAUDE.md\n> hard cap ~35k chars\n" + "x" * 120_000)
        f, = pr.checks.claude_md_size(w, pr._diff_paths(TOUCHES_MD))
        assert "35,000" in f["title"]
        assert f["severity"] == "low"

    def test_a_repo_with_no_stated_cap_gets_claudelints_40k(self, pr, tmp_path):
        w = _repo(tmp_path, "# CLAUDE.md\nno cap here\n" + "x" * 45_000)
        f, = pr.checks.claude_md_size(w, pr._diff_paths(TOUCHES_MD))
        assert "40,000" in f["title"] and "claudelint" in f["title"]

    def test_a_LOOSER_self_cap_does_not_raise_the_bar(self, pr, tmp_path):
        """A file may hold itself to less than 40k, not more — past 40k Claude
        Code truncates whatever the file says about itself."""
        w = _repo(tmp_path, "# CLAUDE.md\n> hard cap ~90k chars\n" + "x" * 45_000)
        f, = pr.checks.claude_md_size(w, pr._diff_paths(TOUCHES_MD))
        assert "40,000" in f["title"]

    def test_a_nested_claude_md_counts(self, pr, tmp_path):
        w = _repo(tmp_path, "x" * 45_000, name="sub/CLAUDE.md")
        diff = "--- a/sub/CLAUDE.md\n+++ b/sub/CLAUDE.md\n@@ -1 +1 @@\n-x\n+y\n"
        f, = pr.checks.claude_md_size(w, pr._diff_paths(diff))
        assert f["file"] == "sub/CLAUDE.md"

    def test_exactly_at_the_cap_fires(self, pr, tmp_path):
        w = _repo(tmp_path, "x" * 40_000)
        assert len(pr.checks.claude_md_size(w, pr._diff_paths(TOUCHES_MD))) == 1


class TestTheLineTargetIsSeparate:
    """Bytes drive TRUNCATION, lines drive ADHERENCE, and in this workspace
    they disagree: caeli-marketing was 624 lines in 42,583 bytes while infra
    was 89 lines in 67,826. A byte-only check calls the first fine; a line-only
    check calls the second fine. Neither is."""

    def test_a_long_file_well_under_the_byte_cap_still_fires(self, pr, tmp_path):
        w = _repo(tmp_path, "x\n" * 300)          # 600 bytes, 301 lines
        f, = pr.checks.claude_md_size(w, pr._diff_paths(TOUCHES_MD))
        assert "301 lines over 200" in f["title"]
        assert "bytes over" not in f["title"], "it is not over the byte cap"

    def test_a_huge_file_on_few_lines_still_fires_on_bytes(self, pr, tmp_path):
        w = _repo(tmp_path, "x" * 60_000)          # 1 line, 60k
        f, = pr.checks.claude_md_size(w, pr._diff_paths(TOUCHES_MD))
        assert "bytes over" in f["title"]
        assert "lines over" not in f["title"], "1 line is not over 200"

    def test_over_on_both_reports_both(self, pr, tmp_path):
        w = _repo(tmp_path, ("x" * 200 + "\n") * 300)
        f, = pr.checks.claude_md_size(w, pr._diff_paths(TOUCHES_MD))
        assert "bytes over" in f["title"] and "lines over" in f["title"]

    def test_exactly_200_lines_is_fine(self, pr, tmp_path):
        w = _repo(tmp_path, "x\n" * 199)          # 200 lines
        assert pr.checks.claude_md_size(w, pr._diff_paths(TOUCHES_MD)) == []


class TestItStaysQuiet:
    def test_a_pr_that_does_not_touch_it_says_nothing(self, pr, tmp_path):
        """EVERY repo is already over, so reporting on untouched files would
        put an unactionable line on every review in the workspace until
        somebody does a cleanup pass."""
        w = _repo(tmp_path, "x" * 120_000)
        other = "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        assert pr.checks.claude_md_size(w, pr._diff_paths(other)) == []

    def test_a_file_under_both_limits_says_nothing(self, pr, tmp_path):
        w = _repo(tmp_path, "x" * 39_999)   # 1 line, just under 40k
        assert pr.checks.claude_md_size(w, pr._diff_paths(TOUCHES_MD)) == []

    def test_a_deleted_claude_md_is_not_a_size_problem(self, pr, tmp_path):
        """A PR that removes one touches the path but leaves no file. This must
        not crash the whole review over a nit."""
        assert pr.checks.claude_md_size(str(tmp_path), pr._diff_paths(TOUCHES_MD)) == []

    def test_a_similarly_named_file_is_not_matched(self, pr, tmp_path):
        w = _repo(tmp_path, "x" * 120_000, name="NOTCLAUDE.md")
        diff = ("--- a/NOTCLAUDE.md\n+++ b/NOTCLAUDE.md\n@@ -1 +1 @@\n-x\n+y\n")
        assert pr.checks.claude_md_size(w, pr._diff_paths(diff)) == []


class TestItDoesNotChangeTheVerdict:
    def test_a_size_nit_alone_still_approves(self, pr, tmp_path):
        """`low` maps to APPROVE. The size is real, but it is not a reason to
        block a change that is otherwise fine — and a check that started
        withholding approval on every doc-touching PR would be turned off."""
        w = _repo(tmp_path, "x" * 45_000)
        assert pr.review_event(pr.checks.claude_md_size(w, pr._diff_paths(TOUCHES_MD))) == "APPROVE"

    def test_it_runs_AFTER_the_agent(self, pr, monkeypatch):
        """The model must not see these, or it can be nudged into repeating or
        arguing with one.

        Asserted by ORDER OF CALL, not by reading the source. A source-text
        assertion passes while the behaviour changes under it and fails when the
        code merely moves — this project has been bitten by that three times on
        one PR.
        """
        order = []
        monkeypatch.setattr(pr, "review_findings",
                            lambda *a, **k: order.append("agent") or [])
        monkeypatch.setattr(pr, "_revise", lambda f, w, r: (f, []))
        monkeypatch.setattr(pr.checks, "run_all",
                            lambda *a, **k: order.append("checks") or [])
        monkeypatch.setattr(pr, "checkout", lambda *a: None)
        monkeypatch.setattr(pr, "build_context", lambda *a: "")
        monkeypatch.setattr(pr, "conversation", lambda *a: "")
        monkeypatch.setattr(pr, "commit_messages", lambda *a: [])
        monkeypatch.setattr(pr, "pr_diff", lambda *a: ("--- a/x\n+++ b/x\n@@\n", False, 0))
        monkeypatch.setattr(pr, "_already_reviewed", lambda *a, **k: "")
        monkeypatch.setattr(pr, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(pr, "post_review", lambda *a, **k: "COMMENT")
        monkeypatch.setattr(pr, "gh", lambda *a, **k: json.dumps(
            {"draft": False, "state": "open", "merged": False, "title": "SCRUM-1 x",
             "user": {"login": "someone"}, "head": {"sha": "a" * 40}}))
        monkeypatch.setattr(pr.sys, "argv", ["pr-review", "infra", "1"])
        pr.main()
        assert order == ["agent", "checks"]


class TestDiffPaths:
    def test_it_reads_the_b_side(self, pr):
        d = ("--- a/x.py\n+++ b/x.py\n@@\n"
             "--- a/y/z.md\n+++ b/y/z.md\n@@\n")
        assert pr._diff_paths(d) == {"x.py", "y/z.md"}

    def test_a_deletion_target_is_not_a_path(self, pr):
        assert pr._diff_paths("--- a/x.py\n+++ /dev/null\n@@\n") == set()

"""What the reviewer is told before it starts looking.

The failure mode these guard against is silent: a context block that is empty,
truncated or fetched from the wrong repository still produces a review, and the
review looks normal. Only the findings it does not make are missing.
"""
import json
import re
import subprocess

import pytest

from agentic_review import context as ctx
from agentic_review import tracker


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def handler(): pass\n")
    (tmp_path / "src" / "CLAUDE.md").write_text("Handlers must validate input.\n")
    (tmp_path / "CLAUDE.md").write_text("# Root rules\nNever log PII.\n")
    (tmp_path / "README.md").write_text("# The project\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return str(tmp_path)


class TestConventionDocs:
    def test_the_root_doc_is_included(self, repo):
        out = ctx.convention_docs(repo, ["src/app.py"])
        assert "Never log PII" in out

    def test_the_doc_BESIDE_a_changed_file_is_included(self, repo):
        """The one that is easiest to miss and most specific: nothing in the
        diff points at `src/CLAUDE.md`, and it binds exactly this code."""
        out = ctx.convention_docs(repo, ["src/app.py"])
        assert "Handlers must validate input" in out

    def test_a_doc_in_an_UNTOUCHED_directory_is_not(self, repo, tmp_path):
        (tmp_path / "other").mkdir()
        (tmp_path / "other" / "CLAUDE.md").write_text("Unrelated rule.\n")
        out = ctx.convention_docs(repo, ["src/app.py"])
        assert "Unrelated rule" not in out

    def test_it_says_the_rules_are_authoritative(self, repo):
        """The block is useless if the model reads it as background."""
        assert "AUTHORITATIVE" in ctx.convention_docs(repo, ["src/app.py"])

    def test_a_repo_with_no_docs_produces_nothing(self, tmp_path):
        assert ctx.convention_docs(str(tmp_path), ["x.py"]) == ""

    def test_an_oversized_doc_is_truncated_AND_SAYS_SO(self, tmp_path):
        """A silent truncation teaches the model it has seen everything, which
        is worse than not including the file at all."""
        (tmp_path / "CLAUDE.md").write_text("x" * 60_000)
        out = ctx.convention_docs(str(tmp_path), [])
        assert len(out) < 40_000
        assert "more chars" in out and "CLAUDE.md" in out

    def test_the_total_budget_is_enforced_across_docs(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("y" * 20_000)
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "CLAUDE.md").write_text("z" * 20_000)
        (tmp_path / "a" / "b").mkdir()
        (tmp_path / "a" / "b" / "CLAUDE.md").write_text("w" * 20_000)
        out = ctx.convention_docs(str(tmp_path), ["a/b/x.py"])
        assert len(out) < ctx.MAX_DOCS_TOTAL + 4_000

    def test_the_ROOT_doc_survives_the_budget(self, tmp_path):
        """If something has to be dropped it must not be the one a human would
        read first."""
        (tmp_path / "CLAUDE.md").write_text("ROOTRULE\n" + "y" * 10_000)
        for i in range(6):
            d = tmp_path / f"d{i}"
            d.mkdir()
            (d / "CLAUDE.md").write_text("x" * 12_000)
        out = ctx.convention_docs(str(tmp_path),
                                  [f"d{i}/f.py" for i in range(6)])
        assert "ROOTRULE" in out


class TestRepoMap:
    def test_it_lists_directories_with_counts(self, repo):
        out = ctx.repo_map(repo)
        assert "src/" in out and "file" in out

    def test_it_is_empty_outside_a_git_checkout(self, tmp_path):
        assert ctx.repo_map(str(tmp_path)) == ""

    def test_a_huge_tree_keeps_the_BIGGEST_directories(self, tmp_path):
        """Pruning alphabetically loses whatever sorts late, which is arbitrary
        — and the big directories are the ones that answer "where would this
        already live?"."""
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        for i in range(30):
            d = tmp_path / f"dir{i:02d}"
            d.mkdir()
            for n in range(1 if i else 20):
                (d / f"f{n}.py").write_text("x")
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        out = ctx.repo_map(str(tmp_path), max_entries=3)
        assert "dir00/" in out, "the 20-file directory was pruned"
        assert "omitted" in out


class TestLinkedPRRefs:
    def test_a_bare_hash_means_THIS_repo(self):
        assert ctx.linked_pr_refs("infra", ["see #141"]) == [("infra", 141)]

    def test_a_url_carries_its_own_repo(self):
        refs = ctx.linked_pr_refs(
            "infra", ["see https://github.com/example-org/slack-app/pull/378"])
        assert refs == [("slack-app", 378)]

    def test_a_markdown_link_produces_ONE_ref_not_two(self):
        """`[#378](…/slack-app/pull/378)` is one pull request. Read naively it
        is two: the real one, and a same-numbered PR in the repo under review
        that is somebody else's change entirely. Wrong context is worse than
        missing context."""
        refs = ctx.linked_pr_refs(
            "infra",
            ["see [#378](https://github.com/example-org/slack-app/pull/378)"])
        assert refs == [("slack-app", 378)]

    def test_a_number_claimed_by_ANOTHER_repos_url_is_not_read_locally(self):
        """Same rule when the URL points somewhere we cannot read at all: the
        number is spoken for, so it must not become a local reference."""
        assert ctx.linked_pr_refs(
            "infra", ["[#12](https://github.com/someone-else/repo/pull/12)"]) == []

    def test_an_unclaimed_bare_hash_still_counts_alongside_a_url(self):
        """The guard must not swallow a genuine second reference."""
        refs = ctx.linked_pr_refs(
            "infra", ["https://github.com/example-org/slack-app/pull/378 and #9"])
        assert refs == [("slack-app", 378), ("infra", 9)]

    def test_another_org_is_ignored(self):
        """Fetching it would 404 at best; at worst it presents somebody else's
        code as context for ours."""
        assert ctx.linked_pr_refs(
            "infra", ["https://github.com/someone-else/repo/pull/1"]) == []

    def test_duplicates_collapse(self):
        assert ctx.linked_pr_refs("infra", ["#5 and #5 again", "#5"]) == [("infra", 5)]

    def test_a_path_like_number_is_not_a_pr(self):
        """`docs/rules#3` and `a/b/#4` are not PR references."""
        assert ctx.linked_pr_refs("infra", ["see docs/rules#3"]) == []

    def test_a_bare_hash_in_TICKET_prose_is_not_a_pr(self):
        """`#3` in a Jira description is a heading, an ordinal, a version or a
        channel. Reading three tickets that way on slack-app#381 produced #4, #3
        and #121 — none of them related to the change."""
        assert ctx.linked_pr_refs("infra", url_only_texts=["see section #3"]) == []

    def test_a_URL_in_ticket_prose_still_counts(self):
        """The other half: a real link in a ticket is often the only place the
        paired PR is named."""
        assert ctx.linked_pr_refs(
            "infra",
            url_only_texts=["fixed by https://github.com/example-org/infra/pull/7"]
        ) == [("infra", 7)]


class TestLinkedPRs:
    def _fetch(self, payloads):
        def fetch(path):
            for key, value in payloads.items():
                if key in path:
                    return json.dumps(value)
            raise KeyError(path)
        return fetch

    def test_it_reports_state_title_and_FILES(self):
        fetch = self._fetch({
            "/files": [{"filename": "src/app.py"}],
            "/pulls/141": {"title": "SCRUM-1 the other half", "merged": True,
                           "state": "closed", "body": "does the thing"},
        })
        out = ctx.linked_prs([("infra", 141)], fetch)
        assert "infra#141 [merged]" in out
        assert "the other half" in out
        assert "src/app.py" in out

    def test_an_unreadable_link_is_skipped_not_fatal(self):
        def fetch(path):
            raise RuntimeError("404")
        assert ctx.linked_prs([("infra", 1)], fetch) == ""

    def test_one_dead_link_does_not_lose_a_live_one(self):
        def fetch(path):
            if "/pulls/1" in path and "/files" not in path:
                raise RuntimeError("404")
            if "/files" in path:
                return json.dumps([])
            return json.dumps({"title": "live", "state": "open"})
        out = ctx.linked_prs([("infra", 1), ("infra", 2)], fetch)
        assert "infra#2" in out

    def test_the_pr_under_review_is_skipped(self):
        """`#<n>` appears in a PR's own body more often than you would think,
        and the result reads as a mysterious duplicate of itself."""
        def fetch(path):
            pytest.fail("fetched the PR under review as context for itself")
        assert ctx.linked_prs([("infra", 9)], fetch, skip={("infra", 9)}) == ""

    def test_it_is_bounded(self):
        seen = []

        def fetch(path):
            seen.append(path)
            return json.dumps([] if "/files" in path else {"title": "x", "state": "open"})
        ctx.linked_prs([("infra", n) for n in range(20)], fetch, limit=2)
        assert len([p for p in seen if "/files" in p]) == 2


class TestTicketIds:
    def test_it_finds_ids_in_order_without_duplicates(self):
        assert tracker.ticket_ids("SCRUM-2 fix", "relates to SCRUM-1 and SCRUM-2") \
            == ["SCRUM-2", "SCRUM-1"]

    def test_a_version_string_is_not_a_ticket(self):
        assert tracker.ticket_ids("bump to v2-1") == []

    def test_empty_input_is_fine(self):
        assert tracker.ticket_ids("", None) == []


class TestTrackerRender:
    TICKET = {"key": "SCRUM-1", "summary": "Do the thing", "status": "In Progress",
              "type": "Story", "description": "It must validate the input.",
              "comments": [{"who": "A person", "body": "and reject empties"}]}

    def test_it_carries_the_description_and_comments(self):
        out = tracker.render([self.TICKET])
        assert "It must validate the input" in out
        assert "and reject empties" in out

    def test_it_tells_the_model_intent_is_reviewable(self):
        assert "does not do what was asked is a finding" in tracker.render([self.TICKET])

    def test_it_warns_against_inventing_scope(self):
        """Without this the model files "the ticket also mentions X" against a
        PR that never claimed to do X — the most common way ticket context makes
        a review worse."""
        assert "Do NOT invent scope" in tracker.render([self.TICKET])

    def test_nothing_fetched_means_no_section(self):
        assert tracker.render([]) == ""
        assert tracker.render([None]) == ""


class TestADFText:
    def test_it_flattens_a_real_description(self):
        doc = {"type": "doc", "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "Line one."}]},
            {"type": "bulletList", "content": [
                {"type": "listItem", "content": [
                    {"type": "paragraph",
                     "content": [{"type": "text", "text": "a point"}]}]}]}]}
        out = tracker._text(doc)
        assert "Line one." in out and "a point" in out

    def test_a_plain_string_passes_through(self):
        assert tracker._text("just text") == "just text"

    def test_an_unknown_node_still_yields_its_text(self):
        """Degrading to slightly wrong whitespace beats degrading to nothing."""
        doc = {"type": "somethingNew",
               "content": [{"type": "text", "text": "still readable"}]}
        assert "still readable" in tracker._text(doc)


class TestCheckResults:
    """Tingyi's suggestion, 2026-09-02: the author's own test results are the
    cheapest evidence there is, and they are already on the commit. Read via
    the Actions API, because that is what the bot token can reach."""

    def _fetch(self, runs_jobs):
        """runs_jobs: list of (workflow_name, [job dicts])."""
        def fetch(path):
            if "/actions/runs?" in path:
                return json.dumps({"workflow_runs": [
                    {"id": i, "name": wf} for i, (wf, _) in enumerate(runs_jobs)]})
            m = re.search(r"/actions/runs/(\d+)/jobs", path)
            if m:
                return json.dumps({"jobs": runs_jobs[int(m.group(1))][1]})
            raise KeyError(path)
        return fetch

    def test_a_failing_job_names_the_failing_tests_from_its_log(self):
        log = ("2026-09-02T10:00:00.000Z ...\n"
               "2026-09-02T10:00:01.000Z FAILED tests/test_x.py::test_a - AssertionError: 3 != 2\n"
               "2026-09-02T10:00:02.000Z not ok 4 - renders the header\n")
        out = ctx.check_results("r", "abc", self._fetch(
            [("Tests", [{"id": 7, "name": "unit", "status": "completed",
                         "conclusion": "failure"}])]),
            fetch_log=lambda path: log)
        assert "Tests / unit: failure" in out
        assert "FAILED tests/test_x.py::test_a" in out
        assert "not ok 4 - renders the header" in out

    def test_a_passing_job_is_reported_as_evidence(self):
        out = ctx.check_results("r", "abc", self._fetch(
            [("Tests", [{"id": 1, "name": "unit", "status": "completed",
                         "conclusion": "success"}])]))
        assert "Tests / unit: success" in out
        assert "do not invent a failure it would have caught" in out

    def test_a_running_job_is_PENDING_not_absent(self):
        """The review and the unit job start on the same push. "No results"
        and "still running" mean different things to a reviewer."""
        out = ctx.check_results("r", "abc", self._fetch(
            [("Tests", [{"id": 1, "name": "unit", "status": "in_progress"}])]))
        assert "still running" in out and "unit" in out
        assert "unknown, not as absent" in out

    def test_our_own_review_workflow_is_not_a_test(self):
        out = ctx.check_results("r", "abc", self._fetch(
            [("PR review", [{"id": 1, "name": "review (Caeli)",
                             "status": "completed", "conclusion": "success"}])]))
        assert out == ""

    def test_no_runs_means_no_section(self):
        assert ctx.check_results("r", "abc", self._fetch([])) == ""

    def test_an_unreadable_api_is_not_fatal_and_names_the_permission(self, capsys):
        def boom(path):
            raise RuntimeError("HTTP Error 403: Forbidden")
        assert ctx.check_results("r", "abc", boom) == ""
        assert "Actions: Read-only" in capsys.readouterr().out

    def test_an_unreadable_log_still_reports_the_failure(self):
        def bad_log(path):
            raise RuntimeError("blob 403")
        out = ctx.check_results("r", "abc", self._fetch(
            [("Tests", [{"id": 7, "name": "unit", "status": "completed",
                         "conclusion": "failure"}])]), fetch_log=bad_log)
        assert "Tests / unit: failure" in out

    def test_failure_lines_are_deduped_and_bounded(self):
        log = "\n".join(["FAILED tests/test_x.py::test_a"] * 5
                        + [f"FAILED tests/test_y.py::test_{i}" for i in range(40)])
        out = ctx.check_results("r", "abc", self._fetch(
            [("Tests", [{"id": 7, "name": "unit", "status": "completed",
                         "conclusion": "failure"}])]), fetch_log=lambda p: log)
        assert out.count("test_x.py::test_a") == 1
        assert out.count("FAILED") <= ctx.MAX_FAIL_LINES


class TestFailureLinesMatchARealRunnerLog:
    """Built from caeli-marketing job 100095120836, not from imagination. The
    first pattern extracted NOTHING from that log: the output is ANSI-coloured
    (so `×` never sits at a line start) and the reliable signal is the runner's
    `##[error]` annotation, which the pattern did not allow for."""

    ESC = "\x1b[31m"
    RESET = "\x1b[39m"
    STAMP = "2026-09-02T01:54:10.1234567Z "
    REAL = "\n".join([
        STAMP + " \x1b[32m✓\x1b[39m test/harness.test.tsx \x1b[2m(2 tests)\x1b[22m 30ms",
        STAMP + "\x1b[31m⎯⎯⎯⎯⎯ Uncaught Exception ⎯⎯⎯⎯⎯\x1b[39m",
        STAMP + "TypeError: answerRef.current?.scrollIntoView is not a function",
        STAMP + " \x1b[36m❯ components/answer/answer-home.tsx:241:28\x1b[39m",
        STAMP + "    239|       // visitor watches it work",
        STAMP + " Test Files  58 passed (58)",
        STAMP + "      Tests  684 passed (684)",
        STAMP + "     Errors  1 error",
        STAMP + "##[error]TypeError: answerRef.current?.scrollIntoView is not a function",
        STAMP + " ❯ components/answer/answer-home.tsx:241:28",
        STAMP + " ❯ Timeout._onTimeout test/setup.ts:67:5",
        STAMP + "##[error]Process completed with exit code 1.",
    ])

    def test_it_finds_the_error_and_its_location(self):
        out = ctx.failure_lines(self.REAL)
        joined = "\n".join(out)
        assert "scrollIntoView is not a function" in joined
        assert "at components/answer/answer-home.tsx:241:28" in joined
        assert "Errors 1 error" in joined
        assert "Uncaught Exception" in joined

    def test_the_process_exit_line_is_not_evidence(self):
        assert not any("exit code 1" in l for l in ctx.failure_lines(self.REAL))

    def test_ansi_and_timestamps_are_stripped(self):
        joined = "\n".join(ctx.failure_lines(self.REAL))
        assert "\x1b[" not in joined and "T01:54" not in joined

    def test_a_passing_line_is_not_a_failure(self):
        assert not any("harness.test.tsx" in l for l in ctx.failure_lines(self.REAL))

    def test_only_the_FIRST_frame_after_an_error_is_kept(self):
        """One location, not a stack."""
        out = ctx.failure_lines(self.REAL)
        assert sum(1 for l in out if l.startswith("    at ")) <= 2
        assert not any("_onTimeout" in l for l in out)

    def test_pytest_and_node_test_still_match(self):
        log = (self.STAMP + "FAILED tests/test_x.py::test_a - AssertionError: 3 != 2\n"
               + self.STAMP + "not ok 4 - renders the header\n"
               + self.STAMP + "\x1b[31m×\x1b[39m renders the footer 12ms\n")
        joined = "\n".join(ctx.failure_lines(log))
        assert "FAILED tests/test_x.py::test_a" in joined
        assert "not ok 4 - renders the header" in joined
        assert "× renders the footer" in joined

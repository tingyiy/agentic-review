"""The shape of a file the diff had no room to show.

caeli-marketing#233: an 87k diff against a 60k budget, and four test files
dropped with "were NOT reviewed". They were 200-line files in a checkout the
agent was standing in, with `read_file` and `grep` in its hands. What it
lacked was a reason to look — a name in a list is not one.
"""
import pathlib

import pytest

from agentic_review import context as ctx


def _repo(tmp_path, files):
    for path, body in files.items():
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return str(tmp_path)


class TestWhatASkeletonSays:
    def test_the_length_and_the_declarations_with_line_numbers(self, tmp_path):
        w = _repo(tmp_path, {"src/a.py": "import os\n\n\ndef alpha():\n    pass\n\n\nclass Beta:\n    pass\n"})
        row = ctx.file_skeleton(w, "src/a.py")
        assert "`src/a.py` (9 lines)" in row
        assert "alpha:4" in row and "Beta:8" in row

    @pytest.mark.parametrize("body,expected", [
        ("export function handleClick() {}\n", "handleClick:1"),
        ("const useThing = (a) => {\n}\n", "useThing:1"),
        ("export const useOther = async () => {}\n", "useOther:1"),
        ("interface WirePayload {\n}\n", "WirePayload:1"),
        ("type Verdict = 'a'\n", "Verdict:1"),
        ("func ServeHTTP(w http.ResponseWriter) {\n}\n", "ServeHTTP:1"),
        ("async def fetch_it():\n    pass\n", "fetch_it:1"),
    ])
    def test_the_languages_these_repositories_actually_use(self, tmp_path, body, expected):
        w = _repo(tmp_path, {"f.ts": body})
        assert expected in ctx.file_skeleton(w, "f.ts")

    def test_a_file_with_nothing_to_declare_still_reports_its_length(self, tmp_path):
        w = _repo(tmp_path, {"data.json": '{\n  "a": 1\n}\n'})
        row = ctx.file_skeleton(w, "data.json")
        assert "3 lines" in row and "no declarations found" in row

    def test_minified_lines_are_not_mined_for_declarations(self, tmp_path):
        w = _repo(tmp_path, {"b.js": "function a(){}" + "x" * 400 + "\n"})
        assert "no declarations found" in ctx.file_skeleton(w, "b.js")

    def test_a_long_file_is_summarised_not_transcribed(self, tmp_path):
        body = "".join(f"def fn_{i}():\n    pass\n" for i in range(50))
        w = _repo(tmp_path, {"many.py": body})
        row = ctx.file_skeleton(w, "many.py")
        assert row.count(":") <= ctx.MAX_SKELETON_DECLS + 2
        assert "…" in row

    def test_an_unreadable_file_is_absent_rather_than_guessed_at(self, tmp_path):
        assert ctx.file_skeleton(str(tmp_path), "gone.py") == ""


class TestTheSection:
    def test_it_tells_the_model_the_files_are_readable(self, tmp_path):
        w = _repo(tmp_path, {"src/a.py": "def alpha():\n    pass\n"})
        out = ctx.skeletons(w, ["src/a.py"])
        assert "`read_file` reaches any" in out
        assert "unreviewed if you do not look" in out
        assert "alpha:1" in out

    def test_no_files_at_all_is_silent(self, tmp_path):
        assert ctx.skeletons(str(tmp_path), []) == ""

    def test_an_unreadable_file_is_still_NAMED(self, tmp_path):
        """It is a changed file either way. Dropping it silently because its
        shape cannot be read is a regression in coverage for exactly the files
        worth asking about — deleted here, or a symlink out of the checkout."""
        out = ctx.skeletons(str(tmp_path), ["missing.py"])
        assert "missing.py" in out and "shape unavailable" in out

    def test_the_number_of_files_is_capped(self, tmp_path):
        files = {f"f{i}.py": "def a():\n    pass\n" for i in range(40)}
        w = _repo(tmp_path, files)
        out = ctx.skeletons(w, sorted(files))
        assert out.count("` (") <= ctx.MAX_SKELETON_FILES
        assert "and 15 more not shown" in out

    def test_the_whole_section_is_bounded(self, tmp_path):
        files = {f"dir{i}/{'n' * 80}.py": "def a():\n    pass\n" for i in range(30)}
        w = _repo(tmp_path, files)
        out = ctx.skeletons(w, sorted(files))
        assert len(out) < ctx.MAX_SKELETON_CHARS + 600

    def test_it_costs_a_fraction_of_the_diff_it_replaces(self, tmp_path):
        """The trade this exists to make: ~11k characters of diff became a few
        hundred of shape, and the file stayed reachable."""
        body = "".join(f"export function fn{i}(a, b) {{\n  return a + b;\n}}\n"
                       for i in range(120))
        w = _repo(tmp_path, {"big.ts": body})
        out = ctx.skeletons(w, ["big.ts"])
        assert len(out) < len(body) / 4


class TestUnreviewedMeansUnopened:
    """A file the diff had no room for is not unreviewed if the agent went and
    read it. Blanket-disclaiming four files the model may well have opened is
    a false caveat, and a false caveat is worse than none: it teaches the
    reader to skip the box."""

    def _drive(self, monkeypatch, opened):
        import json
        from agentic_review import review as pr
        posted = {}
        monkeypatch.setattr(pr, "pr_diff",
                            lambda *a: ("--- a/x\n+++ b/x\n@@\n+x\n",
                                        ["src/big.py", "src/other.py"], 0))
        monkeypatch.setattr(pr, "_already_reviewed", lambda *a, **k: "")
        monkeypatch.setattr(pr, "checkout", lambda *a: None)
        monkeypatch.setattr(pr, "build_context", lambda *a: "")
        monkeypatch.setattr(pr.ctx, "skeletons", lambda *a: "")
        monkeypatch.setattr(pr, "conversation", lambda *a: "")
        monkeypatch.setattr(pr, "commit_messages", lambda *a: [])
        monkeypatch.setattr(pr, "_revise", lambda f, w, r: (f, []))
        monkeypatch.setattr(pr.checks, "run_all", lambda *a, **k: [])
        monkeypatch.setattr(pr, "_pr_is_gone", lambda *a: None)

        def review_findings(prompt, work, repo=""):
            # What `_run_agent` accumulates across every pass — the initial
            # one, the confirmation and the revision — which is what `main`
            # reads. Per-pass `stats` is replaced each time and cannot carry it.
            pr._CURRENT["opened"] = set(opened)
            return []
        monkeypatch.setattr(pr, "review_findings", review_findings)
        monkeypatch.setattr(pr, "post_review",
                            lambda repo, n, ev, body, **k: posted.update(
                                event=ev, body=body) or ev)
        monkeypatch.setattr(pr, "gh", lambda *a, **k: json.dumps(
            {"draft": False, "state": "open", "merged": False,
             "title": "SCRUM-1 x", "user": {"login": "someone"},
             "head": {"sha": "a" * 40}}))
        monkeypatch.setattr(pr.sys, "argv", ["pr-review", "repo", "1"])
        pr.main()
        return posted

    def test_a_file_the_agent_read_is_not_called_unreviewed(self, monkeypatch):
        posted = self._drive(monkeypatch, opened={"src/big.py"})
        assert "src/other.py" in posted["body"]
        assert "src/big.py" not in posted["body"]

    def test_reading_all_of_them_removes_the_caveat_entirely(self, monkeypatch):
        posted = self._drive(monkeypatch, opened={"src/big.py", "src/other.py"})
        assert "Partial review" not in posted["body"]

    def test_but_it_still_may_not_approve(self, monkeypatch):
        """READING A FILE IS NOT SEEING ITS CHANGE. `read_file` returns the
        head version; the diff of an excluded file is never shown to the agent
        at all. So the caveat can go — nobody's file went unlooked-at — while
        the approval cap stays, because an approval would be claiming to have
        reviewed changes this review never saw.

        This test asserted the opposite until this reviewer pointed out the
        overclaim on the PR that introduced it."""
        posted = self._drive(monkeypatch, opened={"src/big.py", "src/other.py"})
        assert "Partial review" not in posted["body"]
        assert posted["event"] == "COMMENT"

    def test_reading_none_of_them_keeps_the_full_caveat(self, monkeypatch):
        posted = self._drive(monkeypatch, opened=set())
        assert "2 changed file(s) were NOT opened" in posted["body"]
        assert posted["event"] == "COMMENT"


class TestTheSecondRoundOfFindings:
    """Five from the review of this PR — four from the other reviewer, one
    from this one, running hosted against its own change for the first time."""

    def test_the_skeleton_read_is_contained(self, tmp_path):
        """A changed path can be a symlink to somewhere else on the host. The
        agent's own `Workspace.resolve` has refused that from the beginning and
        this read was going around it."""
        outside = tmp_path / "outside" / "secret.py"
        outside.parent.mkdir(parents=True)
        outside.write_text("def leaked():\n    pass\n")
        work = tmp_path / "work"
        work.mkdir()
        (work / "link.py").symlink_to(outside)
        assert ctx.file_skeleton(str(work), "link.py") == ""
        assert ctx.file_skeleton(str(work), "../outside/secret.py") == ""
        assert ctx.file_skeleton(str(work), "/etc/hosts") == ""

    def test_a_device_is_not_read(self, tmp_path):
        """`/dev/zero` would have hung the review before the agent started."""
        work = tmp_path / "w"
        work.mkdir()
        try:
            (work / "z").symlink_to("/dev/zero")
        except OSError:
            pytest.skip("cannot symlink /dev/zero here")
        assert ctx.file_skeleton(str(work), "z") == ""

    def test_the_ellipsis_means_something_was_actually_cut(self, tmp_path):
        """Exactly `MAX_SKELETON_DECLS` declarations is not a truncation."""
        exact = "".join(f"def fn_{i}():\n    pass\n"
                        for i in range(ctx.MAX_SKELETON_DECLS))
        w = _repo(tmp_path, {"exact.py": exact})
        assert "…" not in ctx.file_skeleton(w, "exact.py")
        more = exact + "def one_more():\n    pass\n"
        w2 = _repo(tmp_path / "b", {"more.py": more})
        assert "…" in ctx.file_skeleton(w2, "more.py")

    def test_a_failed_read_is_not_recorded_as_opened(self):
        """Our own reviewer's finding: recording before the call meant a read
        that failed — an escaping path, a missing file — still counted, so the
        caveat would drop a file nobody had seen."""
        from agentic_review import agent
        stats = {}
        agent._record_opened(stats, "read_file", '{"path": "src/a.py"}',
                             "error: ValueError: path escapes the checkout")
        assert stats.get("opened") is None
        agent._record_opened(stats, "read_file", '{"path": "src/a.py"}',
                             "1\tdef a():")
        assert stats["opened"] == {"src/a.py"}

    def test_the_path_spelling_is_normalised(self):
        """`read_file("./src/big.py")` reads the same file as `src/big.py`, and
        an unnormalised record made the caveat claim it was never opened."""
        from agentic_review import agent
        stats = {}
        agent._record_opened(stats, "read_file", '{"path": "./src/big.py"}', "ok")
        assert stats["opened"] == {"src/big.py"}


class TestReadingThemAllUnblocksTheStaleBlock:
    """caeli-marketing#233, live: a blocking finding was fixed, later reviews
    found only nits, and the PR stayed at CHANGES_REQUESTED. The dismissal was
    suppressed because the review was truncated — right while an unshown file
    was also unread, and wrong once the agent has opened them."""

    def _truncated_seen_by_post(self, monkeypatch, opened):
        import json
        from agentic_review import review as pr
        seen = {}
        monkeypatch.setattr(pr, "pr_diff",
                            lambda *a: ("--- a/x\n+++ b/x\n@@\n+x\n",
                                        ["src/big.py"], 0))
        monkeypatch.setattr(pr, "_already_reviewed", lambda *a, **k: "")
        monkeypatch.setattr(pr, "checkout", lambda *a: None)
        monkeypatch.setattr(pr, "build_context", lambda *a: "")
        monkeypatch.setattr(pr.ctx, "skeletons", lambda *a: "")
        monkeypatch.setattr(pr, "conversation", lambda *a: "")
        monkeypatch.setattr(pr, "commit_messages", lambda *a: [])
        monkeypatch.setattr(pr, "_revise", lambda f, w, r: (f, []))
        monkeypatch.setattr(pr.checks, "run_all", lambda *a, **k: [])
        monkeypatch.setattr(pr, "_pr_is_gone", lambda *a: None)

        def review_findings(prompt, work, repo=""):
            pr._CURRENT["opened"] = set(opened)
            return []
        monkeypatch.setattr(pr, "review_findings", review_findings)
        monkeypatch.setattr(pr, "post_review",
                            lambda repo, n, ev, body, head_sha="", truncated=False,
                            unread=(), pr_files=():
                            seen.update(truncated=truncated, unread=list(unread)) or ev)
        monkeypatch.setattr(pr, "gh", lambda *a, **k: json.dumps(
            {"draft": False, "state": "open", "merged": False,
             "title": "SCRUM-1 x", "user": {"login": "someone"},
             "head": {"sha": "a" * 40}}))
        monkeypatch.setattr(pr.sys, "argv", ["pr-review", "repo", "1"])
        pr.main()
        return seen["truncated"]

    def test_an_unread_file_still_counts_as_truncated(self, monkeypatch):
        """The guard keeps protecting the case it was written for."""
        assert self._truncated_seen_by_post(monkeypatch, opened=set()) is True

    def test_reading_it_makes_the_review_whole(self, monkeypatch):
        """`_dismiss_stale_block` may then clear our own stale block, because
        the agent did look where the old finding was."""
        assert self._truncated_seen_by_post(
            monkeypatch, opened={"src/big.py"}) is False


class TestTheThirdRoundFromOurOwnReviewer:
    """Five findings, from this reviewer running hosted against this PR."""

    def test_the_revision_pass_contributes_its_opened_files(self, monkeypatch):
        """`_revise` had its own `stats` and dropped them, so a file opened
        only while reconsidering was still reported as never opened — while
        the commit message claimed the accumulation was complete."""
        import json
        from agentic_review import review as pr
        monkeypatch.setitem(pr._CURRENT, "opened", {"already.py"})
        # `_revise` resumes the review pass's conversation, so there has to be
        # one for it to get as far as the agent.
        monkeypatch.setitem(pr._CURRENT, "stats", {"messages": [{"role": "user"}]})
        monkeypatch.setattr(pr.agent, "resume", lambda *a, **k:
                            (k["stats"].update(opened={"during_revision.py"},
                                               tool_calls=1)
                             or (json.dumps({"revisions": [
                                 {"index": 0, "action": "keep"}]}), [])))
        pr._revise([{"file": "a", "line": 1, "severity": "low",
                     "title": "t", "detail": "d"}], "/tmp", "repo")
        assert pr._CURRENT["opened"] == {"already.py", "during_revision.py"}

    def test_a_partial_read_does_not_count_as_opened(self):
        """`_read_file` returns 200 lines and says so. Counting that as opened
        cleared the caveat on a 255-line file the agent had seen four fifths
        of, and let the review approve on it."""
        from agentic_review import agent
        stats = {}
        agent._record_opened(stats, "read_file", '{"path": "big.py"}',
                             "1\tx\n[showing lines 1-200 of 255. "
                             "Call again with offset=201 for the rest.]")
        assert stats.get("opened") is None

    def test_a_clipped_result_is_not_an_opened_file(self):
        """The footer is APPENDED and `_truncate` keeps the HEAD, so a window
        whose numbered lines run past MAX_TOOL_CHARS loses the footer — the
        large file where this matters most. A 🔴 from this reviewer on this
        PR, against the fix in the commit before it."""
        from agentic_review import agent
        stats = {}
        agent._record_opened(
            stats, "read_file", '{"path": "huge.py"}',
            "1\tx\n" * 50 + "\n\n[... truncated: 4211 more chars. "
            "Narrow the pattern, or read a specific range with offset/limit.]")
        assert stats.get("opened") is None

    def test_a_real_clipped_read_of_a_real_file_is_not_counted(self, tmp_path):
        """Driven through `_read_file` itself rather than a hand-written
        string, so the two markers cannot drift from the code that writes
        them."""
        from agentic_review import agent
        big = tmp_path / "wide.py"
        big.write_text("".join(f"x = {'a' * 200}  # {i}\n" for i in range(400)))
        ws = agent.Workspace(str(tmp_path))
        result = agent._read_file(ws, "wide.py")
        assert len(result) <= agent.MAX_TOOL_CHARS + 200
        stats = {}
        agent._record_opened(stats, "read_file", '{"path": "wide.py"}', result)
        assert stats.get("opened") is None, (
            "a clipped window is not a read of the file")

    def test_a_whole_read_still_counts(self):
        from agentic_review import agent
        stats = {}
        agent._record_opened(stats, "read_file", '{"path": "small.py"}',
                             "1\tdef a():\n2\t    pass")
        assert stats["opened"] == {"small.py"}

    def test_a_clipped_file_says_its_length_is_a_prefix(self, tmp_path):
        """Reporting the prefix's line count as the file's length is a wrong
        shape stated confidently."""
        big = tmp_path / "huge.py"
        big.write_text("x = 1\n" * (ctx.MAX_SKELETON_READ // 3))
        row = ctx.file_skeleton(str(tmp_path), "huge.py")
        assert "of a larger file" in row

    def test_a_small_file_reports_its_real_length(self, tmp_path):
        w = _repo(tmp_path, {"s.py": "a = 1\nb = 2\n"})
        assert "(2 lines)" in ctx.file_skeleton(w, "s.py")


class TestTheFourthRoundFromOurOwnReviewer:
    def test_a_tail_only_read_is_not_a_full_read(self):
        """`read_file("big.py", offset=800)` on a 900-line file reaches the
        end, so it carries no partial-read footer — and lines 1-799 were never
        seen. Reaching EOF says where the read stopped, not where it began."""
        from agentic_review import agent
        stats = {}
        agent._record_opened(stats, "read_file",
                             '{"path": "big.py", "offset": 800}', "800\tx")
        assert stats.get("opened") is None

    def test_a_read_from_the_top_still_counts(self):
        from agentic_review import agent
        for raw in ('{"path": "a.py"}', '{"path": "a.py", "offset": 1}'):
            stats = {}
            agent._record_opened(stats, "read_file", raw, "1\tx")
            assert stats["opened"] == {"a.py"}, raw

    def test_a_nonsense_offset_does_not_count(self):
        from agentic_review import agent
        stats = {}
        agent._record_opened(stats, "read_file",
                             '{"path": "a.py", "offset": "top"}', "1\tx")
        assert stats.get("opened") is None


class TestTheFifthRoundFromOurOwnReviewer:
    def test_a_wholly_dropped_skeleton_list_still_names_the_count(self, tmp_path):
        """When every row exceeds the character budget, `rows` is empty — and
        returning "" put the excluded files nowhere at all: not in the diff,
        not in the prompt."""
        long_path = "src/" + "d" * 200
        files = {f"{long_path}{i}.py": "def a():\n    pass\n" for i in range(20)}
        w = _repo(tmp_path, files)
        out = ctx.skeletons(w, sorted(files))
        assert out != ""
        assert "more not shown" in out

    def test_the_approval_cap_is_keyed_on_the_diff_not_the_read(self):
        """Three questions were riding on one flag. This is the one that must
        stay keyed on what was SHOWN."""
        from agentic_review import review as pr
        _, event = pr._finalize_review([], [], excluded=[],
                                       saw_every_change=False)
        assert event == "COMMENT"
        _, event = pr._finalize_review([], [], excluded=[],
                                       saw_every_change=True)
        assert event == "APPROVE"

    def test_an_older_caller_keeps_its_behaviour(self):
        """`saw_every_change` defaults to what `excluded` says."""
        from agentic_review import review as pr
        assert pr._finalize_review([], [], excluded=["a.py"])[1] == "COMMENT"
        assert pr._finalize_review([], [], excluded=[])[1] == "APPROVE"


class TestSequentialWindowsCoverTheFile:
    """The normal way to read a 255-line file is two windows. Requiring one
    top-to-end read reported a fully-read file as unopened — a FALSE caveat on
    every large file, and the exact opposite of the failure the requirement
    was added to prevent. Both are wrong; the ranges are unioned now."""

    def _read(self, stats, path, result, offset=None):
        import json
        from agentic_review import agent
        args = {"path": path}
        if offset is not None:
            args["offset"] = offset
        agent._record_opened(stats, "read_file", json.dumps(args), result)

    def test_two_windows_together_are_the_file(self):
        stats = {}
        self._read(stats, "big.py",
                   "1\tx\n[showing lines 1-200 of 255. Call again with "
                   "offset=201 for the rest.]")
        assert stats.get("opened") is None, "one window is not the file"
        self._read(stats, "big.py", "201\tx", offset=201)
        assert stats["opened"] == {"big.py"}

    def test_a_gap_in_the_middle_is_not_coverage(self):
        stats = {}
        self._read(stats, "g.py",
                   "1\tx\n[showing lines 1-100 of 300. Call again with "
                   "offset=101 for the rest.]")
        self._read(stats, "g.py", "201\tx", offset=201)
        assert stats.get("opened") is None, "lines 101-200 were never shown"

    def test_overlapping_windows_still_count(self):
        stats = {}
        self._read(stats, "o.py",
                   "1\tx\n[showing lines 1-200 of 255. Call again with "
                   "offset=201 for the rest.]")
        self._read(stats, "o.py", "150\tx", offset=150)
        assert stats["opened"] == {"o.py"}

    def test_the_tail_alone_is_still_not_the_file(self):
        stats = {}
        self._read(stats, "t.py", "800\tx", offset=800)
        assert stats.get("opened") is None

    def test_a_whole_small_file_in_one_read_still_counts(self):
        stats = {}
        self._read(stats, "s.py", "1\tdef a():\n2\t    pass")
        assert stats["opened"] == {"s.py"}

    def test_the_skeleton_docstring_matches_what_it_does(self):
        """It said an unreadable file "is simply absent" after the code had
        been changed to name it."""
        assert "still NAMED" in ctx.skeletons.__doc__
        assert "simply absent" not in ctx.skeletons.__doc__


class TestTheSeventhRound:
    def test_a_clean_result_on_unseen_changes_does_not_read_as_an_approval(self):
        """With every excluded file OPENED there is no unreviewed-files note to
        carry the news, so the approval prose stood alone above a COMMENT — the
        same false-clean verdict as a refused approval, by another route."""
        from agentic_review import review as pr
        body, event = pr._finalize_review([], [], head_sha="a" * 40, repo="r",
                                          excluded=[], saw_every_change=False)
        assert event == "COMMENT"
        assert "Not an approval" in body
        assert "**What this approval is.**" not in body
        assert "not seeing what the change did to it" in body or \
               "not the changes made to them" in body

    def test_a_genuine_approval_is_untouched(self):
        from agentic_review import review as pr
        body, event = pr._finalize_review([], [], head_sha="a" * 40, repo="r",
                                          excluded=[], saw_every_change=True)
        assert event == "APPROVE"
        assert "Not an approval" not in body
        assert "**What this approval is.**" in body

    def test_a_marker_inside_the_file_is_not_metadata_about_it(self):
        """Both markers are APPENDED by the tools. A file whose own content
        contains the literal text would otherwise be read as a report about
        itself — and this repository's own source contains both strings."""
        from agentic_review import agent
        stats = {}
        agent._record_opened(
            stats, "read_file", '{"path": "agent.py"}',
            '1\t_CLIPPED = "[... truncated: 12 more chars."\n'
            '2\t_PARTIAL = "[showing lines 1-2 of 9."\n3\tdone')
        assert stats["opened"] == {"agent.py"}, (
            "content that looks like a marker is not a marker")


class TestTheEighthRound:
    def test_windows_landing_in_different_passes_still_cover_the_file(self):
        """The review pass reads lines 1-200 and the revision reads the rest.
        Merging only the finished `opened` sets left that file reported as
        never opened — a false caveat on the exact files this feature is for."""
        from agentic_review import review as pr
        pr._CURRENT["opened"] = set()
        pr._CURRENT["read_ranges"] = {}
        pr._merge_opened({"_read_ranges": {
            "big.py": {"total": 255, "covered": [(1, 200)]}}})
        assert pr._CURRENT["opened"] == set(), "half a file is not the file"
        pr._merge_opened({"_read_ranges": {
            "big.py": {"total": 255, "covered": [(201, 255)]}}})
        assert pr._CURRENT["opened"] == {"big.py"}

    def test_a_gap_across_passes_is_still_a_gap(self):
        from agentic_review import review as pr
        pr._CURRENT["opened"] = set()
        pr._CURRENT["read_ranges"] = {}
        pr._merge_opened({"_read_ranges": {
            "g.py": {"total": 300, "covered": [(1, 100)]}}})
        pr._merge_opened({"_read_ranges": {
            "g.py": {"total": 300, "covered": [(201, 300)]}}})
        assert pr._CURRENT["opened"] == set()

    def test_a_content_line_that_looks_like_the_footer_is_not_the_footer(self):
        """`_read_file` prefixes every content line with `N<tab>`; the footer
        sits on a line of its own. Without the line anchor, a file whose last
        line reads `5<tab>[showing lines 1-2 of 9.]` reported itself partial."""
        from agentic_review import agent
        stats = {}
        agent._record_opened(
            stats, "read_file", '{"path": "doc.py"}',
            "1\tx\n5\t[showing lines 1-2 of 9. Call again with offset=3 for the rest.]")
        assert stats["opened"] == {"doc.py"}

    def test_the_real_footer_on_its_own_line_still_counts_as_partial(self):
        from agentic_review import agent
        stats = {}
        agent._record_opened(
            stats, "read_file", '{"path": "big.py"}',
            "1\tx\n[showing lines 1-200 of 255. Call again with offset=201 for the rest.]")
        assert stats.get("opened") is None

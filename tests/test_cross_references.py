"""Where else the repository mentions the names this diff uses.

Measured against a second reviewer on six pull requests (2026-09-03). Of the
defects only it found, three were the same shape — a name in the diff with a
second definition or a second consumer outside it:

  · a CSS class also declared in the stylesheet, so the two rules fought;
  · a payload key ignored by the handler that reads it;
  · a storage key already written by another component, splitting upgrades.

Each needed one grep the agent had and did not run. Verified afterwards
against the real checkout: all three names come out of `xref_names`, and the
files this points at are the ones holding the answers.
"""
import pytest

from agentic_review import context as ctx


class TestWhichNamesAreWorthChasing:
    @pytest.mark.parametrize("added,expected", [
        ("+  el.classList.add('prog--cpsa');", "prog--cpsa"),
        ("+.prog--cpsa {", "prog--cpsa"),
        ("+  <div class=\"wallet-card\">", "wallet-card"),
        ("+  if (reply['unspecified_keys']) return;", "unspecified_keys"),
        ("+const KEY = 'caeli_visitor_id';", "caeli_visitor_id"),
        ("+  if status == 'not_implemented':", "not_implemented"),
        ("+def adopt_native_visitor_id(x):", "adopt_native_visitor_id"),
        ("+class ProvisioningService:", "ProvisioningService"),
        ("+export function adoptNativeVisitorId() {", "adoptNativeVisitorId"),
        ("+const spendOrder = compute();", "spendOrder"),
    ])
    def test_it_finds_the_name(self, added, expected):
        assert expected in ctx.xref_names(added)

    def test_removed_and_context_lines_are_ignored(self):
        """A name the change merely walks past is not one it makes a claim
        about, and every entry costs a grep."""
        diff = ("-const KEY = 'old_visitor_id';\n"
                "   const other = 'context_only_key';\n")
        assert ctx.xref_names(diff) == []

    def test_the_diff_header_is_not_an_added_line(self):
        assert ctx.xref_names("+++ b/lib/some_module.ts\n") == []

    def test_common_words_are_not_chased(self):
        """A cross-reference for `data` matches half the repository, and noise
        here pushes the real pointer off a capped list."""
        for noise in ("+const data = 1;", "+const config = 1;", "+const state = 1;"):
            assert ctx.xref_names(noise) == []

    def test_each_name_appears_once(self):
        diff = "+a['visitor_key'] = 1;\n+b['visitor_key'] = 2;\n"
        assert ctx.xref_names(diff).count("visitor_key") == 1

    def test_order_is_first_appearance(self):
        diff = "+const alpha_one = 1;\n+const beta_two = 2;\n"
        assert ctx.xref_names(diff) == ["alpha_one", "beta_two"]


class TestTheSection:
    DIFF = "+const KEY = 'caeli_visitor_id';\n"

    def test_it_names_the_files_outside_the_diff(self):
        out = ctx.cross_references(
            "/w", self.DIFF, {"extension/lib/visitor-id.ts"},
            run=lambda w, n: ["extension/lib/visitor-id.ts", "ios/ViewController.swift"])
        assert "`caeli_visitor_id` also in: ios/ViewController.swift" in out

    def test_the_changed_files_themselves_are_not_pointers(self):
        """A name living entirely inside the change has no second consumer."""
        assert ctx.cross_references(
            "/w", self.DIFF, {"extension/lib/visitor-id.ts"},
            run=lambda w, n: ["extension/lib/visitor-id.ts"]) == ""

    def test_a_name_found_nowhere_is_not_listed(self):
        assert ctx.cross_references("/w", self.DIFF, set(), run=lambda w, n: []) == ""

    def test_it_says_an_unrelated_match_is_the_common_case(self):
        """Without that, the list reads as an accusation and the model
        manufactures a finding per row."""
        out = ctx.cross_references("/w", self.DIFF, set(),
                                   run=lambda w, n: ["other.ts"])
        assert "pointer, not an answer" in out and "unrelated match" in out

    def test_the_number_of_names_is_capped(self):
        diff = "".join(f"+const name_{i}_x = 1;\n" for i in range(40))
        out = ctx.cross_references("/w", diff, set(), run=lambda w, n: ["a.ts"])
        assert out.count("` also in:") <= ctx.MAX_XREF_NAMES

    def test_the_cap_bounds_the_SEARCHES_not_just_the_rows(self):
        """A name with no outside hit produces no row, so a row-only cap let a
        large diff spend one subprocess per extracted name. Copilot's finding
        on the PR that added this."""
        diff = "".join(f"+const name_{i}_x = 1;\n" for i in range(40))
        searched = []
        ctx.cross_references("/w", diff, set(),
                             run=lambda w, n: searched.append(n) or [])
        assert len(searched) <= ctx.MAX_XREF_NAMES

    def test_a_cut_list_says_it_was_cut(self):
        diff = "".join(f"+const name_{i}_x = 1;\n" for i in range(40))
        out = ctx.cross_references("/w", diff, set(), run=lambda w, n: ["a.ts"])
        assert "list cut for length" in out and "grep -rl" in out

    def test_dropped_files_are_counted_not_hidden(self):
        """This module's own rule: a truncation says so, or the list reads as
        exhaustive and the model stops looking."""
        out = ctx.cross_references(
            "/w", "+const KEY = 'caeli_visitor_id';\n", set(),
            run=lambda w, n: [f"f{i}.ts" for i in range(9)])
        assert "+5 more" in out and "grep -rl caeli_visitor_id" in out

    def test_the_character_budget_bounds_the_finished_section(self):
        """Checked after the row is built: a repository path can be long, and a
        budget tested only beforehand is one the last row walks past."""
        diff = "".join(f"+const name_{i}_x = 1;\n" for i in range(12))
        long_path = "a/" + "b" * 300 + ".ts"
        out = ctx.cross_references("/w", diff, set(), run=lambda w, n: [long_path])
        assert len(out) < ctx.MAX_XREF_CHARS + 600

    def test_the_files_per_name_are_capped(self):
        out = ctx.cross_references(
            "/w", self.DIFF, set(),
            run=lambda w, n: [f"f{i}.ts" for i in range(20)])
        assert out.count(".ts") <= ctx.MAX_XREF_FILES

    def test_a_grep_that_fails_costs_nothing(self):
        """Never raises: a pointer list is not worth a review."""
        def boom(w, n):
            raise OSError("git is not here")
        assert ctx.cross_references("/w", self.DIFF, set(), run=boom) == ""

    def test_an_empty_diff_is_silent(self):
        assert ctx.cross_references("/w", "", set(), run=lambda w, n: ["a.ts"]) == ""


class TestTheEvalPathUsesIt:
    def test_build_context_requires_the_diff(self):
        """It was optional, `eval/compare.py` omitted it, and the section
        silently switched itself off in the harness that measures whether it
        helps — so the first measurement of this feature measured nothing."""
        import inspect
        from agentic_review import review
        sig = inspect.signature(review.build_context)
        assert sig.parameters["diff"].default is inspect.Parameter.empty

    def test_the_eval_harness_passes_the_runtime_diff(self, monkeypatch):
        """Asserted by CALL, not by source text: a grep for the call site
        passes through any equivalent rewrite and fails on a harmless one.
        This repository's own rule, and the reviewer caught it here."""
        import json
        import eval.compare as compare
        from agentic_review import review

        seen = {}
        diff = "--- a/x\n+++ b/x\n@@\n+const the_key = 1;\n"
        monkeypatch.setattr(review, "gh", lambda *a, **k: json.dumps(
            {"head": {"sha": "a" * 40}, "title": "t", "body": "",
             "user": {"login": "u"}}))
        monkeypatch.setattr(compare, "_diff_at", lambda *a: (diff, [], 0))
        monkeypatch.setattr(review, "checkout", lambda *a: None)
        monkeypatch.setattr(review, "review_findings", lambda *a, **k: [])
        monkeypatch.setattr(review, "_revise", lambda f, w, r: (f, []))
        monkeypatch.setattr(review.checks, "run_all", lambda *a, **k: [])
        monkeypatch.setattr(review, "commit_messages", lambda *a: [])

        def spy(repo, pr, meta, work, changed, diff_arg, excluded=()):
            seen["diff"] = diff_arg
            return ""
        monkeypatch.setattr(review, "build_context", spy)
        compare.run_ours("app", 1)
        assert "the_key" in seen["diff"], (
            "the harness must hand build_context the real diff, or every "
            "diff-derived section measures itself as absent")


class TestItIsInTheContextBlock:
    def test_build_places_it_before_the_map(self):
        out = ctx.build("/w", [], tracker_section="TICKETS\n",
                        linked_section="LINKED\n", xref_section="XREF\n")
        assert out.index("XREF") > out.index("LINKED")

    def test_an_empty_section_adds_nothing(self):
        assert "XREF" not in ctx.build("/w", [], xref_section="")


class TestTheSecondRoundOfReviewFindings:
    """Six findings from the automated review of this PR, each pinned."""

    @pytest.mark.parametrize("line,expected", [
        ("+.wallet-card:hover {", "wallet-card"),
        ("+.prog--cpsa.active {", "prog--cpsa"),
        ("+.side-panel > .row {", "side-panel"),
        ("+.card::after {", "card"),
        ("+.a-list, .b-list {", "a-list"),
    ])
    def test_css_declarations_beyond_brace_and_comma(self, line, expected):
        """CSS is the primary use case and a `{`-or-`,`-only pattern missed
        most real declarations — including the class conflict this feature was
        built to catch."""
        assert expected in ctx.xref_names(line)

    def test_a_cut_is_announced_even_when_no_row_survived(self):
        """The case where it matters most: the searched names had no outside
        hit, and the ones never searched are invisible."""
        diff = "".join(f"+const name_{i}_x = 1;\n" for i in range(40))
        out = ctx.cross_references("/w", diff, set(), run=lambda w, n: [])
        assert "list cut for length" in out

    def test_a_file_the_diff_dropped_is_not_called_untouched(self):
        """`changed` comes from the diff SHOWN, which the budget may have cut.
        Without `also_changed` the prompt says the change does not touch a file
        it does touch, and the model treats an updated consumer as stale."""
        diff = "+const the_key = 1;\n"
        assert ctx.cross_references(
            "/w", diff, {"a.ts"}, run=lambda w, n: ["big.ts"],
            also_changed=["big.ts"]) == ""
        assert "big.ts" in ctx.cross_references(
            "/w", diff, {"a.ts"}, run=lambda w, n: ["big.ts"])


class TestClassNamesAssembledByConcatenation:
    """Verified against the real diff this feature was built for, not against
    a fixture I wrote to match my own regex.

    `row.className = "prog" + (on ? " prog--cpsa" : " prog--off")` hides the
    class behind two things at once: the literal carries a LEADING SPACE so it
    can be joined, and BEM's `--` is two separators in a row. Either alone made
    the class invisible, and the class in question is the one behind a defect
    this reviewer had missed three runs in a row.
    """

    @pytest.mark.parametrize("added,expected", [
        ('+  row.className = "prog" + (on ? " prog--cpsa" : " prog--off");', "prog--cpsa"),
        ('+  row.className = "prog" + (on ? " prog--cpsa" : " prog--off");', "prog--off"),
        ("+  el.className = ' wallet-card ';", "wallet-card"),
        ('+  const cls = "side--panel--open";', "side--panel--open"),
        ('+  cls = "plain_snake_key";', "plain_snake_key"),
    ])
    def test_the_name_is_still_found(self, added, expected):
        assert expected in ctx.xref_names(added)

    def test_a_bare_word_in_quotes_is_still_not_a_name(self):
        """The separator is what makes a quoted word look like a key or a
        class rather than ordinary prose. (`msg` itself IS extracted — by the
        const-definition pattern, which is a different question.)"""
        assert "hello" not in ctx.xref_names('+  const msg = "hello";')
        assert ctx.xref_names('+  return "hello";') == []


class TestEverySelectorClassAndNoGeneratedFiles:
    """The third round of review findings on this PR."""

    @pytest.mark.parametrize("line,expected", [
        ("+.a-list, .b-list {", ["a-list", "b-list"]),
        ("+.panel > .row-item {", ["panel", "row-item"]),
        ("+.card .card-title,", ["card", "card-title"]),
    ])
    def test_every_class_in_the_selector_is_chased(self, line, expected):
        """Only the leading class was taken, so a conflicting rule on the
        SECOND class stayed undiscoverable — and CSS conflicts are the primary
        use case."""
        assert ctx.xref_names(line) == expected

    @pytest.mark.parametrize("line", [
        "+  foo.barBaz(x);",
        "+  return obj.someField;",
    ])
    def test_a_method_call_is_not_a_selector(self, line):
        assert ctx.xref_names(line) == []

    @pytest.mark.parametrize("path", [
        "package-lock.json", "yarn.lock", "node_modules/x/index.js",
        "dist/bundle.min.js", "src/icons/logo.svg", "coverage/lcov.info",
    ])
    def test_generated_and_binary_hits_are_not_pointers(self, path):
        """`pr_diff` drops these keeping only a count, so their paths reach
        neither `changed` nor `excluded` — without this they would be
        presented as files the change does not touch. A cross-reference into
        a lockfile was never a useful pointer anyway."""
        assert ctx.cross_references(
            "/w", "+const the_key = 1;\n", set(),
            run=lambda w, n: [path]) == ""

    def test_a_real_file_beside_a_generated_one_still_counts(self):
        out = ctx.cross_references(
            "/w", "+const the_key = 1;\n", set(),
            run=lambda w, n: ["package-lock.json", "src/real.ts"])
        assert "src/real.ts" in out and "package-lock" not in out


class TestSkippedFilesAreKnownToBeChanged:
    """`pr_diff` drops generated and binary files. It used to keep only a
    COUNT, so a changed `uv.lock` or `.snap` was in neither `changed` nor
    `excluded` and this section advertised it as a file the PR does not touch —
    which reads as an un-updated consumer. A heuristic list of generated-looking
    paths was the first fix and it DRIFTED: `pr_diff` skips `uv.lock`,
    `bun.lockb`, `.snap`, `.webp`, `.onnx`, `.wasm` and `/.output/`, none of
    which the heuristic knew. So the real paths are carried instead.
    """

    def test_pr_diff_reports_the_paths_it_skipped(self, monkeypatch):
        from agentic_review import review as pr
        blob = ("diff --git a/uv.lock b/uv.lock\n--- a/uv.lock\n"
                "+++ b/uv.lock\n@@\n+x\n")
        monkeypatch.setattr(pr, "gh", lambda *a, **k: blob)
        _, _, skipped = pr.pr_diff("repo", 1)
        assert list(skipped) == ["uv.lock"]

    def test_the_third_value_still_counts_and_formats_as_a_number(self, monkeypatch):
        """Three call sites read it as an int, including the posted review's
        'N generated/binary files skipped'."""
        from agentic_review import review as pr
        blob = "".join(
            f"diff --git a/{n} b/{n}\n--- a/{n}\n+++ b/{n}\n@@\n+x\n"
            for n in ("a.snap", "b.webp"))
        monkeypatch.setattr(pr, "gh", lambda *a, **k: blob)
        _, _, skipped = pr.pr_diff("repo", 1)
        assert skipped == 2
        assert f"{skipped} generated/binary files skipped" == (
            "2 generated/binary files skipped")

    def test_an_empty_skip_list_is_falsy_and_zero(self, monkeypatch):
        from agentic_review import review as pr
        blob = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@\n+x\n"
        monkeypatch.setattr(pr, "gh", lambda *a, **k: blob)
        _, _, skipped = pr.pr_diff("repo", 1)
        assert not skipped and skipped == 0

    @pytest.mark.parametrize("path", ["uv.lock", "bun.lockb", "a/b.snap",
                                      "img.webp", "m.onnx", "x.wasm"])
    def test_the_paths_the_heuristic_would_have_missed(self, path):
        """Each of these is skipped by `pr_diff` and unknown to `_GENERATED`;
        passed through `also_changed`, none is called untouched."""
        assert ctx.cross_references(
            "/w", "+const the_key = 1;\n", set(),
            run=lambda w, n: [path], also_changed=[path]) == ""


class TestWhichNamesTheCapSpendsItselfOn:
    """Appearance order put `prog--cpsa` at position 26 of 55 on the PR this
    feature was built for, past a cap of 12, so the class behind the defect was
    never searched. Shape decides the order now."""

    def test_separator_names_come_first(self):
        assert ctx.rank_names(
            ["groups", "renderThing", "prog--cpsa", "byEmployer", "wire_key"]
        ) == ["prog--cpsa", "wire_key", "renderThing", "byEmployer", "groups"]

    def test_order_within_a_tier_is_preserved(self):
        """Deterministic: two runs on the same diff search the same names."""
        assert ctx.rank_names(["a-one", "b-two", "c-three"]) == [
            "a-one", "b-two", "c-three"]

    def test_the_cap_is_sized_on_measured_cost(self):
        """30 greps: 0.89s on browser-extension, 0.69s on slack-app. The
        subprocesses were never the constraint; MAX_XREF_CHARS is."""
        assert ctx.MAX_XREF_NAMES >= 24

    def test_the_row_budget_still_holds_at_the_larger_cap(self):
        diff = "".join(f"+const name_{i}_x = 1;\n" for i in range(60))
        long_path = "src/" + "d" * 120 + ".ts"
        out = ctx.cross_references("/w", diff, set(), run=lambda w, n: [long_path])
        rows = [l for l in out.splitlines() if l.startswith("- `")]
        assert sum(len(r) for r in rows) <= ctx.MAX_XREF_CHARS


class TestFoundByOurOwnReviewerOnThisPR:
    """This reviewer, run against the pull request that adds this feature."""

    def test_the_eval_path_gets_the_skipped_paths_too(self, monkeypatch):
        """Fixed in `main()` and not in the harness — the same divergence that
        once made this whole section measure itself as absent."""
        import json
        import eval.compare as compare
        from agentic_review import review

        seen = {}
        monkeypatch.setattr(review, "gh", lambda *a, **k: json.dumps(
            {"head": {"sha": "a" * 40}, "title": "t", "body": "",
             "user": {"login": "u"}}))
        monkeypatch.setattr(compare, "_diff_at",
                            lambda *a: ("--- a/x\n+++ b/x\n@@\n+x\n",
                                        ["big.ts"], review._Skipped(["uv.lock"])))
        monkeypatch.setattr(review, "checkout", lambda *a: None)
        monkeypatch.setattr(review, "review_findings", lambda *a, **k: [])
        monkeypatch.setattr(review, "_revise", lambda f, w, r: (f, []))
        monkeypatch.setattr(review.checks, "run_all", lambda *a, **k: [])
        monkeypatch.setattr(review, "commit_messages", lambda *a: [])

        def spy(repo, pr, meta, work, changed, diff, excluded=()):
            seen["paths"] = list(excluded)
            return ""
        monkeypatch.setattr(review, "build_context", spy)
        compare.run_ours("app", 1)
        assert seen["paths"] == ["big.ts", "uv.lock"]

    def test_skipped_is_not_hashable(self):
        """A mutable list that also compares equal to an int must not be a
        dict key: `_Skipped([...])` and `2` would collide or not depending on
        insertion order."""
        from agentic_review.review import _Skipped
        with pytest.raises(TypeError):
            {_Skipped(["a.lock"]): 1}

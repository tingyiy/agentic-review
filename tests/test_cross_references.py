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
        assert out.count("\n- ") <= ctx.MAX_XREF_NAMES

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


class TestItIsInTheContextBlock:
    def test_build_places_it_before_the_map(self):
        out = ctx.build("/w", [], tracker_section="TICKETS\n",
                        linked_section="LINKED\n", xref_section="XREF\n")
        assert out.index("XREF") > out.index("LINKED")

    def test_an_empty_section_adds_nothing(self):
        assert "XREF" not in ctx.build("/w", [], xref_section="")

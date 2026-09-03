"""Hunks are grown to their enclosing definition before the model sees them.

Tingyi's suggestion, 2026-09-02, after reading pr-agent's `allow_dynamic_context`
/ `max_extra_lines_before_dynamic_context = 10`. The idea is theirs; ours is
cheaper and more accurate because we already have the checkout on disk, so there
is no raw-file API call and no chance of reading a different revision than the
one under review.

It targets the measured bottleneck: a hunk carries three lines either side,
which is enough to APPLY a patch and not enough to JUDGE one, so the reviewer
spends tool calls rediscovering the function it is standing in — and tool calls
are what ran out on slack-app#380 (64 of them, transcript full, 77-char answer).
"""
import pytest

from agentic_review import context as ctx

MODULE = """\
import os


def unrelated():
    return 0


def handler(event, user):
    validate(event)
    key = event["id"]
    value = lookup(key)
    return value


def after():
    return 1
"""

DIFF = """\
diff --git a/m.py b/m.py
--- a/m.py
+++ b/m.py
@@ -10,1 +10,1 @@
-    value = lookup(key)
+    value = lookup(key, user)
"""


@pytest.fixture
def work(tmp_path):
    (tmp_path / "m.py").write_text(MODULE)
    return str(tmp_path)


class TestItReachesTheEnclosingDefinition:
    def test_the_def_line_is_included(self, work):
        out = ctx.expand_hunks(DIFF, work)
        assert "def handler(event, user):" in out, out

    def test_the_neighbouring_function_is_not(self, work):
        """Padding a fixed ten lines would drag in `unrelated`, which costs the
        same budget as the real thing and says nothing."""
        assert "def unrelated" not in ctx.expand_hunks(DIFF, work)

    def test_lines_after_the_hunk_come_too(self, work):
        assert "return value" in ctx.expand_hunks(DIFF, work)

    def test_the_change_itself_survives(self, work):
        out = ctx.expand_hunks(DIFF, work)
        assert "-    value = lookup(key)" in out
        assert "+    value = lookup(key, user)" in out

    def test_the_hunk_header_is_recounted(self, work):
        """A header whose counts do not match the body is a corrupt diff, and
        the model reads the numbers."""
        out = ctx.expand_hunks(DIFF, work)
        header = next(l for l in out.splitlines() if l.startswith("@@"))
        added = sum(1 for l in out.splitlines()
                    if l.startswith(("+", " ")) and not l.startswith("+++"))
        # new-side count in the header must equal ' ' + '+' lines in the body
        new_count = int(header.split("+")[1].split(",")[1].split()[0])
        assert new_count == added


class TestItNeverBreaksTheReview:
    def test_a_missing_file_falls_back_to_the_raw_hunk(self, tmp_path):
        """Deleted in this PR, or binary. The diff must still arrive."""
        out = ctx.expand_hunks(DIFF, str(tmp_path))
        assert "+    value = lookup(key, user)" in out

    def test_an_empty_diff_is_returned_unchanged(self, work):
        assert ctx.expand_hunks("", work) == ""

    def test_exceeding_the_budget_falls_back(self, work):
        """The un-expanded diff was already sized to fit; growing it past the
        cap would truncate the LAST files entirely, which is worse than showing
        every file with less context."""
        assert ctx.expand_hunks(DIFF, work, max_chars=10) == DIFF

    def test_a_hunk_at_the_top_of_the_file_does_not_underflow(self, tmp_path):
        (tmp_path / "t.py").write_text("x = 1\ny = 2\n")
        diff = ("--- a/t.py\n+++ b/t.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n")
        out = ctx.expand_hunks(diff, str(tmp_path))
        header = next(l for l in out.splitlines() if l.startswith("@@"))
        assert "-0" not in header and "+0" not in header, header

    def test_multiple_files_are_each_expanded(self, tmp_path):
        (tmp_path / "a.py").write_text(MODULE)
        (tmp_path / "b.py").write_text(MODULE)
        two = DIFF.replace("m.py", "a.py") + DIFF.replace("m.py", "b.py")
        out = ctx.expand_hunks(two, str(tmp_path))
        assert out.count("def handler(event, user):") == 2


class TestTheCanonicalDiffIsUntouched:
    def test_expansion_does_not_mutate_its_input(self, work):
        before = DIFF
        ctx.expand_hunks(DIFF, work)
        assert DIFF == before

    def test_the_fingerprint_is_taken_from_the_UNEXPANDED_diff(self, work):
        """Otherwise every open PR looks freshly changed the first time this
        ships, and every one of them gets re-reviewed."""
        from agentic_review import review
        assert review._diff_fp(DIFF) != review._diff_fp(
            ctx.expand_hunks(DIFF, work)), "the two differ, so the source matters"

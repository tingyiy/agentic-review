"""A finding should hand the author the change, not describe it (2026-08-24).

Reading a review and then re-deriving the edit is work the reviewer already did
and threw away. So every finding now carries `fix` — the replacement code as it
would appear in the file — rendered as a fenced block.

THE FLAG IS THE POINT, NOT THE SNIPPET. This reviewer has proposed confident,
plausible, wrong remedies: on slack-app#341 it argued the Gemini batch key should
be `generation_config` when the SDK calls it `config`, with a worked rationale.
That fix read exactly like a correct one. Requiring a snippet without requiring
the model to say whether it CHECKED would have made that failure cheaper to
commit and harder to spot. `fix_verified` is what keeps "I opened the file"
distinguishable from "I remember how this library works".

The fence language is model-written text landing in markdown the bot posts under
its own name, so it goes through an allowlist rather than into the page.
"""
import re

import pytest

from conftest import load_script


@pytest.fixture(scope="module")
def prr():
    return load_script("pr-review")


FINDING = {"file": "a.py", "line": 3, "severity": "high",
           "title": "t", "detail": "d"}


class TestTheFixBlock:
    def test_a_verified_fix_is_fenced_and_labelled_checked(self, prr):
        out = prr._fix_block({**FINDING, "fix": "x = 1", "fix_language": "python",
                              "fix_verified": True})
        assert "```python" in out
        assert "x = 1" in out
        assert any("named symbols were checked" in l for l in out)

    def test_an_unverified_fix_says_so_loudly(self, prr):
        """The case that matters. A remedy the model did not check must not read
        like one it did — that is how `generation_config` nearly landed."""
        out = prr._fix_block({**FINDING, "fix": "x = 1", "fix_verified": False})
        assert any("NOT checked" in l for l in out)
        assert not any("named symbols were checked" in l for l in out)

    def test_a_missing_flag_is_treated_as_unverified(self, prr):
        """Fail toward doubt. An omitted flag is not a claim of having checked."""
        out = prr._fix_block({**FINDING, "fix": "x = 1"})
        assert any("NOT checked" in l for l in out)

    def test_a_truthy_non_true_flag_is_still_unverified(self, prr):
        """`is True`, not truthiness — the model emits strings, and "false" is
        truthy. Reading that as verified would invert the signal."""
        for value in ("false", "no", 1, "true"):
            out = prr._fix_block({**FINDING, "fix": "x = 1", "fix_verified": value})
            assert any("NOT checked" in l for l in out), value

    def test_no_fix_renders_nothing(self, prr):
        """A finding without a remedy is still a finding. The standing rule is
        that an unverifiable fix should be omitted rather than invented, so an
        empty `fix` must not produce an empty code block."""
        assert prr._fix_block(FINDING) == []
        assert prr._fix_block({**FINDING, "fix": "   "}) == []

    def test_an_unknown_language_degrades_to_a_bare_fence(self, prr):
        """`fix_language` is model-written and lands in markdown this bot posts.
        An arbitrary string after the backticks is the model authoring markup."""
        out = prr._fix_block({**FINDING, "fix": "x", "fix_language": "'; rm -rf /"})
        assert "```" in out
        assert not any("rm -rf" in l for l in out)

    def test_model_escaped_backticks_are_unescaped_in_the_fix_too(self, prr):
        """Same defect `render` already fixes for `detail` (infra#106): the model
        escapes backticks as if writing a shell string, and GitHub renders the
        backslash literally — through the one thing a fix most needs to show."""
        out = prr._fix_block({**FINDING, "fix": r"a = \`${X}\`"})
        assert any("`${X}`" in l for l in out)
        assert not any("\\`" in l for l in out)


class TestItReachesTheReview:
    def test_the_rendered_review_carries_the_fix(self, prr):
        body = prr.render([{**FINDING, "fix": "x = 1", "fix_language": "python",
                            "fix_verified": True}], False, 0)
        assert "```python" in body and "x = 1" in body

    def test_a_finding_with_no_fix_still_renders(self, prr):
        body = prr.render([FINDING], False, 0)
        assert "**t**" in body and "```" not in body


class TestTheFenceCannotBeBrokenOut:
    """The AI review's own finding on this PR, and it was right.

    The allowlist guarded the language TOKEN and left the CONTENT open. CommonMark
    closes a fenced block on the first backtick run at least as long as the
    opening fence, so three backticks inside `fix` end the block early and
    everything after renders as LIVE MARKDOWN under this bot's name. `fix` is
    derived from a PR's diff, which its author controls, so it is reachable.
    """

    PAYLOAD = "x = 1\n```\n[click](https://attacker.invalid)\n```\nx = 2"

    def _after_closing_fence(self, out):
        fence = out[1].replace("python", "")
        body = "\n".join(out)
        marker = "\n" + fence + "\n"
        return body.split(marker)[-1] if marker in body else ""

    def test_a_triple_backtick_fix_stays_inside_the_fence(self, prr):
        out = prr._fix_block({**FINDING, "fix": self.PAYLOAD,
                              "fix_language": "python", "fix_verified": True})
        assert len(out[1].replace("python", "")) > 3, "fence was not widened"
        assert "attacker.invalid" not in self._after_closing_fence(out)

    def test_a_longer_run_widens_the_fence_further(self, prr):
        """Four backticks in the body must not be closed by a four-backtick
        fence — the rule is longest-run + 1, not a fixed bump to four."""
        out = prr._fix_block({**FINDING, "fix": "a\n````\n![i](https://attacker.invalid/x.png)\n````\nb",
                              "fix_language": "python", "fix_verified": True})
        assert len(out[1].replace("python", "")) >= 5
        assert "attacker.invalid" not in self._after_closing_fence(out)

    def test_ordinary_code_still_gets_a_plain_three_fence(self, prr):
        """No widening when nothing needs it — an inline span is not a fence."""
        out = prr._fix_block({**FINDING, "fix": "use `foo` here", "fix_verified": True})
        assert out[1] == "```"


class TestTheUnescapeIsShared:
    def test_detail_and_fix_use_the_same_helper(self, prr):
        """The review's 🔵. Two inline copies of the same transformation drift:
        the next escape the model invents gets handled in whichever site the
        fixer happened to be reading."""
        assert prr._unescape_backticks(r"a \`b\` c") == "a `b` c"
        body = prr.render([{**FINDING, "detail": r"see \`x\`",
                            "fix": r"y = \`z\`", "fix_verified": True}], False, 0)
        assert "`x`" in body and "`z`" in body
        assert "\\`" not in body


class TestClickableMarkdownIsDefanged:
    """`detail` and `title` render as RAW MARKDOWN, so a link the model writes is
    posted, live, under this bot's identity across every repo it reviews.

    Found by a human reading the rendered review on infra#110 — the reviewer
    reported a narrower version of this (fence breakout in `fix`) and, in doing
    so, emitted `[click](https://attacker.invalid)` into `detail`, which GitHub
    rendered as a real anchor. The bug it described was in a field I had just
    added; the bug it demonstrated had been shipping since the reviewer was
    built, in a field nothing fenced.
    """

    PAYLOAD = r"Concrete: fix='x = 1\n```\n[click](https://attacker.invalid)\n```\nx=2'"

    def test_an_inline_link_loses_its_click(self, prr):
        out = prr._defang_links("see [click](https://attacker.invalid) here")
        assert "[click](" not in out
        assert "click" in out and "attacker.invalid" in out

    def test_an_image_cannot_load_a_remote_asset(self, prr):
        """An image renders without a click, so it leaks a read to the attacker
        the moment anyone opens the PR."""
        out = prr._defang_links("![x](https://attacker.invalid/beacon.png)")
        assert "![" not in out

    def test_an_autolink_is_defanged(self, prr):
        assert "<https://" not in prr._defang_links("docs at <https://attacker.invalid/d>")

    def test_a_bare_url_is_backticked_not_left_plain(self, prr):
        """THE bug in the first attempt. Leaving the URL as plain text is not
        defanging: GFM autolinks bare URLs, and greedily — it turned the whole
        tail into one anchor. A code span never autolinks."""
        out = prr._defang_links("see https://attacker.invalid/x now")
        assert "`https://attacker.invalid/x`" in out

    def test_an_already_backticked_url_is_not_double_wrapped(self, prr):
        out = prr._defang_links("see `https://ok.invalid/x` now")
        assert "``" not in out

    def test_bracket_syntax_that_is_not_a_link_survives(self, prr):
        """The reason this is a blanket rewrite rather than a code-span-aware
        one: the link pattern needs a literal `](`, so ordinary subscripts are
        untouched and no span tokenizer is needed to protect them."""
        for src in ('use `dict["key"]` here', "`items[0]` and `a[b]`", "a list [1, 2, 3]"):
            assert prr._defang_links(src) == src

    def test_the_real_payload_is_neutralised(self, prr):
        out = prr._defang_links(self.PAYLOAD)
        assert "[click](" not in out

    def test_it_reaches_the_rendered_review(self, prr):
        body = prr.render([{**FINDING, "detail": "see [click](https://attacker.invalid)"}],
                          False, 0)
        assert "[click](" not in body

    def test_the_title_is_defanged_too(self, prr):
        """`title` is interpolated into the bolded heading — same raw markdown,
        same exposure, and it is the first thing a reader's eye lands on."""
        body = prr.render([{**FINDING, "title": "[click](https://attacker.invalid)"}],
                          False, 0)
        assert "[click](" not in body


class TestItNamesTheCommitItRead:
    """GitHub stamps `commit_id` at POST time, not read time (2026-08-25).

    On slack-app#348 the agent checked out a0d780b at 22:24:38, a push landed
    mid-run, and the review posted at 22:26:53 was recorded against fa9b51f.
    Every finding was correct for what it read and looked flatly wrong against
    what it was labelled — one of them asserting a fix was missing that was, by
    then, present in the file.

    The race cannot be removed; a reviewer necessarily reads a snapshot. The
    silence about WHICH snapshot can be, and that is what makes a stale review
    self-evident rather than indistinguishable from a bad one.
    """

    def test_the_footer_names_the_sha(self, prr):
        body = prr.render([FINDING], False, 0, head_sha="a0d780baa652f00d")
        assert "`a0d780b`" in body

    def test_it_is_the_short_sha_not_the_whole_thing(self, prr):
        body = prr.render([FINDING], False, 0, head_sha="a0d780baa652f00d")
        assert "a0d780baa652f00d" not in body

    def test_no_sha_means_no_claim(self, prr):
        """Absent is better than wrong. If the caller cannot say what was read,
        the footer must not imply it knows."""
        body = prr.render([FINDING], False, 0)
        assert "It read" not in body

    def test_the_rest_of_the_footer_survives(self, prr):
        body = prr.render([FINDING], False, 0, head_sha="deadbee1234")
        assert "did not run the tests" in body


class TestTheFileReferenceIsALink:
    """A review that names a file should let you open it (2026-08-26).

    Asked for directly. Reading a finding and then hunting for `foo.py:142` by
    hand is work the review already did and threw away — the same argument as
    `fix`, applied to navigation.

    ABSOLUTE, PINNED TO THE READ COMMIT. Relative links do not work here:
    measured through GitHub's own `/markdown` with `context=example-org/infra`,
    `docs/x.md` and `/docs/x.md` are both emitted VERBATIM, so on a PR page they
    resolve under `/pull/` and 404. And the sha, not a branch, for the reason the
    footer names the commit — a link to `main` points at code that no longer
    matches the finding the moment anything merges.
    """

    SHA = "bbd6f54aa652f00d"

    def _href(self, body):
        m = re.search(r"\]\((https://[^)]+)\)", body)
        return m.group(1) if m else ""

    def test_the_file_is_a_link_to_the_blob_at_the_read_commit(self, prr):
        body = prr.render([{**FINDING, "file": "cron/scripts/pr-review.py", "line": 142}],
                          False, 0, head_sha=self.SHA, repo="infra")
        assert self._href(body) == (
            f"https://github.com/example-org/infra/blob/{self.SHA}"
            "/cron/scripts/pr-review.py#L142")

    def test_the_link_text_is_still_the_path_and_line(self, prr):
        """The heading has to stay scannable. A link whose text is prose loses
        the one thing a reader greps the review for."""
        body = prr.render([{**FINDING, "file": "a/b.py", "line": 3}],
                          False, 0, head_sha=self.SHA, repo="infra")
        assert "[`a/b.py:3`](" in body

    def test_a_finding_with_no_line_links_the_file_without_an_anchor(self, prr):
        body = prr.render([{"file": "a/b.py", "severity": "high", "title": "t",
                            "detail": "d"}], False, 0, head_sha=self.SHA, repo="infra")
        assert self._href(body).endswith("/a/b.py")
        assert "#L" not in body

    @pytest.mark.parametrize("missing", [{"head_sha": ""}, {"repo": ""}])
    def test_without_a_commit_or_a_repo_it_stays_a_plain_span(self, prr, missing):
        """Fall back rather than guess. A link built from half the facts would
        point somewhere confident and wrong."""
        kwargs = {"head_sha": self.SHA, "repo": "infra", **missing}
        body = prr.render([{**FINDING, "file": "a/b.py", "line": 3}], False, 0, **kwargs)
        assert "`a/b.py:3`" in body
        assert "](" not in body

    def test_the_path_cannot_break_out_of_the_link(self, prr):
        """`file` is MODEL-WRITTEN and lands in raw markdown this bot posts under
        its own name — the same exposure `_defang_links` exists for. A `](` in
        the path would close the link early and make the rest live markdown.

        Asserts the payload sits inside an INTACT code span, not that the
        characters are absent: measured through GitHub's `/markdown`, that span
        renders as `<code>` with no anchor at all."""
        payload = "a.py](https://attacker.invalid) [x"
        where = prr._where_link("infra", payload, 3, self.SHA)
        assert not where.startswith("["), "a path this shape must not be linked"
        assert where == f"`{payload}:3`"

    def test_a_backtick_in_the_path_cannot_close_the_span(self, prr):
        """The hole the span-fallback had, inherited from the code this
        replaced. MEASURED: `a.py`](https://attacker.invalid)`:3` renders a real
        anchor, because the backtick ends the span and the rest is live
        markdown. The fence has to outgrow the longest run in the text."""
        where = prr._where_link("infra", "a.py`](https://attacker.invalid)`", 3,
                                self.SHA)
        assert where.startswith("``") and where.endswith("``")
        inner = where.strip("`")
        assert "```" not in where, "fence must be exactly one longer than the run"
        assert "attacker.invalid" in inner, "the path is shown, just inertly"

    def test_the_ordinary_case_keeps_a_plain_single_backtick_span(self, prr):
        """No widening when nothing needs it — the fallback is the common path
        for unlinkable-but-harmless paths and must stay readable."""
        assert prr._where_link("", "a b.py", 3, self.SHA) == "`a b.py:3`"

    @pytest.mark.parametrize("path", [
        "../../etc/passwd",              # traversal
        "a b.py",                        # a space ends the href early
        "a`b.py",                        # backtick closes the code span
        "https://attacker.invalid/x",    # not a path at all
    ])
    def test_a_path_it_cannot_vouch_for_is_not_linked(self, prr, path):
        body = prr.render([{**FINDING, "file": path, "line": 3}],
                          False, 0, head_sha=self.SHA, repo="infra")
        assert "](" not in body

    def test_a_leading_slash_is_not_doubled_into_the_url(self, prr):
        """The model writes repo-absolute paths both ways."""
        body = prr.render([{**FINDING, "file": "/a/b.py", "line": 3}],
                          False, 0, head_sha=self.SHA, repo="infra")
        assert f"/blob/{self.SHA}/a/b.py#L3" in self._href(body)

    def test_a_non_numeric_line_does_not_become_an_anchor(self, prr):
        """`line` is model-written too, and `#Labc` is a link to nowhere."""
        body = prr.render([{**FINDING, "file": "a/b.py", "line": "somewhere"}],
                          False, 0, head_sha=self.SHA, repo="infra")
        assert "#L" not in body
        assert self._href(body).endswith("/a/b.py")

    @pytest.mark.parametrize("line", ["٣", "²", "1٣"])
    def test_a_non_ASCII_digit_is_not_a_line_number(self, prr, line):
        """The AI review's 🔵 on this change, and it was right.

        `str.isdigit()` is True for Arabic-Indic '٣' and superscript '²'; even
        `.isdecimal()` is True for '٣'. Neither is an ASCII test. The path is
        allowlisted to ASCII and the line was not, so this was the one value
        that could reach the href unfiltered — producing `#L٣`, a link that
        resolves to nothing. Not a breakout, just the failure this function
        exists to avoid: sending the reader somewhere confident and wrong.
        """
        body = prr.render([{**FINDING, "file": "a/b.py", "line": line}],
                          False, 0, head_sha=self.SHA, repo="infra")
        assert "#L" not in body
        assert line not in body
        assert self._href(body).endswith("/a/b.py")


class TestTheApprovalNamesItsCommitToo:
    """The AI review's finding on infra#114, and it was the sharper half.

    The findings path named the commit it read; the approval path did not. That
    is backwards: an approval UNBLOCKS A MERGE, so a review silently
    re-attributed to code it never saw does its worst on the one verdict that
    carries authority. GitHub's staleness UI does not rescue it either — that
    flags a review when the head moves AFTER posting, and the mid-run case is
    stamped against the NEW head, so it renders as a current, genuine approval.
    """

    def test_an_approval_names_the_commit(self, prr):
        assert "`a0d780b`" in prr.approval_body("a0d780baa652f00d")

    def test_it_is_the_short_sha(self, prr):
        assert "a0d780baa652f00d" not in prr.approval_body("a0d780baa652f00d")

    def test_no_sha_drops_the_claim_cleanly(self, prr):
        """Absent beats wrong, and neither a literal brace nor an empty backtick
        pair is acceptable — both would reach the author."""
        body = prr.approval_body()
        assert "{head}" not in body
        assert "at ``" not in body
        assert "read the change, explored" in body

    def test_the_rest_of_the_approval_survives_either_way(self, prr):
        for body in (prr.approval_body(), prr.approval_body("deadbee1234")):
            assert "not proof the change is safe to ship" in body
            assert "did NOT run the tests" in body


class TestItSaysWhichConsumersItCouldNotCheck:
    """The reviewer has ONE repo, and that was invisible (2026-08-27).

    It is cloned into a checkout of the PR's repo and nothing else, so a field
    other services branch on is reviewed from one side only — and a clean review
    of a wire change reads exactly like a clean review of an internal one.

    Real case, slack-app#363. The PR changed how "has this program been decided"
    is represented, around `unspecified_keys`. `browser-extension` branches on
    that field in its checkout CTA — 34 references, tests pinning both the empty
    and non-empty cases — and the reviewer could not see any of it. It posted
    findings on the slack-app side and said nothing about the consumer.

    Nothing here can read another repo. This makes the gap VISIBLE, which is the
    same move as `fix_verified`, the truncation note and naming the commit read:
    state what was not checked rather than let silence read as coverage.
    """

    FINDING = {"file": "a.py", "line": 3, "severity": "medium",
               "title": "t", "detail": "d"}

    def test_the_footer_names_the_fields_that_cross(self, prr):
        body = prr.render([self.FINDING], False, 0, repo="slack-app",
                          wire_fields=["unspecified_keys", "decided"])
        assert "`unspecified_keys`" in body and "`decided`" in body
        assert "other repositories were NOT checked" in body

    def test_the_APPROVAL_carries_it_too(self, prr):
        """Where it matters most: the approval is the verdict that unblocks a
        merge, so a clean result on a change other repos consume is exactly the
        one that must not read as full coverage."""
        body = prr.approval_body("deadbee1234", repo="slack-app",
                                 wire_fields=["unspecified_keys"])
        assert "other repositories were NOT checked" in body
        assert "not proof the change is safe to ship" in body

    def test_nothing_crossing_says_nothing(self, prr):
        """A caveat on every review is a caveat nobody reads."""
        for body in (prr.render([self.FINDING], False, 0, repo="slack-app"),
                     prr.approval_body("deadbee1234", repo="slack-app")):
            assert "NOT checked" not in body
            assert "wire boundary" not in body

    def test_it_names_the_repo_it_did_read(self, prr):
        body = prr.render([self.FINDING], False, 0, repo="slack-app",
                          wire_fields=["x"])
        assert "only `slack-app`" in body

    def test_a_model_written_field_cannot_break_out(self, prr):
        """`wire_fields` is model-written and lands in raw markdown this bot
        posts under its own name — same exposure as `file` and `detail`."""
        body = prr.render([self.FINDING], False, 0, repo="slack-app",
                          wire_fields=["a`](https://attacker.invalid)`"])
        assert "](https://attacker.invalid)" not in body.replace("`](https", "X")
        assert "``" in body, "the span must widen around the backtick"


class TestWireFieldsAreTakenSafely:
    """`_wire_fields` NEVER raises: it is a footnote, and `validate_findings` is
    the one path where declining costs a whole review."""

    @pytest.mark.parametrize("parsed", [
        {"findings": []},                          # absent
        {"findings": [], "wire_fields": None},     # null
        {"findings": [], "wire_fields": "decided"},  # a string, not a list
        {"findings": [], "wire_fields": {}},       # wrong type
        {"findings": [], "wire_fields": [""]},     # empty name
        {"findings": [], "wire_fields": [None]},   # junk element
    ])
    def test_a_malformed_value_is_simply_dropped(self, prr, parsed):
        assert prr._wire_fields(parsed) == []

    def test_duplicates_collapse(self, prr):
        assert prr._wire_fields(
            {"wire_fields": ["decided", "decided", "unspecified_keys"]}
        ) == ["decided", "unspecified_keys"]

    def test_a_padded_list_is_capped(self, prr):
        """A caveat naming thirty fields is a caveat nobody reads, and the
        prompt already says an inflated list makes it worthless."""
        assert len(prr._wire_fields({"wire_fields": [f"f{i}" for i in range(40)]})) == 8

    def test_a_long_name_is_truncated(self, prr):
        assert len(prr._wire_fields({"wire_fields": ["x" * 500]})[0]) == 60

    def test_validate_findings_records_them(self, prr):
        """It rides along with the findings rather than reshaping the retry
        path, which runs `validate_findings` twice on an empty result."""
        prr.validate_findings({"findings": [], "wire_fields": ["decided"]})
        assert prr._CURRENT["wire_fields"] == ["decided"]

    def test_a_field_named_by_EITHER_pass_survives(self, prr):
        """THE bug in the first version of this change, found by the AI review.

        An approval only ever comes from the CONFIRMATION pass, and the first
        version REPLACED `_CURRENT["wire_fields"]` with that pass's answer. So
        the caveat went silent on exactly the verdict the change exists for —
        and the test here pinned the replacement as correct, asserting away the
        outcome while checking the mechanism.

        Union, because the failure direction is asymmetric: a second pass that
        forgets to repeat a field would otherwise delete the caveat without a
        trace, and a lost caveat is silent where a redundant one is only noise.
        """
        prr._CURRENT["wire_fields"] = []
        prr.validate_findings({"findings": [], "wire_fields": ["decided"]})
        prr.validate_findings({"findings": []})              # confirmation forgot
        assert prr._CURRENT["wire_fields"] == ["decided"]

    def test_the_union_still_dedupes_and_caps(self, prr):
        prr._CURRENT["wire_fields"] = []
        prr.validate_findings({"findings": [], "wire_fields": ["a", "b"]})
        prr.validate_findings({"findings": [], "wire_fields": ["b", "c"]})
        assert prr._CURRENT["wire_fields"] == ["a", "b", "c"]
        prr.validate_findings({"findings": [], "wire_fields": [f"f{i}" for i in range(20)]})
        assert len(prr._CURRENT["wire_fields"]) == 8

    def test_the_CONFIRMATION_schema_has_a_slot_for_it(self, prr):
        """The schema half of the same bug. The confirmation prompt is a
        DIFFERENT prompt with its own JSON shape; without the key the model has
        nowhere to answer, so every approval reported none."""
        assert '"wire_fields"' in prr.CONFIRM_PROMPT
        assert '"wire_fields"' in prr.PROMPT

    def test_an_APPROVAL_reached_through_the_real_path_carries_it(self, prr,
                                                                  monkeypatch):
        """End-to-end through `review_findings`' empty path, which is what the
        first version's tests never exercised — they called `validate_findings`,
        `render` and `approval_body` directly and so could not see that the
        approval path zeroed the value between them.
        """
        replies = [{"findings": [], "wire_fields": ["unspecified_keys"]},
                   {"findings": [], "checked": [{"file": "a.py", "verified": "read it"}]}]
        monkeypatch.setattr(prr, "_reply", lambda *a, **k: replies.pop(0))
        prr._CURRENT["wire_fields"] = []
        assert prr.review_findings("prompt", "/tmp") == []
        body = prr.approval_body("deadbee1234", repo="slack-app",
                                 wire_fields=prr._CURRENT["wire_fields"])
        assert "unspecified_keys" in body
        assert "other repositories were NOT checked" in body


class TestARefusedReplyIsKept:
    """A reply we refuse must survive the refusal (2026-08-28).

    clientportal-prelaunch-site#33 failed twice in a row — two independent agent
    runs, ~160s each, the same malformed shape — and afterwards it could not be
    explained, because the raise carried six key names and the text was gone.
    The leading theory, a truncated reply where only the inner finding objects
    parsed as top-level, fits the key set exactly and is STILL unproven.

    Same rule this module applies everywhere else, turned on itself:
    `fix_verified` says whether the model checked, the footer names the commit
    it read and the consumers it could not, an alert carries its artifact. A
    guard that refuses without keeping what it refused asks the next person to
    re-derive the cause from nothing.
    """

    def test_the_reply_reaches_the_JOB_LOG(self, prr, capsys):
        """The job log is where a red check on the PR takes you — asked for
        directly. The run summary is a different page."""
        prr._keep_unusable_reply('{"file":"a.py","line":1}', "no findings list", "review")
        out = capsys.readouterr().out
        assert '{"file":"a.py","line":1}' in out
        assert "verbatim" in out

    def test_it_also_renders_on_the_run_page(self, prr, tmp_path, monkeypatch):
        summary = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        assert prr._keep_unusable_reply("SOME REPLY", "because", "review") is True
        text = summary.read_text()
        assert "SOME REPLY" in text and "because" in text

    def test_outside_actions_it_still_logs_and_does_not_crash(self, prr, capsys,
                                                             monkeypatch):
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        assert prr._keep_unusable_reply("LOCAL REPLY", "because") is False
        assert "LOCAL REPLY" in capsys.readouterr().out

    def test_a_long_reply_keeps_BOTH_ends(self, prr, capsys):
        """The tail is where a truncated reply shows it stopped mid-object; the
        head says whether the wrapper was ever emitted. The middle is the
        findings, which is the least useful part when nothing could be read."""
        reply = "HEAD" + ("x" * 50_000) + "TAIL"
        prr._keep_unusable_reply(reply, "r")
        out = capsys.readouterr().out
        assert "HEAD" in out and "TAIL" in out
        assert "omitted from the middle" in out
        assert len(out) < 20_000, "kept the whole thing"

    def test_the_true_length_is_reported_on_BOTH_surfaces(self, prr, capsys,
                                                          tmp_path, monkeypatch):
        """Otherwise a truncation diagnosis reads OUR trim as the model's cutoff
        — the exact wrong conclusion, on the exact question this exists for.

        Both surfaces, because asserting only stdout let a mutation that broke
        the run-page copy pass the whole suite.
        """
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "s.md"))
        prr._keep_unusable_reply("y" * 50_000, "r")
        assert "50000 chars" in capsys.readouterr().out
        assert "50000 chars" in (tmp_path / "s.md").read_text()

    def test_a_reply_full_of_backticks_cannot_break_the_run_page_fence(self, prr,
                                                                      tmp_path,
                                                                      monkeypatch):
        """Reviews quote code, so a refused one is exactly the reply most likely
        to contain fences — the same hazard `_fix_block` already guards."""
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "s.md"))
        prr._keep_unusable_reply("a\n```\n![x](https://attacker.invalid/x.png)\n```\nb", "r")
        text = (tmp_path / "s.md").read_text()
        opening = next(l for l in text.splitlines() if set(l.strip()) == {"`"})
        assert len(opening.strip()) >= 4, "fence was not widened past the content"

    def test_it_NEVER_raises(self, prr, monkeypatch):
        """It runs while a ScanError is already in flight. Failing here would
        replace the real cause with an IO error and lose the diagnosis twice."""
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", "/nonexistent/dir/summary.md")
        assert prr._keep_unusable_reply("r", "why") is False
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", "/nonexistent/dir/summary.md")
        assert prr._keep_unusable_reply(None, Exception("boom")) is False


class TestTheAskPreservesBeforeItRaises:
    """The wiring, not just the writer — the reply used to be passed inline to
    `_usable(run_agent(...))`, so there was nothing left to keep by the time it
    raised."""

    def test_an_unusable_reply_is_kept_and_the_error_still_propagates(self, prr,
                                                                     monkeypatch,
                                                                     capsys):
        monkeypatch.setattr(prr, "run_agent", lambda *a, **k: '{"file":"a.py"}')
        monkeypatch.setattr(prr, "_agent_timeout", lambda: 30)
        with pytest.raises(prr.ReviewError):
            prr._ask("prompt", "/tmp", False)
        assert '{"file":"a.py"}' in capsys.readouterr().out

    def test_a_good_reply_is_not_logged(self, prr, monkeypatch, capsys):
        """No noise on the happy path — every successful review would otherwise
        dump its whole reply into the log."""
        monkeypatch.setattr(prr, "run_agent", lambda *a, **k: '{"findings":[]}')
        monkeypatch.setattr(prr, "_agent_timeout", lambda: 30)
        assert prr._ask("prompt", "/tmp", False) == {"findings": []}
        assert "verbatim" not in capsys.readouterr().out


class TestTheFixIsASketchNotAPatch:
    """Tingyi's call, 2026-09-02, and the numbers back it.

    A verbatim remedy needed its own pass (the answer would not fit otherwise),
    and that pass cost 274s of a 570s review on caeli-marketing#212 — three
    times the review it served — to write code for findings already decided. It
    buys nothing in exchange: this renders a plain fenced block, not a GitHub
    `suggestion`, so nobody can apply it in one click, and a verbatim fix
    carries an authority it has not earned (this reviewer once proposed
    `generation_config` for a key the SDK calls `config`, confidently).
    """

    def test_the_review_pass_asks_for_a_short_direction(self, prr):
        assert "at most two lines" in prr.PROMPT.lower()
        assert "SKETCH THE FIX" in prr.PROMPT

    def test_there_is_no_separate_fix_pass_left(self, prr):
        """It was 60% of the wall clock. Its absence is the point."""
        assert not hasattr(prr, "_add_fixes")
        assert not hasattr(prr, "FIX_PROMPT")

    def test_an_over_long_sketch_is_capped(self, prr):
        """A model that ignores "two lines" and pastes a function body
        re-creates the truncation this shape exists to avoid, and buries the
        direction in code the author must read anyway."""
        long_fix = "\n".join(f"line {i}" for i in range(20))
        out = "\n".join(prr._fix_block({**FINDING, "fix": long_fix}))
        assert "line 0" in out and "line 19" not in out
        assert "…" in out

    def test_a_two_line_sketch_is_untouched(self, prr):
        out = "\n".join(prr._fix_block(
            {**FINDING, "fix": "re-read inside the transaction, not before it"}))
        assert "re-read inside the transaction, not before it" in out
        assert "…" not in out

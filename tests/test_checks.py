"""The deterministic checks.

These are the findings that must NEVER depend on the model's mood, so every test
here pins an exact verdict rather than a shape.
"""
import pytest

from agentic_review import checks


class TestTicketInTitle:
    def test_a_title_with_a_ticket_is_silent(self):
        assert checks.ticket_in_title("SCRUM-1234 add the thing") == []

    def test_a_ticket_anywhere_in_the_title_counts(self):
        """`fix: the widget (SCRUM-9)` is a perfectly good title."""
        assert checks.ticket_in_title("fix: the widget (SCRUM-9)") == []

    def test_a_title_with_no_ticket_is_a_finding(self):
        f, = checks.ticket_in_title("add the thing")
        assert f["severity"] == "medium"
        assert "does not name a ticket" in f["title"]
        # The message must be actionable without decoding a regex.
        assert "SCRUM-1234" in f["title"]

    def test_the_finding_quotes_the_actual_title(self):
        """So the author can see what was read — a title with a stray character
        looks correct until you see it quoted."""
        f, = checks.ticket_in_title("add the thing")
        assert "'add the thing'" in f["detail"]

    def test_a_lowercase_key_does_not_count(self):
        """The tracker's ids are uppercase; accepting `scrum-1` would let a
        title through that no tracker search will ever find."""
        assert checks.ticket_in_title("scrum-1 add the thing")

    def test_a_bare_number_is_not_a_ticket(self):
        assert checks.ticket_in_title("fix 1234")

    def test_it_is_never_high(self):
        """A process defect must not outrank a real bug in the ordering."""
        f, = checks.ticket_in_title("no ticket here")
        assert f["severity"] != "high"

    def test_an_empty_title_is_a_finding_not_a_crash(self):
        assert len(checks.ticket_in_title("")) == 1
        assert len(checks.ticket_in_title(None)) == 1


class TestAgentSessionURL:
    CLAUDE_COMMIT = ("Fix the handler\n\n"
                     "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>")
    SESSION = "https://claude.ai/code/session_01EgtwG3mGGUfpvs5dLP2QZR"

    def test_an_agent_commit_with_no_link_is_a_finding(self):
        f, = checks.agent_session_url([self.CLAUDE_COMMIT])
        assert f["severity"] == "medium"
        assert "session" in f["title"]

    def test_the_link_in_the_commit_satisfies_it(self):
        assert checks.agent_session_url(
            [self.CLAUDE_COMMIT + f"\nClaude-Session: {self.SESSION}"]) == []

    def test_the_link_in_the_PR_BODY_satisfies_it(self):
        """An agent that cannot write the trailer can still be given a body.
        Requiring it in the commit itself would fail a PR whose link IS there."""
        assert checks.agent_session_url([self.CLAUDE_COMMIT],
                                        pr_body=f"see {self.SESSION}") == []

    def test_one_link_covers_a_stack_of_commits(self):
        """Requiring it per-commit would flag every fixup in a stack that
        already carries the link on its first commit."""
        assert checks.agent_session_url(
            [self.CLAUDE_COMMIT + f"\nClaude-Session: {self.SESSION}",
             self.CLAUDE_COMMIT, self.CLAUDE_COMMIT]) == []

    def test_a_human_commit_is_silent(self):
        assert checks.agent_session_url(["Fix the handler"]) == []

    def test_merely_MENTIONING_claude_is_not_attribution(self):
        """These repositories discuss Claude constantly. A reviewer that flags a
        PR for saying the word is a reviewer people turn off."""
        assert checks.agent_session_url(
            ["Add the Claude model id to the config",
             "Bump claude-sonnet-5 in the chain"]) == []

    def test_the_generated_with_footer_counts_as_attribution(self):
        assert len(checks.agent_session_url(
            ["Fix it\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)"])) == 1

    def test_the_claude_code_url_alone_is_not_a_session_link(self):
        """The trap: the standard footer already contains a claude.com URL, so a
        loose URL match would treat every agent commit as compliant and the
        check would silently never fire."""
        assert len(checks.agent_session_url(
            ["Fix it\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)"
             ])) == 1

    def test_a_BARE_session_id_is_enough(self):
        """Tingyi's call on tests#366: `session_01Egtw…` identifies the run
        exactly as well as the full URL. Rejecting it enforces a spelling, not
        a record."""
        assert checks.agent_session_url(
            [self.CLAUDE_COMMIT + "\nClaude-Session: session_01EgtwG3mGGUfpvs5dLP2QZR"]) == []
        assert checks.agent_session_url(
            [self.CLAUDE_COMMIT], pr_body="session_01EgtwG3mGGUfpvs5dLP2QZR") == []

    def test_a_labelled_local_transcript_uuid_is_accepted(self):
        """tests#366, verbatim: a terminal session has no claude.ai URL, and
        the author recorded the local transcript id rather than fabricate a
        link. That is the honest form and the check must take it."""
        body = ("**Agent session (local Claude Code CLI):** "
                "`e3ddebfd-34c7-46ee-bc04-692dedd41208`\n\n"
                "This session ran in the terminal, not on claude.ai.")
        assert checks.agent_session_url([self.CLAUDE_COMMIT], pr_body=body) == []

    def test_a_transcript_path_is_accepted(self):
        assert checks.agent_session_url(
            [self.CLAUDE_COMMIT + "\nTranscript: ~/.claude/projects/-x/"
             "e3ddebfd-34c7-46ee-bc04-692dedd41208.jsonl"]) == []

    def test_an_UNLABELLED_uuid_is_not_a_session(self):
        """A PR body is full of UUIDs that record nothing — Supabase users,
        order ids. Only one sitting next to a session/transcript label counts."""
        assert len(checks.agent_session_url(
            [self.CLAUDE_COMMIT],
            pr_body="user e3ddebfd-34c7-46ee-bc04-692dedd41208 could not sign in")) == 1

    def test_the_word_session_in_prose_is_not_an_id(self):
        assert len(checks.agent_session_url(
            [self.CLAUDE_COMMIT + "\nfixed the session handling"])) == 1

    def test_no_commits_is_silent(self):
        assert checks.agent_session_url([]) == []
        assert checks.agent_session_url([""], pr_body="") == []


class TestClaudeMdSize:
    def _repo(self, tmp_path, body, name="CLAUDE.md"):
        (tmp_path / name).write_text(body)
        return str(tmp_path)

    def test_an_oversized_file_the_diff_touches_is_a_nit(self, tmp_path):
        w = self._repo(tmp_path, "x" * 45_000)
        f, = checks.claude_md_size(w, ["CLAUDE.md"])
        assert f["severity"] == "low"
        assert "45,000 bytes" in f["title"]

    def test_a_file_the_diff_does_not_touch_is_silent(self, tmp_path):
        w = self._repo(tmp_path, "x" * 45_000)
        assert checks.claude_md_size(w, ["src/app.py"]) == []

    def test_too_many_lines_alone_is_a_finding(self, tmp_path):
        w = self._repo(tmp_path, "line\n" * 500)
        f, = checks.claude_md_size(w, ["CLAUDE.md"])
        assert "lines over 200" in f["title"]

    def test_over_the_line_target_only_does_not_claim_truncation(self, tmp_path):
        """445 lines and 30k bytes was told Claude Code 'may truncate the
        file' — a claim about the byte cap it had not crossed. The detail
        says only what fired."""
        w = self._repo(tmp_path, "line\n" * 500)
        f, = checks.claude_md_size(w, ["CLAUDE.md"])
        assert "truncate" not in f["detail"].lower()
        assert "followed less reliably" in f["detail"]

    def test_over_the_byte_cap_does_say_truncation(self, tmp_path):
        w = self._repo(tmp_path, "x" * 45_000)
        f, = checks.claude_md_size(w, ["CLAUDE.md"])
        assert "may truncate" in f["detail"]

    def test_a_stricter_self_declared_cap_is_honoured(self, tmp_path):
        w = self._repo(tmp_path, "# CLAUDE.md\n> hard cap ~20k chars\n" + "x" * 25_000)
        f, = checks.claude_md_size(w, ["CLAUDE.md"])
        assert "own stated ~20k cap" in f["title"]

    def test_over_a_self_declared_cap_but_under_40k_does_not_claim_truncation(self, tmp_path):
        """Copilot on the PR that split the wording: 25k with a stated 20k
        cap is a broken promise, not a truncation risk."""
        w = self._repo(tmp_path, "# CLAUDE.md\n> hard cap ~20k chars\n" + "x" * 25_000)
        f, = checks.claude_md_size(w, ["CLAUDE.md"])
        assert "truncate" not in f["title"].lower()
        assert "truncate the file" not in f["detail"]
        assert "cap it declares for itself" in f["detail"]


class TestRunAll:
    def test_it_returns_every_check(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("x" * 45_000)
        out = checks.run_all(str(tmp_path), ["CLAUDE.md"], title="no ticket",
                             commits=["Fix\n\nCo-Authored-By: Claude Opus 5 <x>"])
        titles = " ".join(f["title"] for f in out)
        assert "does not name a ticket" in titles
        assert "session link" in titles
        assert "bytes over" in titles

    def test_a_clean_pr_produces_nothing(self, tmp_path):
        assert checks.run_all(str(tmp_path), ["src/app.py"],
                              title="SCRUM-1 fix it", commits=["fix it"]) == []

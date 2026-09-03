"""The tool loop: containment, turn accounting, and the failure paths.

Every test here is a way the loop could produce a WRONG REVIEW rather than an
error — a tool escaping the checkout, a forced answer read as a real one, an
empty reply read as a clean bill.
"""
import json
import os

import pytest

from agentic_review import agent as agent_runner
from agentic_review import llm as cronlib
from agentic_review.errors import ReviewError


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "import os\n\n\ndef handler(event):\n    return event['id']\n")
    (tmp_path / "CLAUDE.md").write_text("# rules\nNever use os.system.\n")
    return tmp_path


# --- containment ----------------------------------------------------------

def test_absolute_path_refused(repo):
    ws = agent_runner.Workspace(str(repo))
    with pytest.raises(ValueError, match="absolute"):
        ws.resolve("/etc/passwd")


def test_dotdot_refused(repo):
    ws = agent_runner.Workspace(str(repo))
    with pytest.raises(ValueError, match="escapes"):
        ws.resolve("../../etc/passwd")


def test_symlink_out_of_tree_refused(repo, tmp_path):
    """The case a textual `..` check misses entirely: the argument is clean and
    the SYMLINK does the escaping. A checkout under review can contain one."""
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("token=hunter2\n")
    os.symlink(outside, repo / "link.txt")
    ws = agent_runner.Workspace(str(repo))
    with pytest.raises(ValueError, match="escapes"):
        ws.resolve("link.txt")


def test_sibling_prefix_directory_refused(repo, tmp_path):
    """`/tmp/work-evil` starts with `/tmp/work` as a STRING. commonpath, not
    startswith."""
    sibling = tmp_path.parent / (repo.name + "-evil")
    sibling.mkdir(exist_ok=True)
    (sibling / "x").write_text("nope")
    ws = agent_runner.Workspace(str(repo))
    with pytest.raises(ValueError, match="escapes"):
        ws.resolve(f"../{sibling.name}/x")


def test_bad_path_is_a_result_not_a_crash(repo):
    """A model reaching outside must be TOLD, not kill the review."""
    ws = agent_runner.Workspace(str(repo))
    out = agent_runner._call_tool(ws, "read_file", json.dumps({"path": "/etc/passwd"}))
    assert "absolute" in out
    assert "passwd" not in out.split("absolute")[0]


def test_unknown_tool_is_a_result(repo):
    ws = agent_runner.Workspace(str(repo))
    out = agent_runner._call_tool(ws, "write_file", "{}")
    assert "no such tool" in out and "read_file" in out


def test_malformed_arguments_are_a_result(repo):
    ws = agent_runner.Workspace(str(repo))
    assert "could not parse" in agent_runner._call_tool(ws, "grep", "{not json")


# --- tools ----------------------------------------------------------------

def test_read_file_numbers_lines(repo):
    ws = agent_runner.Workspace(str(repo))
    out = agent_runner._read_file(ws, "src/app.py")
    assert out.startswith("1\timport os")
    assert "4\tdef handler" in out


def test_read_file_offset_limit(repo):
    ws = agent_runner.Workspace(str(repo))
    out = agent_runner._read_file(ws, "src/app.py", offset=4, limit=1)
    assert out.splitlines()[0] == "4\tdef handler(event):"


class TestReadingIsWindowed:
    """A WINDOW, NOT THE WHOLE FILE. slack-app#380 measured what whole-file
    reads cost: 64 tool calls returning 187,526 chars, and the agent filled its
    transcript budget at turn 29, was forced to answer, and returned 77
    characters after reading every changed file. It drowned rather than ran out.

    SWE-agent names the principle and puts a number on it — 18.0% vs 11.0% on
    SWE-bench Lite from interface design alone: "environment feedback should be
    informative but concise… without unnecessary details".
    """

    def _big(self, repo, lines=900):
        (repo / "big.py").write_text("".join(f"line {i}\n" for i in range(1, lines + 1)))
        return agent_runner.Workspace(str(repo))

    def test_a_bare_read_returns_a_window_not_the_file(self, repo):
        ws = self._big(repo)
        out = ws and agent_runner._read_file(ws, "big.py")
        assert "200\tline 200" in out
        assert "201\tline 201" not in out

    def test_it_says_what_is_left_and_where_to_continue(self, repo):
        """A window with no footer is indistinguishable from a whole file, and a
        model that believes it has read the file stops looking — a worse failure
        than the cost of one more call."""
        ws = self._big(repo)
        out = agent_runner._read_file(ws, "big.py")
        assert "showing lines 1-200 of 900" in out
        assert "offset=201" in out

    def test_a_file_that_FITS_gets_no_footer(self, repo):
        """Otherwise every small read carries a false "there is more"."""
        ws = agent_runner.Workspace(str(repo))
        assert "showing lines" not in agent_runner._read_file(ws, "src/app.py")

    def test_an_explicit_limit_is_honoured(self, repo):
        ws = self._big(repo)
        out = agent_runner._read_file(ws, "big.py", offset=500, limit=3)
        assert "500\tline 500" in out and "503\tline 503" not in out
        assert "showing lines 500-502 of 900" in out

    def test_the_tool_description_states_the_window(self, repo):
        """The model cannot page correctly if the schema implies a whole file."""
        desc = next(t["function"]["description"] for t in agent_runner.TOOLS
                    if t["function"]["name"] == "read_file")
        assert str(agent_runner.DEFAULT_READ_LINES) in desc


def test_grep_finds_and_reports_no_match(repo):
    ws = agent_runner.Workspace(str(repo))
    assert "src/app.py" in agent_runner._grep(ws, "handler")
    assert "no matches" in agent_runner._grep(ws, "zzz_not_here_zzz")


class TestGrepReturnsTheSurroundingLines:
    """Tingyi's observation, 2026-09-02: a human — and Claude Code's own Read —
    locates a symbol and then looks at the few lines either side, rather than
    opening the file.

    A bare `path:line:text` forces a SECOND round-trip, and the second is worse
    than the first: the model has a line number and no way to know how much of
    the surrounding function matters, so it pulls a 200-line window to see five
    lines. That is how slack-app#380 accumulated 242,000 characters across 64
    calls and then had no room left to think.
    """

    def _module(self, repo):
        (repo / "m.py").write_text(
            "import os\n"
            "\n"
            "def before():\n"
            "    return 1\n"
            "\n"
            "def target(user_id):\n"
            "    return db.get(user_id)\n"
            "\n"
            "def after():\n"
            "    return 2\n")
        return agent_runner.Workspace(str(repo))

    def test_the_lines_around_a_match_come_back(self, repo):
        ws = self._module(repo)
        out = agent_runner._grep(ws, "def target")
        assert "def target" in out
        assert "return db.get(user_id)" in out, "no line AFTER the match"
        assert "def before" in out or "return 1" in out, "no line BEFORE the match"

    def test_context_is_configurable_and_bounded(self, repo):
        ws = self._module(repo)
        assert "return 2" not in agent_runner._grep(ws, "def target", context=0)
        # A model asking for 500 lines of context would defeat the windowing.
        wide = agent_runner._grep(ws, "def target", context=999)
        assert "import os" in wide

    def test_the_match_count_is_matches_not_lines(self, repo):
        """With context on, counting LINES caps at a handful of hits and tells
        the model to narrow a pattern that was already fine."""
        (repo / "many.py").write_text("".join(f"x = {i}  # hit\n" for i in range(20)))
        ws = agent_runner.Workspace(str(repo))
        out = agent_runner._grep(ws, "hit", max_matches=10)
        assert "more than 10 matches" in out
        assert out.count("# hit") >= 10

    def test_a_small_result_carries_no_narrow_hint(self, repo):
        ws = self._module(repo)
        assert "narrow the pattern" not in agent_runner._grep(ws, "def target")

    def test_the_tool_description_says_context_comes_back(self, repo):
        """The model will keep issuing a follow-up read if the schema implies a
        bare match line."""
        desc = next(t["function"]["description"] for t in agent_runner.TOOLS
                    if t["function"]["name"] == "grep")
        assert "surrounding code" in desc


def test_truncation_tells_the_model_it_truncated():
    out = agent_runner._truncate("x" * 100, limit=10)
    assert out.startswith("x" * 10)
    assert "truncated" in out and "90 more chars" in out


# --- the loop -------------------------------------------------------------

class FakeLLM:
    """A scripted `chat_with_tools`. Records what it was asked."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, tools, tool_choice="auto", **kw):
        self.calls.append({"messages": list(messages), "tool_choice": tool_choice})
        if not self.replies:
            raise AssertionError("FakeLLM ran out of replies")
        return self.replies.pop(0)


def _tool_call(name, args, cid="c1"):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


def test_loop_runs_a_tool_then_answers(repo, monkeypatch):
    llm = FakeLLM([_tool_call("read_file", {"path": "src/app.py"}),
                   {"role": "assistant", "content": '{"findings":[]}'}])
    monkeypatch.setattr(cronlib, "chat_with_tools", llm)
    monkeypatch.setattr(agent_runner.llm, "chat_with_tools", llm)
    text, transcript = agent_runner.run("sys", "user", str(repo), log=None)
    assert text == '{"findings":[]}'
    # The tool RESULT reached the model — otherwise the loop is a one-shot with
    # extra steps, which is exactly the thing being replaced.
    second = llm.calls[1]["messages"]
    assert second[-1]["role"] == "tool"
    assert "def handler" in second[-1]["content"]
    assert any("read_file" in line for line in transcript)


def test_empty_reply_raises_rather_than_reading_as_clean(repo, monkeypatch):
    """A content-free answer is the shrug that posts a formal APPROVE. It must
    be an error, not an empty findings list."""
    llm = FakeLLM([{"role": "assistant", "content": ""}])
    monkeypatch.setattr(agent_runner.llm, "chat_with_tools", llm)
    with pytest.raises(agent_runner.AgentError, match="neither content nor tool"):
        agent_runner.run("sys", "user", str(repo), log=None)


def test_turn_cap_forces_an_answer_with_tools_off(repo, monkeypatch):
    llm = FakeLLM([_tool_call("list_files", {}),
                   {"role": "assistant", "content": "done"}])
    monkeypatch.setattr(agent_runner.llm, "chat_with_tools", llm)
    text, transcript = agent_runner.run("sys", "user", str(repo), max_turns=2,
                                        log=None)
    assert text == "done"
    assert llm.calls[0]["tool_choice"] == "auto"
    assert llm.calls[1]["tool_choice"] == "none"
    assert any("turn cap" in line for line in transcript)


def test_deadline_raises_timeout_carrying_the_transcript(repo, monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(agent_runner.time, "monotonic", lambda: clock["t"])

    def slow(messages, tools, tool_choice="auto", **kw):
        clock["t"] += 500
        return _tool_call("list_files", {})

    monkeypatch.setattr(agent_runner.llm, "chat_with_tools", slow)
    with pytest.raises(agent_runner.Timeout) as e:
        agent_runner.run("sys", "user", str(repo), deadline=600, log=None)
    assert e.value.transcript, "a timeout with no transcript is the old failure"


def test_provider_error_names_the_turn(repo, monkeypatch):
    def boom(*a, **kw):
        raise ReviewError("fireworks -> 500")

    monkeypatch.setattr(agent_runner.llm, "chat_with_tools", boom)
    with pytest.raises(agent_runner.AgentError, match="turn 1: fireworks"):
        agent_runner.run("sys", "user", str(repo), log=None)


def test_transcript_budget_forces_an_answer(repo, monkeypatch):
    """A huge tool result must end the loop with an answer, not with a context
    overflow on the next request."""
    monkeypatch.setattr(agent_runner, "MAX_TRANSCRIPT_CHARS", 100)
    llm = FakeLLM([_tool_call("read_file", {"path": "src/app.py"}),
                   {"role": "assistant", "content": "answer"}])
    monkeypatch.setattr(agent_runner.llm, "chat_with_tools", llm)
    text, transcript = agent_runner.run("sys", "user" * 100, str(repo), log=None)
    assert text == "answer"
    assert any("transcript budget" in line for line in transcript)


class TestATruncatedAnswerIsRecovered:
    """Measured on slack-app#377: ten turns and 90 seconds of good exploration
    became an alert because the ANSWER ran past max_tokens. Truncation is fatal
    at the provider layer for a good reason (a tool call cut off mid-arguments
    is not recoverable), but a cut-off final answer is — and throwing the whole
    review away to report it is the worst of the available outcomes.
    """

    def test_it_retries_once_with_tools_off(self, repo, monkeypatch):
        calls = []

        def flaky(messages, tools, tool_choice="auto", **kw):
            calls.append(tool_choice)
            if len(calls) == 1:
                raise cronlib.ReviewError(
                    "fireworks x truncated at max_tokens=16384 "
                    "(finish_reason=length)")
            return {"role": "assistant", "content": "shorter answer"}

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", flaky)
        text, transcript = agent_runner.run("sys", "user", str(repo), log=None)
        assert text == "shorter answer"
        assert calls == ["auto", "none"], "the retry still offered tools"
        assert any("truncated" in line for line in transcript)

    def test_it_tells_the_model_what_to_shorten(self, repo, monkeypatch):
        """"Try again" produces the same answer. The retry has to say WHICH part
        to drop, or it spends a call to fail identically."""
        seen = []

        def flaky(messages, tools, tool_choice="auto", **kw):
            seen.append(list(messages))
            if len(seen) == 1:
                raise cronlib.ReviewError("x (finish_reason=length)")
            return {"role": "assistant", "content": "ok"}

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", flaky)
        agent_runner.run("sys", "user", str(repo), log=None)
        nudge = seen[1][-1]["content"]
        assert "cut off" in nudge and "fix" in nudge
        assert "detail" in nudge, "dropping detail leaves an unusable finding"

    def test_a_SECOND_truncation_is_reported_not_looped_on(self, repo, monkeypatch):
        """The budget is genuinely too small for this PR. That is an operator
        problem, and looping would burn the deadline discovering it."""
        calls = []

        def always(messages, tools, tool_choice="auto", **kw):
            calls.append(1)
            raise cronlib.ReviewError("x (finish_reason=length)")

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", always)
        with pytest.raises(agent_runner.AgentError, match="length"):
            agent_runner.run("sys", "user", str(repo), log=None)
        assert len(calls) == 2

    def test_a_NON_truncation_error_is_not_retried(self, repo, monkeypatch):
        """A 500 retried here would double every provider outage, and the
        failover in llm.py has already had its go."""
        calls = []

        def boom(messages, tools, tool_choice="auto", **kw):
            calls.append(1)
            raise cronlib.ReviewError("fireworks -> 500")

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", boom)
        with pytest.raises(agent_runner.AgentError):
            agent_runner.run("sys", "user", str(repo), log=None)
        assert len(calls) == 1


class TestAForcedTurnIsActuallyForced:
    """`tool_choice: "none"` is a REQUEST, not a guarantee.

    Measured on slack-app#381: the answer-now turn called `read_file` anyway and
    the loop explored for four more turns before running out of budget. Two
    causes, both fixed — the schemas were being dropped alongside the field (so
    the provider saw an ordinary chat request), and the loop honoured the call
    it got back.
    """

    def test_the_schemas_are_sent_WITH_tool_choice_none(self, repo, monkeypatch):
        """Dropping `tools` to save tokens silently drops `tool_choice` with
        it, which is the whole bug."""
        seen = {}

        def capture(messages, tools, tool_choice="auto", **kw):
            seen["tools"] = tools
            seen["choice"] = tool_choice
            return {"role": "assistant", "content": "done"}

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", capture)
        agent_runner.run("sys", "user", str(repo), max_turns=1, log=None)
        assert seen["choice"] == "none"
        assert seen["tools"], "the schemas were dropped, so the field was too"

    def test_a_tool_call_on_a_forced_turn_is_NOT_executed(self, repo, monkeypatch):
        """Honouring it restarts the exploration the force exists to end."""
        ran = []
        monkeypatch.setattr(agent_runner, "_call_tool",
                            lambda *a: ran.append(1) or "x")
        replies = [_tool_call("read_file", {"path": "src/app.py"}),
                   {"role": "assistant", "content": "answer"}]

        def fake(messages, tools, tool_choice="auto", **kw):
            return replies.pop(0)

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", fake)
        text, _ = agent_runner.run("sys", "user", str(repo), max_turns=1,
                                   log=None)
        assert text == "answer"
        assert ran == [], "the refused tool call was executed anyway"

    def test_a_model_that_NEVER_answers_is_reported_not_looped_on(
            self, repo, monkeypatch):
        """The recovery allowance is bounded. Without a cap this is an infinite
        loop against a model that only ever calls tools."""
        calls = []

        def always_tools(messages, tools, tool_choice="auto", **kw):
            calls.append(1)
            return _tool_call("read_file", {"path": "src/app.py"})

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", always_tools)
        with pytest.raises(agent_runner.AgentError, match="never answered"):
            agent_runner.run("sys", "user", str(repo), max_turns=1, log=None)
        assert len(calls) <= 4, f"looped {len(calls)} times"

    def test_a_recovery_does_not_eat_an_exploration_turn(self, repo, monkeypatch):
        """A retry that consumes the last turn falls out of the loop and reports
        "loop ended without an answer" — a worse report than the failure it was
        recovering from."""
        replies = [_tool_call("read_file", {"path": "src/app.py"}),
                   {"role": "assistant", "content": "answer"}]

        def fake(messages, tools, tool_choice="auto", **kw):
            return replies.pop(0)

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", fake)
        text, _ = agent_runner.run("sys", "user", str(repo), max_turns=1,
                                   log=None)
        assert text == "answer"

    def test_a_refusal_is_retried_and_the_history_stays_valid(self, repo, monkeypatch):
        """The assistant message must not be appended without its tool results —
        a `role:tool` reply has to attach to a call, so a kept-but-unexecuted
        call leaves the conversation malformed and 400s on the next request."""
        sent = []
        replies = [_tool_call("read_file", {"path": "src/app.py"}),
                   {"role": "assistant", "content": "final"}]

        def fake(messages, tools, tool_choice="auto", **kw):
            sent.append([m.get("role") for m in messages])
            return replies.pop(0)

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", fake)
        text, _ = agent_runner.run("sys", "user", str(repo), max_turns=1,
                                   log=None)
        assert text == "final"
        assert "tool" not in sent[-1], "a tool message with no call to attach to"
        assert "assistant" not in sent[-1], "the refused call was kept"

    def test_content_ALONGSIDE_an_ignored_call_is_taken_as_the_answer(
            self, repo, monkeypatch):
        """A model that answers and also asks for a file has answered. Throwing
        that away to ask again costs a call for nothing."""
        def fake(messages, tools, tool_choice="auto", **kw):
            reply = _tool_call("read_file", {"path": "src/app.py"})
            reply["content"] = '{"findings":[]}'
            return reply

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", fake)
        text, transcript = agent_runner.run("sys", "user", str(repo),
                                            max_turns=1, log=None)
        assert text == '{"findings":[]}'
        assert any("ignored" in line for line in transcript)


class TestAnAgentThatNeverLookedHasNotReviewed:
    """slack-app#375, and it is the worst failure shape this tool has.

    The loop was forced to answer on turn ONE by the clock, replied
    `{"findings":[]}` in 1.3s, and the confirmation pass — also forced on turn
    one — reported "confirmed clean after examining 9 file(s)". It had examined
    nothing. That cleared the evidence gate word-for-word.
    """

    def test_a_deadline_already_spent_raises_instead_of_answering(
            self, repo, monkeypatch):
        called = []

        def fake(messages, tools, tool_choice="auto", **kw):
            called.append(tool_choice)
            return {"role": "assistant", "content": '{"findings":[]}'}

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", fake)
        with pytest.raises(agent_runner.Timeout, match="not enough to read"):
            # Below REQUEST_TIMEOUT, so turn 1 would be forced.
            agent_runner.run("sys", "user", str(repo), deadline=30, log=None)
        assert called == [], "it answered from the prompt alone"

    def test_a_turn_CAP_of_one_is_still_allowed(self, repo, monkeypatch):
        """`max_turns=1` is a deliberate configuration, not a budget that ran
        out. Conflating them would break every single-turn caller."""
        monkeypatch.setattr(agent_runner.llm, "chat_with_tools",
                            lambda *a, **k: {"role": "assistant", "content": "ok"})
        text, _ = agent_runner.run("sys", "user", str(repo), max_turns=1,
                                   deadline=900, log=None)
        assert text == "ok"

    def test_a_generous_deadline_is_untouched(self, repo, monkeypatch):
        replies = [_tool_call("read_file", {"path": "src/app.py"}),
                   {"role": "assistant", "content": "answer"}]
        monkeypatch.setattr(agent_runner.llm, "chat_with_tools",
                            lambda *a, **k: replies.pop(0))
        text, _ = agent_runner.run("sys", "user", str(repo), deadline=900,
                                   log=None)
        assert text == "answer"


class TestStatsReportHowMuchLookingHappened:
    def test_tool_calls_and_turns_are_counted(self, repo, monkeypatch):
        replies = [_tool_call("read_file", {"path": "src/app.py"}),
                   _tool_call("grep", {"pattern": "handler"}, cid="c2"),
                   {"role": "assistant", "content": "done"}]
        monkeypatch.setattr(agent_runner.llm, "chat_with_tools",
                            lambda *a, **k: replies.pop(0))
        stats = {}
        agent_runner.run("sys", "user", str(repo), log=None, stats=stats)
        assert stats["tool_calls"] == 2
        assert stats["turns"] == 3

    def test_a_run_that_called_nothing_reports_zero(self, repo, monkeypatch):
        """The number the approval guard reads. It must be zero, not absent —
        an absent key would let the guard read it as "unknown" and pass."""
        monkeypatch.setattr(agent_runner.llm, "chat_with_tools",
                            lambda *a, **k: {"role": "assistant", "content": "x"})
        stats = {}
        agent_runner.run("sys", "user", str(repo), log=None, stats=stats)
        assert stats["tool_calls"] == 0


class TestTheConversationStaysCacheable:
    """Fireworks caches on an EXACT PREFIX MATCH and discounts cached prompt
    tokens by ~50%. In an agent loop every turn re-sends the whole conversation,
    so on a 40,000-character prompt the cache is most of the bill and most of
    the latency.

    The property it depends on is that earlier messages are never edited,
    reordered or removed — one changed token invalidates everything after it.
    That is easy to break by accident (trimming old tool results to save room is
    the obvious way), so it is pinned here rather than left as a comment.
    """

    def test_earlier_messages_are_never_rewritten(self, repo, monkeypatch):
        seen = []

        def capture(messages, tools, tool_choice="auto", **kw):
            seen.append([json.dumps(m, sort_keys=True) for m in messages])
            if len(seen) < 3:
                return _tool_call("read_file", {"path": "src/app.py"},
                                  cid=f"c{len(seen)}")
            return {"role": "assistant", "content": "done"}

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", capture)
        agent_runner.run("sys", "user", str(repo), log=None)
        assert len(seen) >= 3
        for earlier, later in zip(seen, seen[1:]):
            assert later[:len(earlier)] == earlier, (
                "an earlier message changed — the cached prefix is destroyed "
                "and every later turn pays full price")

    def test_the_system_prompt_is_first_and_constant(self, repo, monkeypatch):
        """The most-reused bytes must sit at the front, where the prefix match
        starts."""
        seen = []
        monkeypatch.setattr(
            agent_runner.llm, "chat_with_tools",
            lambda messages, tools, tool_choice="auto", **kw: (
                seen.append(messages[0]),
                {"role": "assistant", "content": "x"})[1])
        agent_runner.run("SYSTEM TEXT", "user", str(repo), log=None)
        assert seen[0] == {"role": "system", "content": "SYSTEM TEXT"}

    def test_a_forced_turn_appends_rather_than_replaces(self, repo, monkeypatch):
        """The "answer now" nudge is a new message, not an edit of the last one
        — editing would invalidate the prefix at the worst moment, on the
        largest conversation."""
        seen = []

        def capture(messages, tools, tool_choice="auto", **kw):
            seen.append(list(messages))
            return {"role": "assistant", "content": "done"}

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", capture)
        agent_runner.run("sys", "user", str(repo), max_turns=1, log=None)
        assert seen[0][0]["role"] == "system"
        assert seen[0][1]["content"] == "user"
        assert seen[0][-1]["role"] == "user"  # the nudge, appended


class TestTheChainOfThoughtSurvivesTheTurn:
    """Tingyi's question, 2026-09-02: does the model's thinking persist turn to
    turn, or is it re-derived every time?

    It persists, and ONLY because the assistant message is appended VERBATIM.
    Verified against the live API: with reasoning on, Fireworks returns the
    thinking in `reasoning_content` (`content` empty), accepts it echoed back,
    and continues from it. With reasoning off — our setting — the thinking lands
    in `content` instead. Either way the field rides along untouched.

    Which makes any RECONSTRUCTION of that message a silent chain-of-thought
    leak. There was one, in the path that refuses a tool call on a forced turn.
    """

    def test_the_assistant_message_is_appended_verbatim(self, repo, monkeypatch):
        seen = []
        replies = [{"role": "assistant", "content": "thinking out loud",
                    "reasoning_content": "the real chain", "tool_calls": [
                        {"id": "c1", "type": "function", "function": {
                            "name": "list_files", "arguments": "{}"}}]},
                   {"role": "assistant", "content": "done"}]

        def capture(messages, tools, tool_choice="auto", **kw):
            seen.append(list(messages))
            return replies.pop(0)

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", capture)
        agent_runner.run("sys", "user", str(repo), log=None)
        echoed = seen[1][2]
        assert echoed.get("reasoning_content") == "the real chain", (
            "the chain of thought was dropped between turns")

    def test_a_refused_tool_call_keeps_everything_but_the_calls(self, repo,
                                                                monkeypatch):
        """The leak that existed. `tool_calls` must go — a `role:tool` reply has
        to attach to a call — but rebuilding from `content` alone throws away
        `reasoning_content` too, and this is the list the revision pass resumes
        from."""
        stats = {}

        def fake(messages, tools, tool_choice="auto", **kw):
            return {"role": "assistant", "content": "answered anyway",
                    "reasoning_content": "why I answered",
                    "tool_calls": [{"id": "c1", "type": "function",
                                    "function": {"name": "list_files",
                                                 "arguments": "{}"}}]}

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", fake)
        agent_runner.run("sys", "user", str(repo), max_turns=1, log=None,
                         stats=stats)
        last = stats["messages"][-1]
        assert last["content"] == "answered anyway"
        assert last.get("reasoning_content") == "why I answered"
        assert "tool_calls" not in last, "a call with no result would 400"


class TestTheHarnessRunsTheRealPipeline:
    def test_compare_calls_the_same_passes_as_main(self, monkeypatch, tmp_path):
        """The eval harness bypasses main() so it can never post — but that
        means it re-lists the passes by hand, and on 2026-09-02 it still called
        `_reflect` an hour after `_revise` replaced it. Three precision runs died
        with AttributeError. A harness that runs a different reviewer than
        production measures nothing.

        Asserted by ORDER OF CALL, not by reading the harness's source — a
        source-text assertion is the pattern this workspace has been bitten by
        three times."""
        import json as _json
        from agentic_review import review
        from eval import compare
        order = []
        monkeypatch.setattr(review, "gh", lambda *a, **k: _json.dumps(
            {"head": {"sha": "a" * 40}, "base": {"sha": "b" * 40},
             "title": "SCRUM-1 x", "body": ""}))
        monkeypatch.setattr(review, "pr_diff",
                            lambda *a: ("--- a/x\n+++ b/x\n@@\n", False, 0))
        monkeypatch.setattr(review, "checkout", lambda *a: None)
        monkeypatch.setattr(review, "build_context", lambda *a: "")
        monkeypatch.setattr(review, "commit_messages", lambda *a: [])
        monkeypatch.setattr(review, "review_findings",
                            lambda *a, **k: order.append("agent") or [])
        monkeypatch.setattr(review, "_revise",
                            lambda f, w, r: (order.append("revise"), (f, []))[1])
        monkeypatch.setattr(review.checks, "run_all",
                            lambda *a, **k: order.append("checks") or [])
        compare.run_ours("repo", 1)
        assert order == ["agent", "revise", "checks"]
        for gone in ("_reflect", "_second_look", "_add_fixes"):
            assert not hasattr(review, gone)


class TestAResumedPassGetsItsOwnRoom:
    """caeli-marketing#212 at a 240k budget: the review pass was forced by the
    transcript budget at turn 19, so its conversation was AT the cap — and the
    revision that resumed it was forced on turn 1 with zero tool calls. It could
    neither check a drop nor add much. The old separate pass never had this
    problem because it started from empty."""

    def _big_conversation(self, chars):
        return [{"role": "system", "content": "s"},
                {"role": "user", "content": "x" * chars},
                {"role": "assistant", "content": "answer"}]

    def test_resume_is_not_forced_on_turn_one_by_the_inherited_size(
            self, repo, monkeypatch):
        monkeypatch.setattr(agent_runner, "MAX_TRANSCRIPT_CHARS", 1000)
        monkeypatch.setattr(agent_runner, "RESUME_HEADROOM", 5000)
        seen = []
        replies = [_tool_call("list_files", {}),
                   {"role": "assistant", "content": "done"}]

        def fake(messages, tools, tool_choice="auto", **kw):
            seen.append(tool_choice)
            return replies.pop(0)

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", fake)
        stats = {}
        agent_runner.resume(self._big_conversation(3000), "revise", str(repo),
                            log=None, stats=stats)
        assert seen[0] == "auto", "forced on turn 1 — no room to look"
        assert stats["tool_calls"] == 1

    def test_the_headroom_is_still_a_cap(self, repo, monkeypatch):
        """Room, not infinite rope: once the resumed pass has spent its
        allowance on tool results, it is forced like any other."""
        monkeypatch.setattr(agent_runner, "MAX_TRANSCRIPT_CHARS", 1000)
        # Smaller than any read result, so the first tool result spends it.
        monkeypatch.setattr(agent_runner, "RESUME_HEADROOM", 20)
        seen = []
        replies = [_tool_call("read_file", {"path": "src/app.py"}),
                   {"role": "assistant", "content": "done"}]

        def fake(messages, tools, tool_choice="auto", **kw):
            seen.append(tool_choice)
            return replies.pop(0)

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", fake)
        agent_runner.resume(self._big_conversation(3000), "revise", str(repo),
                            log=None)
        # turn 1 had room (the question is small); the tool result spent it.
        assert seen == ["auto", "none"]

    def test_a_fresh_run_still_uses_the_global_budget(self, repo, monkeypatch):
        monkeypatch.setattr(agent_runner, "MAX_TRANSCRIPT_CHARS", 10)
        seen = []
        monkeypatch.setattr(agent_runner.llm, "chat_with_tools",
                            lambda m, t, tool_choice="auto", **kw: (
                                seen.append(tool_choice),
                                {"role": "assistant", "content": "done"})[1])
        agent_runner.run("sys", "u" * 50, str(repo), log=None)
        assert seen[0] == "none"


class TestTheEchoSurvivesAProviderSwitch:
    """infra#155, 2026-09-02, the first time failover fired MID-LOOP live: turn
    1 timed out on Fireworks and failed over to OpenRouter, whose reply carried
    `refusal: null` and `reasoning: null`. Echoed verbatim, turn 2 went back to
    Fireworks with those keys in `messages[2]` and was refused — "Extra inputs
    are not permitted". The review died on the path that exists to save it.
    """

    def test_provider_decoration_and_nulls_are_not_echoed(self, repo, monkeypatch):
        seen = []
        replies = [{"role": "assistant", "content": "", "refusal": None,
                    "reasoning": None, "tool_calls": [
                        {"id": "c1", "type": "function", "function": {
                            "name": "list_files", "arguments": "{}"}}]},
                   {"role": "assistant", "content": "done"}]

        def capture(messages, tools, tool_choice="auto", **kw):
            seen.append(list(messages))
            return replies.pop(0)

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", capture)
        agent_runner.run("sys", "user", str(repo), log=None)
        echoed = seen[1][2]
        assert "refusal" not in echoed and "reasoning" not in echoed
        assert echoed["tool_calls"], "the calls the tool replies attach to"

    def test_reasoning_content_still_rides_along(self, repo, monkeypatch):
        """The chain of thought is the one extra field that MUST survive."""
        seen = []
        replies = [{"role": "assistant", "content": "", "reasoning_content": "why",
                    "refusal": None, "tool_calls": [
                        {"id": "c1", "type": "function", "function": {
                            "name": "list_files", "arguments": "{}"}}]},
                   {"role": "assistant", "content": "done"}]

        def capture(messages, tools, tool_choice="auto", **kw):
            seen.append(list(messages))
            return replies.pop(0)

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", capture)
        agent_runner.run("sys", "user", str(repo), log=None)
        assert seen[1][2].get("reasoning_content") == "why"

    def test_the_resumed_conversation_is_clean_too(self, repo, monkeypatch):
        """The revision pass resumes `stats["messages"]`; a decorated final
        message there would 400 the very next call."""
        monkeypatch.setattr(agent_runner.llm, "chat_with_tools",
                            lambda *a, **k: {"role": "assistant", "content": "x",
                                             "refusal": None, "reasoning": None})
        stats = {}
        agent_runner.run("sys", "user", str(repo), log=None, stats=stats)
        last = stats["messages"][-1]
        assert "refusal" not in last and "reasoning" not in last

    def test_a_missing_content_is_sent_as_empty_string(self):
        out = agent_runner._for_echo({"role": "assistant", "tool_calls": []})
        assert out["content"] == ""

"""SCRUM-1243: a reply that repeats itself is cut in seconds, not at the budget.

Captured on caeli-marketing#268, 2026-09-07: with reasoning off the model
narrates in `content` ("Now let me look at…") and on a long transcript that
narration cycles — 62,451 chars, 433 lines of which 12 were distinct, one
sentence pair 136 times, 83 seconds, the whole 16,384-token budget. The retry
then answered in 4 seconds. Half of 28 recent runs across seven repos had at
least one.
"""
import json

import pytest

from agentic_review import agent as agent_runner
from agentic_review import llm
from agentic_review import review


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def handler(event):\n    return 1\n")
    return tmp_path


CYCLE = ("Let me now look at the `brandSells` function's output for a brand "
         "with categories. The output is \"A, B and C from Brand.\"\n\n"
         "Let me now look at the `brandSells` function's interaction with the "
         "`brandCategories` function. The `brandCategories` function returns "
         "chips with `label` from `categoryLabel`.\n\n")


def _sse(chunks, done=True):
    """Bytes lines the way the provider sends them: one `data:` per chunk."""
    lines = [f"data: {json.dumps(c)}\n".encode() for c in chunks]
    if done:
        lines.append(b"data: [DONE]\n")
    return lines


def _content(text):
    return {"choices": [{"index": 0, "delta": {"content": text},
                         "finish_reason": None}]}


class _Stream:
    """A fake `urlopen` response: a context manager that yields lines lazily
    and counts how many were consumed, so a test can prove the read stopped."""

    def __init__(self, lines, content_type="text/event-stream"):
        self.lines = lines
        self.consumed = 0
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        for line in self.lines:
            self.consumed += 1
            yield line

    def read(self):
        return b"".join(self.lines)


def _serve(stream, monkeypatch):
    monkeypatch.setattr(llm.urllib.request, "urlopen",
                        lambda req, timeout=None: stream)


class TestRepeats:
    def test_short_text_is_never_judged(self):
        assert len(CYCLE * 2) < llm.LOOP_MIN_CHARS
        assert llm.repeats(CYCLE * 2) == 0

    def test_a_cycle_counts_its_own_tail(self):
        text = CYCLE * 12
        assert len(text) >= llm.LOOP_MIN_CHARS
        assert llm.repeats(text) >= llm.LOOP_REPEATS

    def test_prose_that_does_not_cycle_counts_once(self):
        text = "".join(f"Paragraph {i} says something different about file{i}.py "
                       f"and line {i * 7}.\n" for i in range(200))
        assert len(text) >= llm.LOOP_MIN_CHARS
        assert llm.repeats(text) == 1


class TestTheStreamIsReassembled:
    """Nothing above `_post` may notice that the wire changed."""

    def test_content_tool_calls_reasoning_and_usage(self, monkeypatch):
        llm.reset_usage()
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant"},
                          "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"reasoning_content": "hm "},
                          "finish_reason": None}]},
            _content("Look"), _content("ing."),
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "read_file", "arguments": ""}}]},
                "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "{\"path\": "}}]},
                "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "\"a.py\"}"}}]},
                "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 280, "completion_tokens": 40,
                                      "prompt_tokens_details": {"cached_tokens": 100}}},
        ]
        _serve(_Stream(_sse(chunks)), monkeypatch)
        choice = llm._post("https://x", "k", {"messages": []}, 5, "fireworks", "m")
        msg = choice["message"]
        assert msg["content"] == "Looking."
        assert msg["reasoning_content"] == "hm "
        assert choice["finish_reason"] == "tool_calls"
        call = msg["tool_calls"][0]
        assert call["id"] == "call_1"
        assert call["function"]["name"] == "read_file"
        assert json.loads(call["function"]["arguments"]) == {"path": "a.py"}
        assert llm.USAGE["calls"] == 1
        assert llm.USAGE["prompt"] == 280
        assert llm.USAGE["cached"] == 100
        assert llm.USAGE["completion"] == 40
        assert llm.USAGE["estimated"] == 0

    def test_the_request_asks_for_a_stream_with_usage(self, monkeypatch):
        sent = {}

        def urlopen(req, timeout=None):
            sent.update(json.loads(req.data.decode()))
            return _Stream(_sse([_content("ok")]))

        monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
        llm._post("https://x", "k", {"messages": [], "max_tokens": 5}, 5, "fw", "m")
        assert sent["stream"] is True
        assert sent["stream_options"] == {"include_usage": True}
        assert sent["max_tokens"] == 5, "the caller's payload was not kept"

    def test_a_provider_that_ignores_stream_still_works(self, monkeypatch):
        body = json.dumps({"choices": [{"message": {"content": "hi"},
                                        "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        _serve(_Stream([body.encode()], content_type="application/json"),
               monkeypatch)
        choice = llm._post("https://x", "k", {"messages": []}, 5, "fw", "m")
        assert choice["message"]["content"] == "hi"

    def test_sse_comments_and_keepalives_are_ignored(self, monkeypatch):
        lines = [b": OPENROUTER PROCESSING\n", b"\n"] + _sse([_content("ok")])
        _serve(_Stream(lines), monkeypatch)
        choice = llm._post("https://x", "k", {"messages": []}, 5, "openrouter", "m")
        assert choice["message"]["content"] == "ok"


class TestSamplingReachesTheWire:
    def test_chat_with_tools_forwards_sampling_fields(self, monkeypatch):
        sent = {}

        def post(url, key, payload, timeout, provider, model):
            sent.update(payload)
            return {"message": {"content": "ok"}, "finish_reason": "stop"}

        monkeypatch.setattr(llm, "_post", post)
        monkeypatch.setattr(llm.env, "get", lambda k: "key")
        llm.chat_with_tools([], [], sampling={"repetition_penalty": 1.1})
        assert sent["repetition_penalty"] == 1.1
        assert sent["temperature"] == 0.2, "the review's own settings stay"


class TestACyclingReplyIsCutEarly:
    def test_it_raises_looping_before_the_stream_ends(self, monkeypatch):
        llm.reset_usage()
        # 300 chunks of the cycle: a 16k-token reply's worth if read to the end.
        stream = _Stream(_sse([_content(CYCLE) for _ in range(300)]))
        _serve(stream, monkeypatch)
        with pytest.raises(llm.Looping) as e:
            llm._post("https://x", "k", {"messages": [{"role": "user",
                                                      "content": "x" * 350}]},
                      5, "fireworks", "m")
        assert e.value.repeats >= llm.LOOP_REPEATS
        assert stream.consumed < 40, (
            f"read {stream.consumed} of 301 lines — the cut was not early")
        assert "repeating itself" in str(e.value)

    def test_the_cut_call_is_still_billed_and_says_it_was_estimated(self, monkeypatch):
        llm.reset_usage()
        _serve(_Stream(_sse([_content(CYCLE) for _ in range(300)])), monkeypatch)
        with pytest.raises(llm.Looping):
            llm._post("https://x", "k", {"messages": [{"role": "user",
                                                      "content": "x" * 3500}]},
                      5, "fireworks", "m")
        assert llm.USAGE["calls"] == 1
        assert llm.USAGE["estimated"] == 1
        assert llm.USAGE["completion"] > 0, "the cycle's tokens were generated and are paid for"
        assert llm.USAGE["prompt"] >= 1000, "the prompt was sent and is paid for"
        assert "estimated" in llm.usage_line()

    def test_the_cut_call_is_cached_at_the_runs_measured_rate(self, monkeypatch):
        """A cut turn's prompt is the same conversation as the turn before it;
        priced as fresh, the guard read as a bill increase."""
        llm.reset_usage()
        llm._record("fw", "m", {"usage": {"prompt_tokens": 1000, "completion_tokens": 10,
                                          "prompt_tokens_details": {"cached_tokens": 800}}}, {})
        _serve(_Stream(_sse([_content(CYCLE) for _ in range(300)])), monkeypatch)
        with pytest.raises(llm.Looping):
            llm._post("https://x", "k", {"messages": [{"role": "user",
                                                      "content": "x" * 35000}]},
                      5, "fireworks", "m")
        added = llm.USAGE["prompt"] - 1000
        assert added >= 9000
        assert abs((llm.USAGE["cached"] - 800) / added - 0.8) < 0.01

    def test_a_long_genuine_answer_is_not_cut(self, monkeypatch):
        """Findings JSON repeats its KEYS on every entry; that is not a cycle."""
        findings = [{"title": f"Finding {i} about file{i}.py",
                     "detail": f"Line {i * 3} of file{i}.py does thing {i} "
                               f"which is wrong because of reason {i}.",
                     "file": f"file{i}.py", "line": i * 3, "severity": "medium",
                     "fix": f"change {i} to {i + 1}"} for i in range(40)]
        text = json.dumps({"findings": findings}, indent=1)
        assert len(text) > 5000
        chunks = [_content(text[i:i + 100]) for i in range(0, len(text), 100)]
        _serve(_Stream(_sse(chunks)), monkeypatch)
        choice = llm._post("https://x", "k", {"messages": []}, 5, "fw", "m")
        assert json.loads(choice["message"]["content"])["findings"] == findings


class TestTheAgentReasksWithoutSayingShorter:
    """"Shorter" tells the model to drop findings. After a cycle there was
    nothing long to shorten — the turn should have been a tool call or the
    answer, and that is what it is asked for."""

    def test_a_loop_is_reasked_with_tools_still_on(self, repo, monkeypatch):
        calls = []

        def flaky(messages, tools, tool_choice="auto", **kw):
            calls.append((tool_choice, messages[-1]["content"]))
            if len(calls) == 1:
                raise llm.Looping("fw", "m", chars=3000, repeats=9, elapsed=4.2)
            return {"role": "assistant", "content": '{"findings":[]}'}

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", flaky)
        stats = {}
        text, transcript = agent_runner.run("sys", "user", str(repo), log=None,
                                            stats=stats)
        assert text == '{"findings":[]}'
        assert calls[1][0] == "auto", "a loop on an exploration turn is not the answer turn"
        nudge = calls[1][1]
        assert "repeated" in nudge and "think aloud" in nudge
        assert "shorter" not in nudge.lower(), "that instruction drops findings"
        assert "confident" not in nudge.lower()
        assert stats["loops"] == 1
        assert not stats.get("shortened"), "nothing was shortened"
        assert any("repeated itself" in line for line in transcript)

    def test_the_retry_samples_differently_and_only_the_retry(self, repo, monkeypatch):
        """Replayed: at the review's settings 5 of 8 retries cycled again;
        with repetition_penalty 1.1, 0 of 8. The turn after the retry is an
        ordinary turn and must not inherit it."""
        seen = []

        def flaky(messages, tools, tool_choice="auto", **kw):
            seen.append(kw.get("sampling"))
            if len(seen) == 1:
                raise llm.Looping("fw", "m", chars=3000, repeats=9, elapsed=4.2)
            if len(seen) == 2:
                return {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {
                        "name": "read_file", "arguments": json.dumps({"path": "src/app.py"})}}]}
            return {"role": "assistant", "content": "answer"}

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", flaky)
        agent_runner.run("sys", "user", str(repo), log=None)
        assert seen == [None, agent_runner.LOOP_RETRY_SAMPLING, None]
        assert "repetition_penalty" in agent_runner.LOOP_RETRY_SAMPLING

    def test_a_loop_on_the_answer_turn_stays_forced(self, repo, monkeypatch):
        calls = []

        def flaky(messages, tools, tool_choice="auto", **kw):
            calls.append(tool_choice)
            if len(calls) == 1:
                raise llm.Looping("fw", "m", chars=3000, repeats=9, elapsed=4.2)
            return {"role": "assistant", "content": "answer"}

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", flaky)
        text, _ = agent_runner.run("sys", "user", str(repo), log=None, max_turns=1)
        assert text == "answer"
        assert calls == ["none", "none"], "the re-ask offered tools on a forced turn"

    def test_a_second_loop_forces_the_answer(self, repo, monkeypatch):
        """caeli-marketing#268 on the first cut of this guard: re-asked with
        tools on, the model made one real tool call, cycled again, cycled a
        third time, and a 25-file review posted nothing."""
        calls = []

        def flaky(messages, tools, tool_choice="auto", **kw):
            calls.append((tool_choice, messages[-1]["content"]))
            if len(calls) <= 2:
                raise llm.Looping("fw", "m", chars=4500, repeats=5, elapsed=15)
            return {"role": "assistant", "content": "answer"}

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", flaky)
        stats = {}
        text, _ = agent_runner.run("sys", "user", str(repo), log=None, stats=stats)
        assert text == "answer"
        assert [c[0] for c in calls] == ["auto", "auto", "none"], (
            "the first re-ask keeps the tools; the second asks for the answer")
        assert "already read" in calls[2][1]
        assert stats["loops"] == 2

    def test_a_third_loop_is_reported_not_looped_on(self, repo, monkeypatch):
        calls = []

        def always(messages, tools, tool_choice="auto", **kw):
            calls.append(1)
            raise llm.Looping("fw", "m", chars=3000, repeats=9, elapsed=4.2)

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", always)
        with pytest.raises(agent_runner.AgentError, match="repeating itself"):
            agent_runner.run("sys", "user", str(repo), log=None)
        assert len(calls) == 3, "one attempt plus the two recovery turns the loop allows"

    def test_a_truncation_marks_the_answer_shortened(self, repo, monkeypatch):
        calls = []

        def flaky(messages, tools, tool_choice="auto", **kw):
            calls.append(1)
            if len(calls) == 1:
                raise agent_runner.ReviewError("x (finish_reason=length)")
            return {"role": "assistant", "content": "short"}

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", flaky)
        stats = {}
        agent_runner.run("sys", "user", str(repo), log=None, stats=stats)
        assert stats["shortened"] is True


class TestTheReviewSaysWhenItWasShortened:
    FINDING = {"title": "t", "detail": "d", "file": "a.py", "line": 1,
               "severity": "medium"}

    def test_the_footer_names_it(self, monkeypatch):
        monkeypatch.setitem(review._CURRENT, "answer_shortened", True)
        body = review.render([dict(self.FINDING)], False, 0, head_sha="abc1234")
        assert "may be incomplete" in body
        assert "overran its output budget" in body

    def test_and_is_silent_otherwise(self, monkeypatch):
        monkeypatch.setitem(review._CURRENT, "answer_shortened", False)
        body = review.render([dict(self.FINDING)], False, 0, head_sha="abc1234")
        assert "may be incomplete" not in body


class TestAForcedAnswerIsHeldToASchema:
    """Replayed on a captured forced turn (caeli-marketing#268): without a
    schema, usable JSON once in 24 samples; with one, 12 of 12."""

    def test_the_review_pass_passes_the_schema_for_forced_turns_only(self, monkeypatch, tmp_path):
        seen = {}

        def run(system, user, root, **kw):
            seen.update(kw)
            return '{"findings":[]}', []

        monkeypatch.setattr(review.agent, "run", run)
        review._run_agent("prompt", str(tmp_path))
        assert seen["answer_schema"] is review.ANSWER_SCHEMA
        assert seen["schema_reask"] is False

    def test_the_schema_admits_what_both_passes_answer(self):
        props = review.ANSWER_SCHEMA["json_schema"]["schema"]["properties"]
        assert set(props) == {"wire_fields", "findings", "checked"}
        assert review.ANSWER_SCHEMA["json_schema"]["schema"]["required"] == ["findings"]
        finding = props["findings"]["items"]
        assert set(finding["required"]) == {"file", "line", "severity", "title", "detail"}
        assert "fix" in finding["properties"] and "fix_verified" in finding["properties"]

    def test_a_forced_turn_carries_the_schema_and_a_free_answer_is_not_reasked(self, repo, monkeypatch):
        calls = []
        schema = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}

        def fake(messages, tools, tool_choice="auto", response_format=None, **kw):
            calls.append((tool_choice, response_format))
            return {"role": "assistant", "content": '{"findings":[]}'}

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", fake)
        agent_runner.run("sys", "user", str(repo), log=None, answer_schema=schema,
                         schema_reask=False)
        assert calls == [("auto", None)], "a free answer was re-asked"
        calls.clear()
        agent_runner.run("sys", "user", str(repo), log=None, answer_schema=schema,
                         schema_reask=False, max_turns=1)
        assert calls == [("none", schema)], "the forced turn did not carry the schema"

    def test_the_revision_keeps_its_reask(self, repo, monkeypatch):
        calls = []
        schema = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}

        def fake(messages, tools, tool_choice="auto", response_format=None, **kw):
            calls.append((tool_choice, response_format))
            return {"role": "assistant", "content": "free text"}

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", fake)
        agent_runner.run("sys", "user", str(repo), log=None, answer_schema=schema)
        assert calls == [("auto", None), ("none", schema)]


class TestNarrationIsCutAtTheProseCap:
    """The first live run of the tail check cut its four loops at 19k-48k
    chars and 30-83s: a cycle with a 7k-char period recurs four times only
    after 30k. Prose that long on an agent turn is never used."""

    PARA = ("Now let me consider what the {i}th helper does with its input and "
            "whether the caller at line {j} guards the empty case before it.\n")

    def _prose(self, n):
        return "".join(self.PARA.format(i=i, j=i * 13) for i in range(n))

    def test_long_prose_on_an_agent_turn_is_cut(self, monkeypatch):
        text = self._prose(200)
        assert len(text) > llm.PROSE_CAP * 2 and llm.repeats(text) == 1, "not a cycle"
        chunks = [_content(text[i:i + 100]) for i in range(0, len(text), 100)]
        stream = _Stream(_sse(chunks))
        _serve(stream, monkeypatch)
        with pytest.raises(llm.Looping) as e:
            llm._post("https://x", "k", {"messages": [], "tools": [{}]}, 5, "fw", "m")
        assert e.value.repeats == 0
        assert "narrating" in str(e.value)
        assert llm.PROSE_CAP <= e.value.chars < llm.PROSE_CAP + 1000
        assert stream.consumed < len(chunks)

    def test_long_json_is_not(self, monkeypatch):
        findings = [{"title": f"Finding {i}", "detail": f"Line {i * 3} of file{i}.py "
                     f"does thing {i} which is wrong because of reason {i}.",
                     "file": f"file{i}.py", "line": i * 3, "severity": "low"}
                    for i in range(60)]
        text = json.dumps({"findings": findings}, indent=1)
        assert len(text) > llm.PROSE_CAP
        chunks = [_content(text[i:i + 100]) for i in range(0, len(text), 100)]
        _serve(_Stream(_sse(chunks)), monkeypatch)
        choice = llm._post("https://x", "k", {"messages": [], "tools": [{}]}, 5, "fw", "m")
        assert json.loads(choice["message"]["content"])["findings"] == findings

    def test_a_plain_chat_may_write_prose(self, monkeypatch):
        text = self._prose(200)
        chunks = [_content(text[i:i + 100]) for i in range(0, len(text), 100)]
        _serve(_Stream(_sse(chunks)), monkeypatch)
        choice = llm._post("https://x", "k", {"messages": []}, 5, "fw", "m")
        assert choice["message"]["content"] == text

    def test_the_agent_names_the_shape(self, repo, monkeypatch):
        calls = []

        def flaky(messages, tools, tool_choice="auto", **kw):
            calls.append(1)
            if len(calls) == 1:
                raise llm.Looping("fw", "m", chars=8200, repeats=0, elapsed=12)
            return {"role": "assistant", "content": "answer"}

        monkeypatch.setattr(agent_runner.llm, "chat_with_tools", flaky)
        _, transcript = agent_runner.run("sys", "user", str(repo), log=None)
        assert any("narrated" in line and "no tool call" in line for line in transcript)


class TestReviewRoundOne:
    """agentic-review#18, review at d5d2ab7 — three of its four questions."""

    def test_the_prose_cap_yields_to_a_tool_call_in_progress(self, monkeypatch):
        """A turn that has started a tool call is doing its job, however much
        it said first; the cut is for turns that never get there."""
        text = TestNarrationIsCutAtTheProseCap()._prose(200)
        call = {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "c1", "type": "function",
             "function": {"name": "grep", "arguments": "{}"}}]},
            "finish_reason": None}]}
        chunks = [call] + [_content(text[i:i + 100]) for i in range(0, len(text), 100)]
        _serve(_Stream(_sse(chunks)), monkeypatch)
        choice = llm._post("https://x", "k", {"messages": [], "tools": [{}]}, 5, "fw", "m")
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "grep"
        assert len(choice["message"]["content"]) == len(text)

    def test_the_estimate_counts_the_whole_request(self):
        msgs = [{"role": "user", "content": "x" * 3500}]
        bare = llm._estimate_tokens({"messages": msgs})
        with_tools = llm._estimate_tokens({"messages": msgs, "tools": [{"d": "y" * 3500}],
                                           "stream": True, "stream_options": {}})
        assert with_tools >= bare + 900, "the tool schemas are part of the prompt"

    def test_a_shortened_revision_marks_the_review(self, monkeypatch):
        monkeypatch.setitem(review._CURRENT, "stats", {"messages": [{"role": "user"}]})
        monkeypatch.setitem(review._CURRENT, "answer_shortened", False)

        def resume(messages, question, root, **kw):
            kw["stats"].update(tool_calls=1, turns=1, shortened=True)
            return json.dumps({"revisions": [{"index": 0, "action": "keep", "why": "ok"}]}), []

        monkeypatch.setattr(review.agent, "resume", resume)
        finding = {"file": "f.py", "line": 1, "severity": "high", "title": "t", "detail": "d"}
        kept, withdrawn = review._revise([finding], ".", "r")
        assert kept and not withdrawn
        assert review._CURRENT["answer_shortened"] is True

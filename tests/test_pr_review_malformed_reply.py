"""The reply that broke pr-review on 2026-08-31, and the retry that repeated it.

caeli-marketing `scrum-1194-brand-pages`, run 33451583827. The agent emitted an
unescaped `"` inside a JSON string value — `Rendering "Not covered" turns "we
never classified this" into ...` — in BOTH draws, escaping correctly inside the
`fix` field of the same object each time. It is a mechanical slip, and the retry
re-sent the prompt unchanged, so there was nothing to slip differently against.

Two separate defects fall out of that run, and this file pins both:

  1. The page never named the cause. The first draw's fragment-salvage returned
     one finding out of the broken list, so the error read "parsed but carried
     no `findings` list (keys: ['detail', 'file', ...])" — a description of the
     fragment. The second draw's error printed only the reply's TAIL, which was
     well-formed, so it read like truncation. It was neither.
  2. The retry was a second draw, not a correction.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from conftest import load_script  # noqa: E402


@pytest.fixture
def pr():
    return load_script("pr-review")


#: Shrunk from the real reply. The unescaped quotes around `Not covered` are the
#: whole defect; everything else is shape.
BROKEN = (
    '{"wire_fields": [],\n "findings": [\n'
    '  {"file": "lib/brand-summary.ts", "line": 80, "severity": "medium",\n'
    '   "title": "asserts a definite no",\n'
    '   "detail": "Rendering "Not covered" turns an unconfirmed product into a '
    'denied one.",\n'
    '   "fix_verified": true},\n'
    '  {"file": "lib/brand-summary.ts", "line": 96, "severity": "low",\n'
    '   "title": "plural verb for a singular subject",\n'
    '   "detail": "one of the N ... qualify should be qualifies.",\n'
    '   "fix_verified": true}\n ]}'
)


class TestTheErrorNamesTheCause:
    def test_a_malformed_outer_object_is_not_salvaged_into_a_fragment(self, pr):
        """The scavenged fragment IS parseable, so the old code returned it and
        the real syntax error was never reported."""
        with pytest.raises(pr.ReviewError) as e:
            pr.parse_findings(BROKEN)
        assert "malformed top-level object" in str(e.value)

    def test_it_quotes_the_offending_text(self, pr):
        """The correction is only useful if it shows the model its own slip."""
        with pytest.raises(pr.ReviewError) as e:
            pr.parse_findings(BROKEN)
        assert "Not covered" in str(e.value), "the retry has nothing to correct"
        assert "char " in str(e.value)

    def test_a_wellformed_non_answer_is_still_handed_on_leniently(self, pr):
        """`{}` and `{"checked": [...]}` are the model's defeat-shrugs. They
        decode FINE at the top level, only validate_findings can say they are
        not answers, and the new raise must not start swallowing them.

        TWO top-level objects on purpose. My first version passed `"{}"`, which
        never reaches this branch at all: `cronlib.parse_json_reply` succeeds on
        it and `parse_findings` returns before the scanner runs, so the test was
        green against a mutation that deleted the branch outright. A reply that
        cronlib CANNOT read, whose objects each decode cleanly, is what actually
        exercises it.
        """
        two = '{"note": "here it is"}\n{"checked": ["a"]}'
        with pytest.raises(pr.ReviewError):
            pr.llm.parse_json_reply(two)   # the early path must not fire
        assert pr.parse_findings(two) == {"checked": ["a"]}

    def test_a_good_reply_is_unaffected(self, pr):
        ok = '{"findings": [{"file": "a.ts", "line": 1, "severity": "low"}]}'
        assert len(pr.parse_findings(ok)["findings"]) == 1

    def test_prose_around_the_object_is_still_tolerated(self, pr):
        assert pr.parse_findings(
            'Here you go:\n{"findings": []}\nhope that helps'
        ) == {"findings": []}


class TestTheRetryCorrectsRatherThanRerolls:
    def test_the_second_ask_carries_the_parser_message(self, pr):
        err = pr.ReviewError('malformed top-level object: Expecting \',\' '
                           'delimiter at char 894. Around there: \'Rendering '
                           '"Not covered" turns\'')
        out = pr._correction(err)
        assert "Not covered" in out, "the model is not shown its own slip"
        assert "char 894" in out

    def test_it_names_the_escaping_rule_that_was_broken(self, pr):
        out = pr._correction(pr.ReviewError("x"))
        assert '\\"' in out and "\\n" in out
        assert "REJECTED" in out

    def test_it_is_bounded(self, pr):
        """A 30k reply's error must not double the prompt."""
        assert len(pr._correction(pr.ReviewError("x" * 40_000))) < 1500

    def test_the_retry_actually_appends_it(self, pr, monkeypatch):
        """A correction nothing sends is decoration. The old bug was precisely
        that `_reply` re-sent `prompt` unchanged."""
        seen = []

        def fake_ask(prompt, work, need_evidence):
            seen.append(prompt)
            if len(seen) == 1:
                raise pr.ReviewError("malformed top-level object: bad quote")
            return {"findings": []}

        monkeypatch.setattr(pr, "_ask", fake_ask)
        monkeypatch.setattr(pr, "_remaining_budget", lambda: 900)
        pr._reply("ORIGINAL PROMPT", "/tmp", "review")
        assert len(seen) == 2
        assert seen[0] == "ORIGINAL PROMPT"
        assert seen[1].startswith("ORIGINAL PROMPT")
        assert "REJECTED" in seen[1], "the retry was a blind re-roll again"
        assert "bad quote" in seen[1]

    def test_an_agent_failure_is_still_not_retried(self, pr, monkeypatch):
        """A provider outage re-rolled as a shrug spends the budget on a run
        that cannot succeed — the existing distinction must survive."""
        calls = []

        def fake_ask(prompt, work, need_evidence):
            calls.append(prompt)
            raise pr.AgentFailed("hermes exited 1")

        monkeypatch.setattr(pr, "_ask", fake_ask)
        monkeypatch.setattr(pr, "_remaining_budget", lambda: 900)
        with pytest.raises(pr.AgentFailed):
            pr._reply("P", "/tmp", "review")
        assert len(calls) == 1

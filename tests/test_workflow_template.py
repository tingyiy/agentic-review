"""The reusable workflow's shape — the parts a YAML typo would break silently.

The gate expression appears four times (concurrency, name, runs-on, the Gate
step) because Actions cannot share an expression between them. A copy that
drifts reviews on the wrong runner, or names the wrong arm, with no error.
"""
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REUSABLE = ROOT / ".github/workflows/pr-review.yml"
SELF_CALLER = ROOT / ".github/workflows/pr-review-caller.yml"
EXAMPLE = ROOT / "examples/pr-review-caller.yml"


def _norm(expr):
    return " ".join(str(expr).split())


@pytest.fixture(scope="module")
def wf():
    return yaml.safe_load(REUSABLE.read_text())


def _gate_from_runs_on(job):
    m = re.fullmatch(r"\$\{\{\s*fromJSON\((.*)\)\s*\}\}", _norm(job["runs-on"]))
    assert m, job["runs-on"]
    cond = m.group(1)
    depth = 0
    for i, ch in enumerate(cond):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0:
            return cond[:i + 1]
    raise AssertionError("unbalanced")


class TestTheGateIsEvaluatedOnceAndCopiedExactly:
    def test_the_gate_step_and_runs_on_agree(self, wf):
        job = wf["jobs"]["review"]
        gate_step = next(s for s in job["steps"] if s.get("id") == "gate")
        m = re.search(r"work=\$\{\{(.*)\}\}", _norm(gate_step["run"]))
        assert m, gate_step["run"]
        assert _norm(m.group(1)) == _norm(_gate_from_runs_on(job))

    def test_every_working_step_reads_the_gate_output(self, wf):
        steps = {s["name"]: s for s in wf["jobs"]["review"]["steps"] if "name" in s}
        for name in ("Fetch the reviewer", "Review", "Objection check", "Clean up"):
            assert "steps.gate.outputs.work == 'true'" in steps[name]["if"], name
        assert "runner.environment" not in REUSABLE.read_text(), (
            "the runner kind is not the gate any more — both arms can land on "
            "GitHub-hosted")

    def test_the_review_and_the_objection_check_are_exclusive(self, wf):
        steps = {s["name"]: s for s in wf["jobs"]["review"]["steps"] if "name" in s}
        assert "!github.event.comment" in steps["Review"]["if"]
        assert "github.event.comment" in steps["Objection check"]["if"]


class TestHostedRuns:
    def test_the_secrets_are_declared_and_optional(self, wf):
        secrets = wf[True]["workflow_call"]["secrets"]
        for name in ("FIREWORKS_API_KEY", "OPENROUTER_API_KEY", "REVIEW_GITHUB_TOKEN"):
            assert secrets[name]["required"] is False, name

    def test_an_empty_secret_never_shadows_a_self_hosted_environment(self, wf):
        """Exported only when non-empty — otherwise a hosted-mode field set to
        '' on a self-hosted run would hide the key in the env files."""
        steps = {s["name"]: s for s in wf["jobs"]["review"]["steps"] if "name" in s}
        for name in ("Review", "Objection check"):
            run = steps[name]["run"]
            assert 'export FIREWORKS_API_KEY="$IN_FIREWORKS_API_KEY"' in run
            assert '[ -n "$IN_FIREWORKS_API_KEY" ] &&' in run
            assert "FIREWORKS_API_KEY" not in steps[name]["env"], (
                "the secret must arrive under a staging name, not as the key itself")

    def test_a_fork_pr_without_a_key_skips_instead_of_failing(self, wf):
        steps = {s["name"]: s for s in wf["jobs"]["review"]["steps"] if "name" in s}
        assert "not reviewing" in steps["Review"]["run"]
        assert "exit 0" in steps["Review"]["run"]

    def test_this_repository_has_no_tracker_so_no_title_check(self):
        caller = yaml.safe_load(SELF_CALLER.read_text())
        assert caller["jobs"]["review"]["with"]["ticket_pattern"] == ""
        wf = yaml.safe_load(REUSABLE.read_text())
        review = next(s for s in wf["jobs"]["review"]["steps"] if s.get("name") == "Review")
        assert review["env"]["REVIEW_TICKET_PATTERN"] == "${{ inputs.ticket_pattern }}"

    def test_this_repository_reviews_itself_hosted(self):
        caller = yaml.safe_load(SELF_CALLER.read_text())
        job = caller["jobs"]["review"]
        assert job["with"]["runner"] == '["ubuntu-latest"]'
        assert job["with"]["post_as_actions_bot"] is True
        assert job["with"]["bot_login"] == "github-actions[bot]"
        assert caller["permissions"]["pull-requests"] == "write"
        assert "issue_comment" in caller[True]

    def test_the_example_grants_the_permissions_the_workflow_needs(self):
        ex = yaml.safe_load(EXAMPLE.read_text())
        for k, v in yaml.safe_load(REUSABLE.read_text())["permissions"].items():
            assert ex["permissions"].get(k) == v, k


class TestTheBotLoginCanBeToldInsteadOfAsked:
    def test_review_bot_login_wins_over_the_api(self, monkeypatch):
        """The workflow's own GITHUB_TOKEN cannot call /user; the caller knows
        the login anyway."""
        from agentic_review import review
        review._me.cache_clear()          # cached per process; the env is the test
        monkeypatch.setenv("REVIEW_BOT_LOGIN", "github-actions[bot]")
        monkeypatch.setattr(review, "gh", lambda *a, **k: (_ for _ in ()).throw(AssertionError("asked")))
        assert review._me() == "github-actions[bot]"

    def test_without_it_the_api_is_asked(self, monkeypatch):
        from agentic_review import review
        review._me.cache_clear()
        monkeypatch.delenv("REVIEW_BOT_LOGIN", raising=False)
        monkeypatch.setattr(review, "gh", lambda *a, **k: '{"login": "review-bot"}')
        assert review._me() == "review-bot"

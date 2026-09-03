"""One pass that can drop, correct and add, against the conversation that read
the code.

Tingyi's simplification, 2026-09-02. It replaced two passes that were asking the
same model about the same evidence from different chairs: a "second look" that
could only ADD, and a `_reflect` that could only DELETE — and `_reflect` was a
fresh agent loop with its own tools, spending ~84s re-reading files the review
pass already had in context.

The third action exists because of caeli-marketing#212. Of eight findings
posted, the author took one, disputed two with measurements, and escalated one
as a PRODUCT CALL rather than a defect. Deleting that last one would have been
wrong; keeping it as written was wrong too. Neither old pass could re-cast it.
"""
import json

import pytest


@pytest.fixture
def pr():
    from agentic_review import review
    return review


def _findings(n=2):
    return [{"file": f"f{i}.py", "line": 10 + i, "severity": "high",
             "title": f"t{i}", "detail": f"d{i}"} for i in range(n)]


def _armed(pr, monkeypatch, reply, tool_calls=1):
    monkeypatch.setitem(pr._CURRENT, "stats", {"messages": [{"role": "user"}]})

    def fake_resume(messages, question, root, **kw):
        kw.get("stats", {}).update(tool_calls=tool_calls, turns=1)
        if isinstance(reply, Exception):
            raise reply
        return reply, ["turn 1: answered"]

    monkeypatch.setattr(pr.agent, "resume", fake_resume)


class TestItDrops:
    def test_a_wrong_finding_is_withdrawn(self, pr, monkeypatch):
        """The most valuable action and the one the model is most reluctant to
        take. On #212 its own judgement caught all three React claims the author
        later disputed by reproducing the reconciler."""
        _armed(pr, monkeypatch, json.dumps({"revisions": [
            {"index": 0, "action": "drop", "why": "state survives reconciliation"},
            {"index": 1, "action": "keep", "why": "confirmed"}]}))
        kept, withdrawn = pr._revise(_findings(), "/tmp", "repo")
        assert [f["title"] for f in kept] == ["t1"]
        assert withdrawn[0][0]["title"] == "t0"
        assert "reconciliation" in withdrawn[0][2]

    def test_the_reason_reaches_the_withdrawn_note(self, pr, monkeypatch):
        """The post shows its work — a finding that vanished with no reason
        reads as the reviewer losing its nerve."""
        _armed(pr, monkeypatch, json.dumps({"revisions": [
            {"index": 0, "action": "drop", "why": "already handled in crud.py"},
            {"index": 1, "action": "keep"}]}))
        _, withdrawn = pr._revise(_findings(), "/tmp", "repo")
        assert "crud.py" in pr._withdrawn_note(withdrawn)


class TestItEdits:
    def test_severity_and_wording_can_be_corrected(self, pr, monkeypatch):
        _armed(pr, monkeypatch, json.dumps({"revisions": [
            {"index": 0, "action": "edit", "severity": "low",
             "title": "is this deliberate?", "why": "intentional design"},
            {"index": 1, "action": "keep"}]}))
        kept, withdrawn = pr._revise(_findings(), "/tmp", "repo")
        assert kept[0]["severity"] == "low"
        assert kept[0]["title"] == "is this deliberate?"
        assert withdrawn == [], "an edit is not a withdrawal"

    def test_an_omitted_field_keeps_its_value(self, pr, monkeypatch):
        """A terse edit must not blank a finding's detail."""
        _armed(pr, monkeypatch, json.dumps({"revisions": [
            {"index": 0, "action": "edit", "severity": "medium"},
            {"index": 1, "action": "keep"}]}))
        kept, _ = pr._revise(_findings(), "/tmp", "repo")
        assert kept[0]["detail"] == "d0" and kept[0]["title"] == "t0"

    def test_an_empty_string_does_not_blank_a_field(self, pr, monkeypatch):
        _armed(pr, monkeypatch, json.dumps({"revisions": [
            {"index": 0, "action": "edit", "detail": "   "},
            {"index": 1, "action": "keep"}]}))
        kept, _ = pr._revise(_findings(), "/tmp", "repo")
        assert kept[0]["detail"] == "d0"


class TestItAdds:
    def test_a_new_finding_is_appended(self, pr, monkeypatch):
        _armed(pr, monkeypatch, json.dumps({
            "revisions": [{"index": 0, "action": "keep"},
                          {"index": 1, "action": "keep"}],
            "additions": [{"file": "n.py", "line": 5, "severity": "medium",
                           "title": "route is untested", "detail": "nothing "
                           "under tests/ names this path"}]}))
        kept, _ = pr._revise(_findings(), "/tmp", "repo")
        assert [f["title"] for f in kept][-1] == "route is untested"

    def test_a_restatement_is_not_added_twice(self, pr, monkeypatch):
        _armed(pr, monkeypatch, json.dumps({
            "revisions": [{"index": 0, "action": "keep"},
                          {"index": 1, "action": "keep"}],
            "additions": [{"file": "f0.py", "line": 10, "severity": "high",
                           "title": "T0!", "detail": "same thing reworded"}]}))
        kept, _ = pr._revise(_findings(), "/tmp", "repo")
        assert len(kept) == 2

    def test_a_malformed_addition_does_not_lose_the_kept_findings(
            self, pr, monkeypatch):
        _armed(pr, monkeypatch, json.dumps({
            "revisions": [{"index": 0, "action": "keep"},
                          {"index": 1, "action": "keep"}],
            "additions": [{"nonsense": True}]}))
        kept, _ = pr._revise(_findings(), "/tmp", "repo")
        assert len(kept) == 2


class TestItNeverLosesAReview:
    def test_an_unusable_reply_posts_everything_as_written(self, pr, monkeypatch):
        _armed(pr, monkeypatch, "not json at all")
        monkeypatch.setattr(pr, "_keep_unusable_reply", lambda *a, **k: None)
        kept, withdrawn = pr._revise(_findings(), "/tmp", "repo")
        assert len(kept) == 2 and withdrawn == []

    def test_a_MISSING_entry_refuses_the_whole_revision(self, pr, monkeypatch):
        """STRICT, and this is the rule that stops a broken revision from
        silently emptying a review. A missing entry is not a verdict of keep and
        certainly not one of drop — coercing either way decides something the
        model never said."""
        _armed(pr, monkeypatch, json.dumps({"revisions": [
            {"index": 0, "action": "drop", "why": "wrong"}]}))
        monkeypatch.setattr(pr, "_keep_unusable_reply", lambda *a, **k: None)
        kept, withdrawn = pr._revise(_findings(3), "/tmp", "repo")
        assert len(kept) == 3 and withdrawn == []

    def test_no_conversation_means_no_call(self, pr, monkeypatch):
        monkeypatch.setitem(pr._CURRENT, "stats", {})
        monkeypatch.setattr(pr.agent, "resume",
                            lambda *a, **k: pytest.fail("asked with no messages"))
        assert pr._revise(_findings(), "/tmp", "repo")[0] == _findings()

    def test_a_provider_failure_keeps_the_originals(self, pr, monkeypatch):
        _armed(pr, monkeypatch, "")
        assert len(pr._revise(_findings(), "/tmp", "repo")[0]) == 2

    def test_preserving_the_reply_cannot_itself_lose_the_review(self, pr,
                                                                monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("no run page")
        _armed(pr, monkeypatch, "junk")
        monkeypatch.setattr(pr, "_keep_unusable_reply", boom)
        assert len(pr._revise(_findings(), "/tmp", "repo")[0]) == 2

    def test_no_findings_makes_no_call(self, pr, monkeypatch):
        monkeypatch.setitem(pr._CURRENT, "stats", {"messages": [{"role": "u"}]})
        monkeypatch.setattr(pr.agent, "resume",
                            lambda *a, **k: pytest.fail("asked with 0 findings"))
        assert pr._revise([], "/tmp", "repo") == ([], [])


class TestDroppingEverythingApprovesOnEvidence:
    """new-employer-portal#34: three findings raised, all three re-read in the
    checkout with 10 tool calls and scored 0 — and the post was a COMMENT
    headed "NOT a clean review", asking a person to re-verify the reviewer's
    own retractions. Nobody does that, so the PR just sat unapproved.

    A drop is only honoured when the revision opened files, so an all-withdrawn
    result already carries the evidence an approval requires: the code was
    re-read and each finding was explained away in writing. That is a stronger
    clean verdict than an empty first pass, not a weaker one."""
    def _all_dropped(self, pr, monkeypatch):
        _armed(pr, monkeypatch, json.dumps({"revisions": [
            {"index": 0, "action": "drop", "why": "wrong"},
            {"index": 1, "action": "drop", "why": "already handled"}]}))
        kept, withdrawn = pr._revise(_findings(), "/tmp", "repo")
        assert kept == [] and len(withdrawn) == 2
        return kept, withdrawn

    def test_an_all_dropped_review_approves(self, pr, monkeypatch):
        kept, withdrawn = self._all_dropped(pr, monkeypatch)
        _, event = pr._finalize_review(kept, withdrawn)
        assert event == "APPROVE"

    def test_the_withdrawals_stay_visible_in_the_approval(self, pr, monkeypatch):
        """The retractions are the evidence; hiding them would make this
        indistinguishable from a first pass that found nothing."""
        kept, withdrawn = self._all_dropped(pr, monkeypatch)
        body, _ = pr._finalize_review(kept, withdrawn)
        assert "t0" in body and "t1" in body
        assert "withdrawn" in body
        # And the headline says so, rather than the plain "no findings" a
        # clean first pass posts.
        assert "### AI review — no findings\n" not in body
        assert "no findings stood" in body
        assert "What this approval is" in body

    def test_a_partial_review_still_does_not_approve(self, pr, monkeypatch):
        """Exclusions cap the verdict regardless of how the findings went."""
        kept, withdrawn = self._all_dropped(pr, monkeypatch)
        body, event = pr._finalize_review(kept, withdrawn, excluded=["src/b.py"])
        assert event == "COMMENT"
        assert "src/b.py" in body and "t0" in body
        # And the body must not claim the approval the event is not posting.
        assert "approval rests" not in body
        assert "verdict rests" in body

    def test_a_full_review_names_it_an_approval(self, pr, monkeypatch):
        kept, withdrawn = self._all_dropped(pr, monkeypatch)
        body, event = pr._finalize_review(kept, withdrawn)
        assert event == "APPROVE" and "approval rests" in body


class TestThePromptAsksForTheRightThings:
    def test_it_asks_the_model_to_disagree_with_itself(self, pr):
        """The risk this shape takes on is self-consistency: the model judges
        findings it is invested in, without the fresh framing a separate pass
        gave it. That cannot be designed away, only asked for and measured."""
        q = pr.REVISE.lower()
        assert "disagree with yourself" in q
        # "the action you will be most reluctant to take" was REMOVED on purpose:
        # it over-steered, and the model dropped 15 of 15 findings to look
        # decisive. The guard now runs the other way.
        assert "do not drop to look decisive" in q

    def test_it_offers_asking_instead_of_asserting(self, pr):
        """The #212 case: deliberate behaviour is neither a defect to assert nor
        a finding to delete."""
        assert "into a question" in pr.REVISE.lower()

    def test_it_says_adding_nothing_is_acceptable(self, pr):
        assert "Adding nothing is a perfectly good answer" in pr.REVISE

    def test_it_names_the_categories_we_measured_ourselves_missing(self, pr):
        q = pr.REVISE.lower()
        assert "test directory" in q and "below it" in q
        assert "second writer" in q
        assert "docstring" in q


class TestTheRevisionCanStillInvestigate:
    """Tingyi's correction, 2026-09-02, to a simplification of mine that went
    too far.

    I had moved this pass onto a tools-off single call to stop it re-exploring,
    and removed its ability to CHECK anything in the process. That guts `drop`,
    which is the most valuable action it has: "this is already handled in
    crud.py" and "nothing under tests/ names this route" are both lookups, and a
    model that cannot look can only drop what it can disprove from memory — the
    old reflect prompt said so outright ("open the file, check the path exists").
    """

    def test_it_resumes_the_conversation_rather_than_starting_one(self, pr,
                                                                  monkeypatch):
        """Both halves matter: the code the review pass read is still in
        context, AND the prefix is unchanged so the whole pass is a cache hit."""
        seen = {}
        monkeypatch.setitem(pr._CURRENT, "stats",
                            {"messages": [{"role": "system", "content": "s"},
                                          {"role": "user", "content": "u"}]})

        def fake_resume(messages, question, root, **kw):
            seen["n"] = len(messages)
            seen["root"] = root
            return json.dumps({"revisions": [{"index": 0, "action": "keep"},
                                             {"index": 1, "action": "keep"}]}), []

        monkeypatch.setattr(pr.agent, "resume", fake_resume)
        pr._revise(_findings(), "/checkout", "repo")
        assert seen["n"] == 2, "the prior conversation was not carried"
        assert seen["root"] == "/checkout", "no checkout means no tools"

    def test_the_tool_count_is_reported(self, pr, monkeypatch, capsys):
        """So a run where the revision checked nothing is visible, rather than
        being assumed to have verified its drops."""
        _armed(pr, monkeypatch, json.dumps({"revisions": [
            {"index": 0, "action": "drop", "why": "handled in crud.py"},
            {"index": 1, "action": "keep"}]}), tool_calls=3)
        pr._revise(_findings(), "/tmp", "repo")
        assert "3 tool call(s)" in capsys.readouterr().out

    def test_a_loop_failure_posts_the_originals(self, pr, monkeypatch):
        """With tools on, this pass can now time out or hit a provider error —
        neither may cost the review it is revising."""
        _armed(pr, monkeypatch, pr.agent.Timeout("deadline exhausted", []))
        kept, withdrawn = pr._revise(_findings(), "/tmp", "repo")
        assert len(kept) == 2 and withdrawn == []

    def test_supersession_still_propagates(self, pr, monkeypatch):
        """A newer run taking over must not be swallowed as a revision failure —
        this run would carry on and post a review nobody is waiting for."""
        _armed(pr, monkeypatch, pr.Superseded("a newer run took over"))
        with pytest.raises(pr.Superseded):
            pr._revise(_findings(), "/tmp", "repo")

    def test_a_merged_pr_still_propagates(self, pr, monkeypatch):
        _armed(pr, monkeypatch, pr.PRClosed("merged mid-run"))
        with pytest.raises(pr.PRClosed):
            pr._revise(_findings(), "/tmp", "repo")


class TestADropNeedsEvidence:
    """caeli-marketing#212 at the labelled commit, two runs: the revision
    dropped 5 of 5 and then 10 of 10 findings with ZERO tool calls, on reasons
    like "deliberate per the ticket comment" and "correct as written". It judged
    from memory and abandoned the one finding the author later took. Same rule
    as the approval gate: a judgement is worth the evidence behind it."""

    def test_a_drop_with_no_tool_calls_is_kept(self, pr, monkeypatch, capsys):
        _armed(pr, monkeypatch, json.dumps({"revisions": [
            {"index": 0, "action": "drop", "why": "deliberate per the ticket"},
            {"index": 1, "action": "keep"}]}), tool_calls=0)
        kept, withdrawn = pr._revise(_findings(), "/tmp", "repo")
        assert len(kept) == 2 and withdrawn == []
        assert "NOT honoured" in capsys.readouterr().out

    def test_a_drop_with_evidence_is_honoured(self, pr, monkeypatch):
        _armed(pr, monkeypatch, json.dumps({"revisions": [
            {"index": 0, "action": "drop", "why": "opened crud.py: handled at L40"},
            {"index": 1, "action": "keep"}]}), tool_calls=2)
        kept, withdrawn = pr._revise(_findings(), "/tmp", "repo")
        assert len(kept) == 1 and len(withdrawn) == 1

    def test_edits_do_not_need_tool_calls(self, pr, monkeypatch):
        """Re-casting a deliberate behaviour as a question is a wording change,
        not a claim the finding is false."""
        _armed(pr, monkeypatch, json.dumps({"revisions": [
            {"index": 0, "action": "edit", "severity": "low",
             "title": "is it intended that the CTA shows beside Sign in?"},
            {"index": 1, "action": "keep"}]}), tool_calls=0)
        kept, _ = pr._revise(_findings(), "/tmp", "repo")
        assert kept[0]["severity"] == "low"

    def test_the_prompt_says_deliberate_is_not_a_drop(self, pr):
        q = pr.REVISE
        assert "NOT reasons to drop" in q
        assert "turn it\n          into a QUESTION" in q or "into a QUESTION" in q

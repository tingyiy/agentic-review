"""The reviewer's own parse and shape guards (2026-08-22).

This file exists because the reviewer asked for it, reviewing the very branch
that was named for the parse fix:

    "The robust-parse logic this branch is named for has zero test coverage …
     it is the one scenario where a regression is silent-as-an-APPROVE (the
     worst possible outcome)."

That is the whole argument. An empty findings list means APPROVE, so anything
that degrades a reply into "no findings" does not fail — it silently blesses
code nobody reviewed. `.get("findings", [])` was exactly that bug, found in the
single-call reviewer this one replaces.

The parse cases are real shapes, not invented ones: the agent NARRATES before it
answers, and its prose quotes fragments of the schema on the way past — which is
why the fallback takes the LAST balanced object rather than the first, and why a
whole-body `json.loads` (right for a json_mode completion) died 489 chars into
the first real run.
"""
import json

import pytest
from agentic_review.errors import ReviewError as ScanError

from conftest import load_script


@pytest.fixture(scope="module")
def prr():
    return load_script("pr-review")


FINDING = {"severity": "high", "title": "t", "file": "a.py", "detail": "d"}


class TestParseFindings:
    def test_a_clean_json_body_parses(self, prr):
        """The json_mode path. Must keep working — cronlib's parser is tried
        first precisely so a clean reply is never touched by salvage logic."""
        assert prr.parse_findings(json.dumps({"findings": [FINDING]}))["findings"] == [FINDING]

    def test_narration_before_the_answer(self, prr):
        """THE failure that prompted this. Verbatim shape of the first live run:
        the agent opened with prose containing braces and quotes, and the answer
        was at the end."""
        reply = ('I\'ve read the checkout. Key verification: the workflow invokes '
                 '`$HOME/.hermes/{profile}/scripts/pr-review.py`, and "findings" '
                 'in that context means {something else}.\n\n'
                 + json.dumps({"findings": [FINDING]}))
        assert prr.parse_findings(reply)["findings"] == [FINDING]

    def test_a_fenced_block_after_prose_with_braces(self, prr):
        reply = ("Here is what I found — note the `{}` idiom used in cronlib.\n"
                 "```json\n" + json.dumps({"findings": []}) + "\n```\n")
        assert prr.parse_findings(reply)["findings"] == []

    def test_the_last_object_wins_not_the_first(self, prr):
        """Load-bearing direction. The narration quotes the SCHEMA on the way
        past, so the first balanced object is an example and the last is the
        answer. Taking the first would post the example as a review."""
        reply = ('The schema you want is {"findings": [{"severity": "low"}]} — '
                 'here is the real answer.\n' + json.dumps({"findings": [FINDING]}))
        assert prr.parse_findings(reply)["findings"] == [FINDING]

    def test_braces_inside_a_quoted_snippet(self, prr, monkeypatch):
        """THE failure that took down the review of #95, and the reason this no
        longer counts braces.

        Counting `{`/`}` to find a balanced span cannot tell a brace in CODE from
        a brace in a STRING — and quoting code back at you is this reviewer's
        entire job. Reviewing a workflow file means quoting `${{ … }}`, so one
        partial expression inside a finding's `detail` threw the span off and a
        completed, perfectly valid review was discarded.

        Forced onto the fallback path deliberately: `cronlib.parse_json_reply`
        has salvage heuristics that rescue this case, which is exactly why the
        bug survived — the earlier tests never reached the code under test.
        """
        monkeypatch.setattr(prr.llm, "parse_json_reply",
                            lambda r: (_ for _ in ()).throw(ScanError("forced")))
        for detail in (
            "put `${{ github.event.action` into concurrency.group",   # unbalanced {
            "the group key needs a closing } here",                    # lone }
            "use `${{ github.event.action }}` as normal",              # balanced
        ):
            obj = {"findings": [{**FINDING, "detail": detail}]}
            reply = "I read the workflow.\n\n" + json.dumps(obj)
            assert prr.parse_findings(reply) == obj, f"failed on: {detail}"

    def test_a_trailing_postscript_does_not_bury_the_answer(self, prr, monkeypatch):
        """The behaviour the answer/fallback split actually adds.

        An object AFTER the real answer is the case that separates the two
        implementations. Old code took the last balanced span, full stop — so a
        `{"note": …}` postscript became "the reply", and validate_findings then
        declined a review that had in fact been made. Scanning every candidate
        and preferring the last one that HAS `findings` is what fixes that.

        Written this way on the reviewer's advice: the first version put the
        decoy BEFORE the answer, where plain 'last wins' decides it and the test
        passes with or without the fix. It guarded nothing, and the mutation run
        said so — one failure, not two — which I read straight past.
        """
        monkeypatch.setattr(prr.llm, "parse_json_reply",
                            lambda r: (_ for _ in ()).throw(ScanError("forced")))
        real = {"findings": [FINDING]}
        reply = ('The shape is {"findings": [{"severity": "low", "title": "x"}]}.\n'
                 + json.dumps(real)
                 + '\n\n{"note": "reviewed at HEAD, tests not run"}')
        assert prr.parse_findings(reply) == real

    def test_a_nested_findings_key_cannot_hijack_the_answer(self, prr, monkeypatch):
        """The silent APPROVE. Found by the reviewer, on its own merged PR.

        Scanning every `{` without skipping past what was decoded walks straight
        back into the object just parsed. A NESTED dict carrying a `findings`
        key then wins the slot by being later, and an empty one turns a review
        that DID find something into a formal approval:

            {"findings": [{… "meta": {"findings": []}}]}   -> APPROVE

        Which is the worst outcome this module has, so nesting is excluded
        structurally — only top-level objects are candidates — rather than
        hoped against.

        Note the reviewer's own example did not reproduce: it put the decoy
        inside a STRING, where the quotes are escaped in the raw text and
        `raw_decode` fails on the backslash. The mechanism was real anyway, and
        this is a case that actually fires.
        """
        monkeypatch.setattr(prr.llm, "parse_json_reply",
                            lambda r: (_ for _ in ()).throw(ScanError("forced")))
        real = {"findings": [{**FINDING, "meta": {"findings": []}}]}
        assert prr.parse_findings(json.dumps(real)) == real
        assert len(prr.validate_findings(prr.parse_findings(json.dumps(real)))) == 1

    def test_no_json_at_all_raises(self, prr):
        """A reviewer that produced NOTHING must not look like one that found
        nothing. This is the silent-APPROVE direction."""
        with pytest.raises(ScanError):
            prr.parse_findings("I was unable to review this. Sorry!")

    def test_an_unterminated_object_raises(self, prr):
        """Truncation at the token budget. There is no balanced object to find,
        and half a findings list must never be posted as a whole review."""
        with pytest.raises(ScanError):
            prr.parse_findings('prose {"findings": [{"severity": "high",')


class TestValidateFindings:
    def test_findings_pass_through(self, prr):
        assert prr.validate_findings({"findings": [FINDING]}) == [FINDING]

    def test_an_empty_list_is_a_real_approval(self, prr):
        """The ONE case that legitimately approves: the agent answered, and the
        answer was 'nothing'. It must stay distinguishable from every failure
        below, all of which are the agent NOT answering."""
        assert prr.validate_findings({"findings": []}) == []

    @pytest.mark.parametrize("parsed", [
        {},                                  # key missing entirely
        {"findings": None},                  # null
        {"findings": {}},                    # wrong container
        {"findings": "none"},                # a sentence, not a list
        {"result": "looks good"},            # answered a different question
    ], ids=["missing", "null", "dict", "string", "wrong-key"])
    def test_a_degraded_reply_never_approves(self, prr, parsed):
        """Each of these is falsy under `.get("findings", [])` — the original
        bug — and each would have posted a formal APPROVE on a review that never
        happened."""
        with pytest.raises(ScanError):
            prr.validate_findings(parsed)

    @pytest.mark.parametrize("findings", [
        ["oops"], [42], [None], [{"severity": "high"}, "trailing"],
    ], ids=["str", "int", "none", "mixed"])
    def test_non_object_elements_are_named_not_crashed(self, prr, findings):
        """These clear the list check and then die on `.get` inside render(),
        which guard_main reports as an undifferentiated crash. Right direction,
        wrong route — the error should say which element and how many."""
        with pytest.raises(ScanError) as e:
            prr.validate_findings({"findings": findings})
        assert "not objects" in str(e.value)

    def test_a_non_object_reply_raises(self, prr):
        """`parse_findings` can only return what it found; a bare list or string
        must not reach `.get`."""
        with pytest.raises(ScanError):
            prr.validate_findings([FINDING])


class TestConversation:
    """The author's replies. Reading these is what stops the tool re-raising a
    point somebody already answered."""

    def _stub(self, prr, monkeypatch, payloads):
        def fake_gh(path, **kw):
            for frag, body in payloads.items():
                # Compare the PATH, not the query string. `/commits` carries
                # `?per_page=100`, so a stub keyed on the bare path stopped
                # matching and this helper returned `[]` — silently, so the
                # test still passed while the code under test saw nothing.
                if path.split("?")[0].endswith(frag):
                    if isinstance(body, Exception):
                        raise body
                    return json.dumps(body)
            return "[]"
        monkeypatch.setattr(prr, "gh", fake_gh)

    def test_inline_review_replies_are_included(self, prr, monkeypatch):
        """The regression the reviewer caught. A reply typed under a diff line is
        a REVIEW comment (`/pulls/{n}/comments`), not an issue comment — and it
        is the default way an author rebuts a specific finding. Reading only
        `/issues/{n}/comments` presented that as silence."""
        self._stub(prr, monkeypatch, {
            "/pulls/94/comments": [{
                "body": "Deliberate — the floor is measured, see the runbook.",
                "user": {"login": "octocat"}, "path": "cron/scripts/x.py",
                "created_at": "2026-08-22T10:00:00Z"}],
        })
        out = prr.conversation("infra", 94)
        assert "Deliberate — the floor is measured" in out
        assert "cron/scripts/x.py" in out, "an inline reply should say what it is about"

    def test_all_three_endpoints_are_merged_in_order(self, prr, monkeypatch):
        self._stub(prr, monkeypatch, {
            # `submitted_at`, because that is what /pulls/{n}/reviews returns —
            # a review object has no `created_at`. The fixture used to say
            # `created_at`, so the ordering assertion below passed against a
            # shape production never sends, and the real reviews sorted as one
            # undated block for as long as nobody looked.
            "/pulls/94/reviews": [{"body": "first", "user": {"login": "bot"},
                                   "state": "COMMENTED",
                                   "submitted_at": "2026-08-22T09:00:00Z"}],
            "/pulls/94/comments": [{"body": "second", "user": {"login": "a"},
                                    "created_at": "2026-08-22T10:00:00Z"}],
            "/issues/94/comments": [{"body": "third", "user": {"login": "b"},
                                     "created_at": "2026-08-22T11:00:00Z"}],
        })
        out = prr.conversation("infra", 94)
        assert out.index("first") < out.index("second") < out.index("third")

    def test_a_review_sorts_by_when_it_was_submitted(self, prr, monkeypatch):
        """A review dated BETWEEN two comments must land between them.

        The ordering test above cannot catch this: it expects the review first,
        so a review that sorts to the front by accident still satisfies it. And
        it did sort by accident — `/pulls/{n}/reviews` returns `submitted_at`
        and never `created_at`, so keying on `created_at` gave every review the
        empty string, and the empty string sorts before every ISO date.

        Measured on infra#106 before the fix: six reviews arrived as one undated
        block ahead of seven dated commits, so the model was handed every
        finding followed by every answer rather than the argument they form.
        That adjacency is the entire point of feeding the conversation back.
        """
        self._stub(prr, monkeypatch, {
            "/issues/94/comments": [
                {"body": "earliest", "user": {"login": "a"},
                 "created_at": "2026-08-22T09:00:00Z"},
                {"body": "latest", "user": {"login": "b"},
                 "created_at": "2026-08-22T11:00:00Z"},
            ],
            "/pulls/94/reviews": [{"body": "middle", "user": {"login": "bot"},
                                   "state": "COMMENTED",
                                   "submitted_at": "2026-08-22T10:00:00Z"}],
        })
        out = prr.conversation("infra", 94)
        assert out.index("earliest") < out.index("middle") < out.index("latest"), (
            "a review must sort by submitted_at — keying on created_at alone "
            "puts every review in an undated block at the front"
        )

    # ---- the commit branch ------------------------------------------------
    #
    # It shipped with no coverage at all, and the stub helper was returning `[]`
    # for it, so a test written before that fix would have asserted nothing
    # while passing. Fix the helper first, then these.

    def _commit(self, message, date, name="Someone", login=None):
        """A commit as /pulls/{n}/commits actually returns it: prose under
        `commit.message`, date under `commit.author.date`, and a top-level
        `author` that is null whenever the email matches no GitHub account."""
        return {"commit": {"message": message, "author": {"name": name, "date": date}},
                "author": {"login": login} if login else None}

    def test_a_commit_message_reaches_the_conversation(self, prr, monkeypatch):
        """The whole point: an author who cannot comment as the repo owner
        answers a finding in the commit that responds to it."""
        self._stub(prr, monkeypatch, {
            "/pulls/94/commits": [self._commit(
                "Declined: flipping it would zero the funnel, here is why",
                "2026-08-22T10:00:00Z")],
        })
        out = prr.conversation("infra", 94)
        assert "Declined: flipping it would zero the funnel" in out
        assert "— commit]" in out, "a commit must be labelled, not read as a comment"

    def test_a_commit_falls_back_to_the_git_author_name(self, prr, monkeypatch):
        """`author.login` is null whenever the commit email matches no GitHub
        account — true of every commit on tests#291 — so the git name is not a
        defensive extra, it is the normal path."""
        self._stub(prr, monkeypatch, {
            "/pulls/94/commits": [self._commit("body", "2026-08-22T10:00:00Z",
                                               name="Tingyi Yang")],
        })
        assert "[Tingyi Yang — commit]" in prr.conversation("infra", 94)

    def test_a_commit_sorts_by_its_author_date(self, prr, monkeypatch):
        """Interleaved with the other sources, not appended in a block. A
        commit answering a finding has to sit NEXT to it — that adjacency is
        the entire reason the conversation is fed back at all."""
        self._stub(prr, monkeypatch, {
            "/issues/94/comments": [
                {"body": "earliest", "user": {"login": "a"},
                 "created_at": "2026-08-22T09:00:00Z"},
                {"body": "latest", "user": {"login": "b"},
                 "created_at": "2026-08-22T11:00:00Z"},
            ],
            "/pulls/94/commits": [self._commit("middle", "2026-08-22T10:00:00Z")],
        })
        out = prr.conversation("infra", 94)
        assert out.index("earliest") < out.index("middle") < out.index("latest")

    def test_an_empty_commit_message_is_dropped(self, prr, monkeypatch):
        self._stub(prr, monkeypatch, {
            "/pulls/94/commits": [self._commit("   ", "2026-08-22T10:00:00Z"),
                                  self._commit("real", "2026-08-22T11:00:00Z")],
        })
        out = prr.conversation("infra", 94)
        assert "real" in out
        assert out.count("— commit]") == 1

    def test_a_long_commit_message_is_capped(self, prr, monkeypatch):
        """Commit messages in this org run long. Uncapped, one could crowd out
        the findings it is meant to sit beside."""
        self._stub(prr, monkeypatch, {
            "/pulls/94/commits": [self._commit("x" * 4000, "2026-08-22T10:00:00Z")],
        })
        out = prr.conversation("infra", 94)
        assert "x" * 1500 in out
        assert "x" * 1600 not in out

    def test_one_endpoint_failing_keeps_the_others(self, prr, monkeypatch):
        """Losing the whole conversation is what makes the tool repeat itself, so
        a single failing endpoint must not discard the rest."""
        self._stub(prr, monkeypatch, {
            "/pulls/94/reviews": RuntimeError("502"),
            "/pulls/94/comments": [{"body": "still here", "user": {"login": "a"},
                                    "created_at": "2026-08-22T10:00:00Z"}],
        })
        assert "still here" in prr.conversation("infra", 94)

    def test_empty_bodies_are_dropped(self, prr, monkeypatch):
        """A bare APPROVE carries no argument; passing it through as an empty
        block just spends context."""
        self._stub(prr, monkeypatch, {
            "/pulls/94/reviews": [{"body": "", "user": {"login": "bot"},
                                   "state": "APPROVED",
                                   "created_at": "2026-08-22T09:00:00Z"}],
        })
        assert prr.conversation("infra", 94) == ""

    def test_no_conversation_is_empty_not_a_header(self, prr, monkeypatch):
        """A first-push review must not be told about an argument that never
        happened."""
        self._stub(prr, monkeypatch, {})
        assert prr.conversation("infra", 94) == ""


class TestReviewEvent:
    """Severity decides what the review DOES, not only what it says.

    Before this, every finding posted COMMENT — which neither blocks nor
    withholds an approval — so a 🔴 rode along on a merge as advice. Under
    `required_approving_review_count: 1` these become the actual gate.
    """

    def _f(self, sev):
        return {**FINDING, "severity": sev}

    def test_no_findings_approves(self, prr):
        assert prr.review_event([]) == "APPROVE"

    def test_only_low_approves(self, prr):
        assert prr.review_event([self._f("low"), self._f("low")]) == "APPROVE"

    def test_medium_withholds_approval(self, prr):
        assert prr.review_event([self._f("low"), self._f("medium")]) == "COMMENT"

    def test_high_requests_changes(self, prr):
        assert prr.review_event([self._f("high")]) == "REQUEST_CHANGES"

    def test_the_worst_severity_wins(self, prr):
        assert prr.review_event(
            [self._f("low"), self._f("high"), self._f("medium")]) == "REQUEST_CHANGES"

    @pytest.mark.parametrize("sev", ["critical", "blocker", "", None, "LOW ", "Medium"])
    def test_case_and_padding_are_tolerated_but_unknowns_never_approve(self, prr, sev):
        """`LOW ` and `Medium` are the same verdicts with different spelling, so
        they must map normally. `critical` and `blocker` are NOT in the
        vocabulary the model was given — mapping an unknown down to low would
        approve the most serious thing it ever found, which is the same
        inversion every other guard in this module exists to prevent."""
        event = prr.review_event([self._f(sev)])
        if str(sev).strip().lower() in ("low",):
            assert event == "APPROVE"
        elif str(sev).strip().lower() in ("medium",):
            assert event == "COMMENT"
        else:
            assert event != "APPROVE", f"{sev!r} must not produce an approval"

    def test_an_unknown_alongside_low_does_not_approve(self, prr):
        """The dangerous shape: one nit and one word we don't understand. If the
        unknown were ignored, the nit would carry the whole review to APPROVE."""
        assert prr.review_event([self._f("low"), self._f("critical")]) != "APPROVE"


class TestPostReview:
    """The POST path, which decides whether the strongest verdict quietly
    becomes a comment. Untested until the reviewer pointed out that the branch
    changing it had no coverage — 'the same class of guard the file otherwise
    treats as first-class'."""

    def _http_error(self, code, body):
        import io
        import urllib.error
        return urllib.error.HTTPError(
            "https://api.github.com/x", code, "err", {},
            io.BytesIO(body.encode()))

    def _gh(self, prr, monkeypatch, fail_first=None):
        posted = []

        def fake_gh(path, method="GET", body=None, **kw):
            posted.append(body["event"])
            if fail_first and len(posted) == 1:
                raise fail_first
            return "{}"
        monkeypatch.setattr(prr, "gh", fake_gh)
        return posted

    SELF = '{"message":"Unprocessable Entity","errors":[' \
           '{"message":"Can not approve your own pull request"}]}'

    def test_a_normal_post_returns_the_event(self, prr, monkeypatch):
        posted = self._gh(prr, monkeypatch)
        assert prr.post_review("infra", 1, "REQUEST_CHANGES", "b") == "REQUEST_CHANGES"
        assert posted == ["REQUEST_CHANGES"]

    @pytest.mark.parametrize("event", ["APPROVE", "REQUEST_CHANGES"])
    def test_a_self_review_refusal_falls_back_to_comment(self, prr, monkeypatch, event):
        """GitHub refuses BOTH verdicts on your own PR. The findings are still
        worth posting — just without the verdict."""
        posted = self._gh(prr, monkeypatch,
                          fail_first=self._http_error(422, self.SELF))
        got = prr.post_review("infra", 1, event, "b")
        assert posted == [event, "COMMENT"]
        assert got.startswith("COMMENT") and "own PR" in got

    def test_some_OTHER_422_is_raised_not_downgraded(self, prr, monkeypatch):
        """THE fix. 422 is GitHub's catch-all for unprocessable — the head moved,
        the body is too long, the PR closed underneath us. Swallowing all of them
        dropped a blocking verdict AND printed a fabricated reason for it. A
        review that could not be posted must be loud."""
        import urllib.error
        posted = self._gh(prr, monkeypatch, fail_first=self._http_error(
            422, '{"message":"body is too long (maximum is 65536 characters)"}'))
        with pytest.raises(urllib.error.HTTPError):
            prr.post_review("infra", 1, "REQUEST_CHANGES", "b")
        assert posted == ["REQUEST_CHANGES"], "must not have posted a fallback"

    def test_a_non_422_is_raised(self, prr, monkeypatch):
        import urllib.error
        self._gh(prr, monkeypatch, fail_first=self._http_error(500, "boom"))
        with pytest.raises(urllib.error.HTTPError):
            prr.post_review("infra", 1, "APPROVE", "b")

    def test_a_failing_COMMENT_is_never_retried_as_itself(self, prr, monkeypatch):
        """COMMENT is already the fallback; retrying it would loop."""
        import urllib.error
        posted = self._gh(prr, monkeypatch,
                          fail_first=self._http_error(422, self.SELF))
        with pytest.raises(urllib.error.HTTPError):
            prr.post_review("infra", 1, "COMMENT", "b")
        assert posted == ["COMMENT"]


class TestSeverityPresentation:
    def test_an_unknown_severity_is_not_drawn_as_a_nit(self, prr):
        """The two severity paths had drifted: `review_event` treated an unknown
        as blocking while `render` drew it with the blue nit icon, so a review
        WITHHOLDING approval looked identical to one granting it."""
        body = prr.render([{**FINDING, "severity": "critical", "title": "x"}], False, 0)
        assert prr.ICON["low"] not in body, "an unknown severity must not look like a nit"
        assert prr.ICON["unknown"] in body

    def test_the_verdict_and_the_icon_agree(self, prr):
        """One vocabulary, both paths. If a severity does not approve, it must
        not be drawn as the severity that does."""
        for sev in ("high", "medium", "low", "critical", "", None):
            norm = prr.normalize_severity(sev)
            body = prr.render([{**FINDING, "severity": sev}], False, 0)
            approves = prr.review_event([{**FINDING, "severity": sev}]) == "APPROVE"
            assert (prr.ICON[norm] in body)
            assert approves == (norm == "low"), f"{sev!r} -> {norm}"

    def test_findings_sort_worst_first_with_unknown_above_low(self, prr):
        body = prr.render([{**FINDING, "severity": s, "title": f"t-{s}"}
                           for s in ("low", "critical", "high", "medium")], False, 0)
        order = [body.index(f"t-{s}") for s in ("high", "medium", "critical", "low")]
        assert order == sorted(order), "worst first; an unknown outranks a nit"


class TestSeverityBreakdown:
    """`COMMENT: 3 finding(s)` says a review happened but not whether it found
    anything that matters. Those are very different runs to scroll past."""

    def test_worst_first(self, prr):
        out = prr.severity_breakdown([{**FINDING, "severity": s} for s in
                                      ("low", "high", "critical", "medium", "high")])
        assert out == "2 high, 1 medium, 1 unknown, 1 low"

    def test_no_findings(self, prr):
        assert prr.severity_breakdown([]) == "none"

    def test_unknowns_are_counted_as_unknown_not_low(self, prr):
        """The breakdown reads the same vocabulary as the verdict, so a run that
        withholds approval cannot report itself as all-nits."""
        assert prr.severity_breakdown([{**FINDING, "severity": "blocker"}]) == "1 unknown"


class TestConfirmEmptyReview:
    """An empty result is the one that APPROVES, so it gets asked twice.

    Measured on this reviewer's own prompt (n=3 per setting, 131k budget), the
    model returns the literal 15-char `{"findings":[]}` in 7 of 12 replies —
    several after single-digit output tokens. That is a shrug, and nothing
    downstream can tell it from a genuinely clean review.
    """

    def _agent(self, prr, monkeypatch, *replies):
        calls = []

        # `**k`: this stub records what was ASKED, not how it was called —
        # pinning the arity broke it the moment run_agent gained a timeout.
        def fake(prompt, work, **k):
            calls.append(1)
            r = replies[min(len(calls) - 1, len(replies) - 1)]
            if isinstance(r, Exception):
                raise r
            return r
        monkeypatch.setattr(prr, "run_agent", fake)
        return calls

    def test_findings_on_the_first_pass_are_not_second_guessed(self, prr, monkeypatch):
        """The common case must stay one agent run — the second pass is only
        paid when the first found nothing, which is also the fast case."""
        calls = self._agent(prr, monkeypatch, json.dumps({"findings": [FINDING]}))
        assert prr.review_findings("p", "/tmp") == [FINDING]
        assert len(calls) == 1

    def test_an_empty_result_is_asked_again(self, prr, monkeypatch):
        """And a bare repeat of the shrug no longer approves: the confirmation
        pass demands evidence, so `{"findings":[]}` twice is refused rather than
        treated as agreement. Two correlated draws are not a confirmation.

        THREE calls, not two: the confirmation's evidence-less reply is itself
        unusable, so it gets the same single retry every other unusable shape
        gets. That is the point — `{"findings": []}` is this pass's most common
        non-answer, and it used to be the one shape that was never asked twice."""
        calls = self._agent(prr, monkeypatch, '{"findings":[]}')
        with pytest.raises(ScanError):
            prr.review_findings("p", "/tmp")
        assert len(calls) == 3, "review, confirmation, and the confirmation's retry"

    def test_the_second_pass_can_rescue_a_shrug(self, prr, monkeypatch):
        """THE point. A first-pass shrug followed by a real review must post the
        real review, not the approval."""
        calls = self._agent(prr, monkeypatch,
                            '{"findings":[]}',
                            json.dumps({"findings": [FINDING]}))
        assert prr.review_findings("p", "/tmp") == [FINDING]
        assert len(calls) == 2

    def test_an_unreadable_second_pass_refuses_to_approve(self, prr, monkeypatch):
        """One empty result plus one unreadable one is not evidence of clean
        code. Raising reaches guard_main, which is loud; approving would not be."""
        self._agent(prr, monkeypatch, '{"findings":[]}', "I could not review this.")
        with pytest.raises(ScanError) as e:
            prr.review_findings("p", "/tmp")
        assert "unconfirmed" in str(e.value)

    def test_a_failing_first_pass_still_raises(self, prr, monkeypatch):
        """Unchanged: a first pass that cannot be read was never an approval."""
        self._agent(prr, monkeypatch, "not json at all")
        with pytest.raises(ScanError):
            prr.review_findings("p", "/tmp")


class TestConfirmationShowsItsWork:
    """The second pass asks a DIFFERENT question, so two empties are evidence
    rather than a second roll of the same dice."""

    def _agent(self, prr, monkeypatch, *replies):
        seen = []

        # `**k`: this stub records what was ASKED, not how it was called —
        # pinning the arity broke it the moment run_agent gained a timeout.
        def fake(prompt, work, **k):
            seen.append(prompt)
            return replies[min(len(seen) - 1, len(replies) - 1)]
        monkeypatch.setattr(prr, "run_agent", fake)
        return seen

    def test_the_confirmation_prompt_is_not_the_original(self, prr, monkeypatch):
        """A re-run of the same prompt is a correlated draw. The shrug is a state
        the model gets into, not independent noise, so asking again the same way
        does not divide the risk the way independence would."""
        seen = self._agent(prr, monkeypatch, '{"findings":[]}',
                           '{"findings":[],"checked":[{"file":"a.py","verified":"x"}]}')
        prr.review_findings("ORIGINAL-PROMPT", "/tmp")
        assert len(seen) == 2
        assert seen[0] != seen[1], "the second pass must ask a different question"
        assert "SHOW YOUR WORK" in seen[1]
        assert "ORIGINAL-PROMPT" in seen[1], "it still needs the change under review"

    def test_an_approval_must_say_what_it_checked(self, prr, monkeypatch):
        """`findings: []` is producible by a model that read nothing.
        `checked: [...]` is not — that is the artifact a shrug cannot fake."""
        self._agent(prr, monkeypatch, '{"findings":[]}', '{"findings":[]}')
        with pytest.raises(ScanError) as e:
            prr.review_findings("p", "/tmp")
        assert "without saying what it checked" in str(e.value)

    @pytest.mark.parametrize("checked", ['[]', 'null', '"a.py"', '{}'])
    def test_an_empty_or_wrong_shaped_checked_is_refused(self, prr, monkeypatch, checked):
        self._agent(prr, monkeypatch, '{"findings":[]}',
                    '{"findings":[],"checked":%s}' % checked)
        with pytest.raises(ScanError):
            prr.review_findings("p", "/tmp")

    def test_evidence_approves(self, prr, monkeypatch):
        self._agent(prr, monkeypatch, '{"findings":[]}',
                    '{"findings":[],"checked":[{"file":"a.py","verified":"guard holds"}]}')
        assert prr.review_findings("p", "/tmp") == []

    def test_the_second_pass_can_still_find_what_the_first_missed(self, prr, monkeypatch):
        """Findings win over evidence — a second pass that finds something real
        reports it rather than approving."""
        self._agent(prr, monkeypatch, '{"findings":[]}',
                    json.dumps({"findings": [FINDING], "checked": []}))
        assert prr.review_findings("p", "/tmp") == [FINDING]


class TestUnreadableReplyIsRetried:
    """A reply that cannot be parsed loses the whole review.

    The job exits non-zero, nothing is posted, and the PR shows a red check with
    no findings — recovering costs a human noticing and re-requesting. That is
    not hypothetical: tests#289's review completed, produced 2145 characters, and
    was thrown away; re-running the same review by hand parsed cleanly, so it is
    a transient rather than a format we cannot read.
    """

    GOOD = '{"findings": [{"severity": "high", "title": "t", "file": "a.py", "detail": "d"}]}'
    BAD = "I looked at this and, honestly, prose."

    def _agent(self, prr, monkeypatch, *replies):
        calls = []

        # `**k`: this stub records what was ASKED, not how it was called —
        # pinning the arity broke it the moment run_agent gained a timeout.
        def fake(prompt, work, **k):
            calls.append(prompt)
            return replies[min(len(calls) - 1, len(replies) - 1)]
        monkeypatch.setattr(prr, "run_agent", fake)
        return calls

    def test_a_clean_reply_is_not_asked_twice(self, prr, monkeypatch):
        calls = self._agent(prr, monkeypatch, self.GOOD)
        assert len(prr.review_findings("p", "/tmp")) == 1
        assert len(calls) == 1, "a readable reply must cost exactly one agent run"

    def test_an_unreadable_reply_is_asked_again(self, prr, monkeypatch):
        """THE fix. The second reply is the review that gets posted."""
        calls = self._agent(prr, monkeypatch, self.BAD, self.GOOD)
        assert len(prr.review_findings("p", "/tmp")) == 1
        assert len(calls) == 2

    def test_two_unreadable_replies_still_raise(self, prr, monkeypatch):
        """Twice in a row is not a blip. A reviewer that cannot say what it found
        must be loud, not silent — and must never look like a clean review."""
        calls = self._agent(prr, monkeypatch, self.BAD, self.BAD)
        with pytest.raises(ScanError):
            prr.review_findings("p", "/tmp")
        assert len(calls) == 2, "it must not keep asking forever"

    def test_the_retry_asks_the_same_review_plus_the_correction(
            self, prr, monkeypatch):
        """This asserted `calls[0] == calls[1]` — the retry sends the prompt
        UNCHANGED — and that is the behaviour that produced the 2026-08-31
        caeli-marketing outage: the model made a mechanical JSON slip, was asked
        the identical question, and made it again. A test can pin a bug as
        firmly as a feature.

        The intent behind it is still right and still checked: the retry is a
        re-roll of an unreadable ANSWER, not a different REVIEW. So the question
        must be the same question, with only a correction appended.
        """
        calls = self._agent(prr, monkeypatch, self.BAD, self.GOOD)
        prr.review_findings("ORIGINAL", "/tmp")
        assert calls[0] == "ORIGINAL", "the first ask must be unadorned"
        assert calls[1].startswith("ORIGINAL"), "the retry changed the review"
        assert "REJECTED" in calls[1], "the retry is a blind re-roll again"

    def test_the_confirmation_pass_is_retried_too(self, prr, monkeypatch):
        """An unreadable CONFIRMATION is the dangerous one — it sits on the path
        that would otherwise approve."""
        calls = self._agent(
            prr, monkeypatch,
            '{"findings":[]}', self.BAD,
            '{"findings":[],"checked":[{"file":"a.py","verified":"x"}]}')
        assert prr.review_findings("p", "/tmp") == []
        assert len(calls) == 3, "review, unreadable confirmation, retried confirmation"
        assert "SHOW YOUR WORK" in calls[2]


class TestEveryUnusableShapeIsRetried:
    """The retry must fire for every reply that cannot be used, not only for the
    ones with no JSON at all.

    Found by the reviewer on the PR that added the retry: `parse_findings` is
    lenient by design — its fallback returns ANY top-level object and raises only
    when nothing decodes — so gating on it alone covered "plain prose" and missed
    `{}`, which is the model's defeat-shrug and the COMMON garbled shape. The
    retry existed for the rare case and skipped the frequent one.
    """

    GOOD = '{"findings": [{"severity": "low", "title": "t", "file": "a.py", "detail": "d"}]}'

    def _agent(self, prr, monkeypatch, *replies):
        calls = []

        def fake(prompt, work, **k):
            calls.append(prompt)
            return replies[min(len(calls) - 1, len(replies) - 1)]
        monkeypatch.setattr(prr, "run_agent", fake)
        return calls

    @pytest.mark.parametrize("bad", [
        "{}",                                        # the shrug — parses, unusable
        '{"checked":[{"file":"a.py"}]}',             # answered a different question
        '{"findings": null}',                        # null
        '{"findings": {}}',                          # wrong container
        '{"findings": ["oops"]}',                    # non-object element
        "I had a look and it seems fine.",           # no JSON at all
    ], ids=["empty-obj", "wrong-key", "null", "dict", "non-object", "prose"])
    def test_each_unusable_shape_is_asked_again(self, prr, monkeypatch, bad):
        calls = self._agent(prr, monkeypatch, bad, self.GOOD)
        assert len(prr.review_findings("p", "/tmp")) == 1
        assert len(calls) == 2, f"{bad!r} was not retried"

    def test_a_genuinely_empty_review_is_not_treated_as_unusable(self, prr, monkeypatch):
        """`{"findings": []}` is a real answer — the confirmation pass handles it,
        and it must not be burned as an unusable reply here."""
        calls = self._agent(
            prr, monkeypatch, '{"findings":[]}',
            '{"findings":[],"checked":[{"file":"a.py","verified":"x"}]}')
        assert prr.review_findings("p", "/tmp") == []
        assert len(calls) == 2, "the second call must be the CONFIRMATION, not a retry"
        assert "SHOW YOUR WORK" in calls[1]


class TestRetryRespectsTheJobBudget:
    def test_it_declines_to_retry_without_time_to_finish(self, prr, monkeypatch):
        """Two 900s runs exceed the job's 25-minute cap, so a retry started too
        late loses the review to the JOB timeout instead of the parse — the same
        outcome, plus a long run holding the single runner."""
        monkeypatch.setattr(prr, "run_agent", lambda p, w, **k: "not json")
        monkeypatch.setattr(prr, "_remaining_budget", lambda: 5)
        with pytest.raises(ScanError) as e:
            prr.review_findings("p", "/tmp")
        assert "not retrying" in str(e.value)
        assert "5s" in str(e.value), "it should say how little was left"

    def test_the_retry_is_capped_to_what_is_left(self, prr, monkeypatch):
        seen = []

        def fake(prompt, work, timeout=None):
            seen.append(timeout)
            return "not json" if len(seen) == 1 else '{"findings":[]}'
        monkeypatch.setattr(prr, "run_agent", fake)
        monkeypatch.setattr(prr, "_remaining_budget", lambda: 300)
        with pytest.raises(ScanError):
            prr.review_findings("p", "/tmp")   # empty -> confirmation -> no evidence
        assert seen[1] == 300, f"retry ran with {seen[1]}, not the remaining budget"


class TestConfirmationShrugIsRetried:
    """The confirmation pass's own most common non-answer must be asked again.

    Found by the reviewer on the retry PR itself. `_usable` originally checked
    only the `findings` shape, and the `checked`-evidence requirement lived AFTER
    the call — so a confirmation replying `{"findings":[]}` with no evidence was
    "usable", returned without a retry, and rejected a moment later. The rare
    garbles (`{}`, prose) were retried; the frequent one was not, for the same
    defeat plus one key.
    """

    def _agent(self, prr, monkeypatch, *replies):
        calls = []

        def fake(prompt, work, **k):
            calls.append(prompt)
            return replies[min(len(calls) - 1, len(replies) - 1)]
        monkeypatch.setattr(prr, "run_agent", fake)
        return calls

    EVIDENCE = '{"findings":[],"checked":[{"file":"a.py","verified":"the guard holds"}]}'

    def test_an_evidence_less_confirmation_is_asked_again(self, prr, monkeypatch):
        """THE case: the retry rescues it, and the review is approved on the
        second confirmation rather than thrown away."""
        calls = self._agent(prr, monkeypatch, '{"findings":[]}', '{"findings":[]}',
                            self.EVIDENCE)
        assert prr.review_findings("p", "/tmp") == []
        assert len(calls) == 3
        assert "SHOW YOUR WORK" in calls[2], "the retry must re-ask the confirmation"

    def test_a_confirmation_that_finds_something_needs_no_evidence(self, prr, monkeypatch):
        """Findings ARE the evidence. Demanding `checked` alongside them would
        retry a perfectly good review that just happened to find something on the
        second look."""
        calls = self._agent(prr, monkeypatch, '{"findings":[]}',
                            json.dumps({"findings": [FINDING]}))
        assert prr.review_findings("p", "/tmp") == [FINDING]
        assert len(calls) == 2, "a confirmation with findings must not be retried"

    def test_the_review_pass_does_not_demand_evidence(self, prr, monkeypatch):
        """Only the confirmation carries that bar. A first pass that finds
        nothing is meant to fall through to the confirmation, not be retried as
        though it were malformed."""
        calls = self._agent(prr, monkeypatch, '{"findings":[]}', self.EVIDENCE)
        assert prr.review_findings("p", "/tmp") == []
        assert len(calls) == 2, "the first empty result was retried instead of confirmed"


class TestInfraFailureIsNotAContentFailure:
    """hermes not answering is a different thing from hermes answering badly.

    Found by the reviewer on the retry PR. `run_agent` raises on any non-zero
    hermes exit, so a provider outage was being classified as "reply was
    unusable", re-rolled at full price against the down provider, and then
    reported with the wrong cause. CLAUDE.md records the 2026-08-11 Fireworks
    event (503 on 13 of 15 probes) as the shape that does this for real.
    """

    def test_a_process_failure_is_not_retried(self, prr, monkeypatch):
        calls = []

        def dead(prompt, work, **k):
            calls.append(1)
            raise prr.AgentFailed("hermes exited 1: connection refused")
        monkeypatch.setattr(prr, "run_agent", dead)
        with pytest.raises(prr.AgentFailed):
            prr.review_findings("p", "/tmp")
        assert len(calls) == 1, "an outage was re-rolled against the same dead provider"

    def test_the_cause_survives_to_the_alert(self, prr, monkeypatch):
        """guard_main puts this straight into Telegram. "reply was unusable" for
        a dead provider sends triage at the wrong thing — the attribution
        discipline this repo keeps insisting on."""
        monkeypatch.setattr(prr, "run_agent",
                            lambda p, w, **k: (_ for _ in ()).throw(
                                prr.AgentFailed("hermes exited 1: provider 503")))
        with pytest.raises(prr.AgentFailed) as e:
            prr.review_findings("p", "/tmp")
        assert "provider 503" in str(e.value)
        assert "unusable" not in str(e.value)

    def test_an_unusable_ANSWER_is_still_retried(self, prr, monkeypatch):
        """The distinction must not cost the feature: bad content still re-rolls."""
        calls = []

        def fake(prompt, work, **k):
            calls.append(1)
            return "prose" if len(calls) == 1 else \
                '{"findings":[{"severity":"low","title":"t","file":"a","detail":"d"}]}'
        monkeypatch.setattr(prr, "run_agent", fake)
        assert len(prr.review_findings("p", "/tmp")) == 1
        assert len(calls) == 2


class TestEveryAttemptIsBudgeted:
    def test_the_first_attempt_is_capped_too(self, prr, monkeypatch):
        """Budgeting only the retry left the arithmetic wrong where it matters:
        a 900s review plus a 900s confirmation is ~1800s against a 1410s budget,
        so Actions kills the process and the `left < 120` gate never fires."""
        seen = []

        def fake(prompt, work, timeout=None):
            seen.append(timeout)
            return '{"findings":[{"severity":"low","title":"t","file":"a","detail":"d"}]}'
        monkeypatch.setattr(prr, "run_agent", fake)
        monkeypatch.setattr(prr, "_remaining_budget", lambda: 240)
        prr.review_findings("p", "/tmp")
        assert seen == [240], f"first attempt ran uncapped at {seen}"

    def test_the_confirmation_is_capped_by_what_the_review_left(self, prr, monkeypatch):
        seen = []

        def fake(prompt, work, timeout=None):
            seen.append(timeout)
            return '{"findings":[]}' if len(seen) == 1 else \
                '{"findings":[],"checked":[{"file":"a","verified":"x"}]}'
        monkeypatch.setattr(prr, "run_agent", fake)
        monkeypatch.setattr(prr, "_remaining_budget", lambda: 300)
        assert prr.review_findings("p", "/tmp") == []
        assert seen == [300, 300], f"confirmation ran uncapped at {seen}"

    def test_it_never_asks_for_a_nonsense_timeout(self, prr, monkeypatch):
        """A budget already blown must not produce a zero or negative timeout —
        subprocess would take that as 'expire immediately'."""
        monkeypatch.setattr(prr, "_remaining_budget", lambda: -500)
        assert prr._agent_timeout() >= 30


class TestTheFailureTaxonomyHolds:
    """AgentFailed means "hermes did not answer". Every path that can produce
    that must say so, and no path may re-label it as bad content.

    Both gaps found by the reviewer on the PR that introduced the class — one of
    them contradicting its own docstring.
    """

    def test_a_timeout_is_an_agent_failure_not_a_crash(self, prr, monkeypatch):
        """`AgentFailed` claims "the model did not answer, or the clock ran
        out", but only the first was mapped. An expired run escaped as a bare
        exception into the generic handler and was announced as "pr-review
        crashed" — wrong cause, wrong severity."""
        def never_answers(system, user, root, on_turn=None, **kw):
            raise prr.agent.Timeout("deadline of 0s exhausted after 0 turn(s)",
                                    ["turn 1: 1 tool call(s) in 0.1s"])

        monkeypatch.setattr(prr.agent, "run", never_answers)
        monkeypatch.setattr(prr, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(prr, "_superseding_run_exists", lambda *a: False)
        with pytest.raises(prr.AgentFailed) as e:
            prr.run_agent("p", "/tmp", timeout=0.05)
        assert "exhausted" in str(e.value)

    def test_a_timeout_is_not_retried_as_content(self, prr, monkeypatch):
        """It rides the infra path, so it must not be re-rolled either."""
        calls = []

        def timeout(prompt, work, **k):
            calls.append(1)
            raise prr.AgentFailed("hermes timed out after 30s")
        monkeypatch.setattr(prr, "run_agent", timeout)
        with pytest.raises(prr.AgentFailed):
            prr.review_findings("p", "/tmp")
        assert len(calls) == 1

    def test_an_outage_on_the_CONFIRMATION_keeps_its_cause(self, prr, monkeypatch):
        """The confirmation wrapped every ScanError as "could not be read", and
        AgentFailed is a ScanError — so a provider outage there was reported as
        unreadable content. The first-pass path was already correct; this one was
        the backslide, and nothing covered it."""
        calls = []

        def fake(prompt, work, **k):
            calls.append(prompt)
            if len(calls) == 1:
                return '{"findings":[]}'
            raise prr.AgentFailed("hermes exited 1: provider 503")
        monkeypatch.setattr(prr, "run_agent", fake)
        with pytest.raises(prr.AgentFailed) as e:
            prr.review_findings("p", "/tmp")
        assert "provider 503" in str(e.value)
        assert "could not be read" not in str(e.value), \
            "an outage on the confirmation was re-labelled as bad content"
class TestRenderedFindingsAreReadable:
    def test_model_escaped_backticks_do_not_reach_the_reader(self, prr):
        """The model escapes backticks as if writing a shell or JS string, and
        GitHub renders the backslash literally — so a finding that quotes code
        arrives with visible slashes through the one thing it most needs to
        show. Seen on infra#106, where every code span read `\\`${X}\\``."""
        body = prr.render([{**FINDING, "detail": r'use \`${IMG}\` here'}], False, 0)
        assert "`${IMG}`" in body
        assert "\\`" not in body

    def test_an_ordinary_backslash_survives(self, prr):
        """Only the escaped-backtick sequence is normalised — a Windows path or a
        regex in a finding must not be mangled."""
        body = prr.render([{**FINDING, "detail": r'the pattern \d+ and C:\tmp'}], False, 0)
        assert r"\d+" in body and r"C:\tmp" in body

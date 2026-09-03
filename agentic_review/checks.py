"""Findings that are ARITHMETIC, not judgement.

Anything with a fixed threshold and a deterministic answer belongs here rather
than in the prompt. Three reasons, all of them measured:

  1. The model does not do it reliably. On slack-app#378 Copilot flagged a
     112,848-char CLAUDE.md against the cap written at line 5 of that same file;
     ours ran six rounds and nine findings — including the real cross-tenant bug
     Copilot also found — and never mentioned size once.
  2. It cannot hallucinate. A byte count is a byte count.
  3. It costs no tokens, so it runs on every review including the ones where the
     agent has already spent its budget.

Every check here returns findings in the same shape the model produces, so
rendering, severity ordering and the approval decision need no special case.
"""
import os
import pathlib
import re
import subprocess

from .config import TICKET_PATTERN

#: claudelint's default. Past it, Claude Code warns about degraded performance
#: and may TRUNCATE the file — so rules at the bottom stop being read while
#: still appearing to be in force.
CLAUDE_MD_MAX = 40_000

#: The docs' adherence target. Under the byte cap the whole file is read; past
#: this it is followed less reliably. Two numbers because they fail differently.
CLAUDE_MD_MAX_LINES = 200

#: A file that states its own, stricter cap is taken at its word.
_SELF_CAP = re.compile(r"cap[^.\n]{0,20}?~?\s*(\d{2,3})\s*k\b", re.I)


def claude_md_size(work, changed_paths):
    """A convention doc the diff pushes over its cap.

    Only for a file the DIFF TOUCHES. Most repositories are already over, so
    reporting on files a PR does not touch would put an unactionable line on
    every review until somebody does a cleanup pass.

    `low`, so it never withholds approval: the size is real, and it is not a
    reason to block a change that is otherwise fine.
    """
    out = []
    for path in sorted({p for p in changed_paths
                        if p == "CLAUDE.md" or p.endswith("/CLAUDE.md")}):
        try:
            body = (pathlib.Path(work) / path).read_bytes()
        except OSError:
            continue  # deleted in this PR, or unreadable — not a size problem
        size = len(body)
        lines = body.count(b"\n") + 1
        cap, source = CLAUDE_MD_MAX, "claudelint's 40k default"
        m = _SELF_CAP.search(body[:2000].decode("utf-8", "replace"))
        if m and int(m.group(1)) * 1000 < CLAUDE_MD_MAX:
            cap = int(m.group(1)) * 1000
            source = f"this file's own stated ~{m.group(1)}k cap"
        over = []
        if size >= cap:
            over.append(f"{size:,} bytes over {cap:,} ({size / cap:.1f}x, "
                        f"{source}) — Claude Code may TRUNCATE it")
        if lines > CLAUDE_MD_MAX_LINES:
            over.append(f"{lines:,} lines over {CLAUDE_MD_MAX_LINES} "
                        f"({lines / CLAUDE_MD_MAX_LINES:.1f}x) — the docs' "
                        "adherence target")
        if not over:
            continue
        out.append({
            "severity": "low",
            "file": path,
            "line": 1,
            "title": f"{path}: " + "; ".join(over),
            "detail": (
                "Measured at this PR's head. Past the byte cap Claude Code "
                "warns about degraded performance and may truncate the file, "
                "so rules at the bottom stop being read while still appearing "
                "to be in force. Past the line target the whole file is read "
                "but followed less reliably. This is not a reason to hold the "
                "PR (hence a nit); it is a reason to move per-ticket rationale "
                "and migration history to the ticket, to push per-repo detail "
                "down into that repo's own doc, or to split into "
                "`.claude/rules/`."),
        })
    return out


_TICKET_IN_TITLE = re.compile(rf"\b{TICKET_PATTERN}\b") if TICKET_PATTERN else None


def ticket_in_title(title):
    """The PR title must name the ticket it implements.

    Not bureaucracy: the ticket is the only durable record of WHY, and a PR
    title is the string that ends up in the merge commit, the release notes and
    every future `git log` search. A change whose reason is findable in six
    months costs one prefix now; one whose reason is not costs an afternoon of
    archaeology later, and the person paying is never the person who saved the
    prefix.

    `medium`, not `high`: it withholds approval — which is the point, because a
    nit gets ignored and this is trivially fixable — but it is a process defect
    and must never outrank a real bug in the ordering.
    """
    if _TICKET_IN_TITLE is None or _TICKET_IN_TITLE.search(title or ""):
        return []
    return [{
        "severity": "medium",
        "file": "",
        "line": 0,
        "title": f"PR title does not name a ticket (expected e.g. {_example()})",
        "detail": (
            f"The title is {title!r}. Every PR should carry its tracker id so "
            "the change is traceable from `git log` and the merge commit back "
            "to the reason it was made. Edit the title — no code change is "
            "needed — then re-request this reviewer: a title edit is not a "
            "review trigger, so the finding will not clear on its own."),
    }]


def _example():
    """A concrete id in the message, derived from the configured pattern.

    A message that quotes a REGEX at somebody is a message they have to decode
    before they can act on it, and the action here is "type six characters".
    """
    return "SCRUM-1234" if TICKET_PATTERN == r"[A-Z][A-Z0-9]+-\d+" else TICKET_PATTERN


#: Text that claims an AI agent wrote the change. Deliberately narrow: matching
#: the word "claude" anywhere would fire on a PR that merely DISCUSSES Claude,
#: which is a routine thing to do in these repositories.
_AGENT_ATTRIBUTION = re.compile(
    r"(co-authored-by:\s*claude"
    r"|generated with \[?claude code"
    r"|🤖 generated with"
    r"|\bclaude opus\b|\bclaude sonnet\b)", re.I)

#: The session an agent-written commit must name. EITHER the full link OR the
#: bare id — Tingyi's call on tests#366: `session_01Egtw…` identifies the run
#: exactly as well as `https://claude.ai/code/session_01Egtw…`, and a check
#: that rejects the id because it lacks the prefix is enforcing a spelling,
#: not a record. Anchored on the `session_` prefix plus the id's alphabet so
#: the word "session" in prose cannot satisfy it.
#:
#: AND the local transcript id — a UUID under `~/.claude/projects/`, labelled.
#: Accepted on Tingyi's call (tests#366), and it is the RIGHT record for a
#: terminal session, not a fallback. The lesson that made it so: this check
#: withholds approval, and under that pressure an agent that could not find its
#: own link went and found A link — a claude.ai URL belonging to a DIFFERENT
#: session — and put it in four PR descriptions. The reviewer cannot tell a
#: borrowed link from a real one, so the only defence is to make the honest
#: answer plainly acceptable and to say that a wrong link is worse than none.
#: The UUID has to sit near a "session"/"transcript" label, because a PR body
#: is full of other UUIDs (Supabase users, order ids) that record nothing.
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_SESSION_URL = re.compile(
    r"https://claude\.ai/(?:code/)?session[_/][\w-]+"
    r"|(?<![\w/])session_[A-Za-z0-9]{8,}\b"
    r"|(?:session|transcript)[^\n]{0,60}?`?" + _UUID + r"`?"
    r"|\.claude/projects/[^\s`]*" + _UUID, re.I)


def agent_session_url(commits, pr_body=""):
    """An agent-attributed commit must link the session that produced it.

    The attribution alone says a machine wrote it; the session URL says WHICH
    RUN, and that is the difference between knowing a commit is agent-written
    and being able to go and read what the agent was told, what it tried, and
    what it was refused. When an agent-written change turns out to be wrong —
    and they do — the transcript is the only artefact that explains why, and it
    is not reconstructable after the fact.

    Checked across the whole PR: one link anywhere (any commit message, or the
    PR body) satisfies it. Requiring it per-commit would flag every fixup in a
    stack that already carries the link on its first commit, which is noise.

    `medium`. It withholds approval deliberately — the fix is one line in the
    PR body — but it never outranks a defect in the code.
    """
    texts = [c for c in commits if c] + ([pr_body] if pr_body else [])
    if not any(_AGENT_ATTRIBUTION.search(t) for t in texts):
        return []
    if any(_SESSION_URL.search(t) for t in texts):
        return []
    return [{
        "severity": "medium",
        "file": "",
        "line": 0,
        "title": "agent-written commit with no session link",
        "detail": (
            "A commit on this PR is attributed to an AI agent, but neither the "
            "commit messages nor the PR body link the session that produced it. "
            "The attribution says a machine wrote this; the link is what lets "
            "somebody read what it was asked, what it tried and what it was "
            "told — which is the only record of why, and it cannot be "
            "reconstructed later. Add a `Claude-Session: …` line to the commit "
            "message or the PR description: the claude.ai session URL if you "
            "ran there, or — for a terminal session — the local transcript id, "
            "labelled: `Agent session (local Claude Code CLI): <uuid>`. Either "
            "is accepted. DO NOT go looking for a URL you are not certain is "
            "THIS session's: a link to someone else's transcript is worse than "
            "no link, because it reads as evidence and points at the wrong "
            "run."),
    }]


def run_all(work, changed_paths, title="", commits=(), pr_body="", diff=""):
    """Every deterministic check, in severity-independent order.

    Called AFTER the agent, so the model never sees these and cannot be
    influenced into repeating or contradicting one.
    """
    return (ticket_in_title(title)
            + agent_session_url(commits, pr_body)
            + route_without_test(work, diff)
            + claude_md_size(work, changed_paths))


# --------------------------------------------------------------------------
# A new route with no test that names it
# --------------------------------------------------------------------------
# THE FINDING WE MISSED THREE TIMES OUT OF THREE. Copilot made it on
# slack-app#381 ("these tests call `upsert_company` directly, so they do not
# cover the new external CompanyPayload path"), the outgoing reviewer made it on
# #378 ("AdditiveTests cannot fail — it never exercises the code under review"),
# and BOTH made it on #380 ("the new route has no handler-level coverage,
# although the neighbouring /companies/access route is tested"). A prompt rule
# written specifically for it produced nothing on any of the three.
#
# It produced nothing because it is not a judgement. It is a lookup, and the
# model was being asked to notice an ABSENCE — which is the one thing a reader
# is worst at and a grep is best at.
#
# WHY NOT A CALL GRAPH. PyCG and friends were the obvious answer and they are
# the wrong one, measured on this repository: tests reach a route by its HTTP
# PATH STRING (`client.post("/provisioning/companies/access", …)`) and never by
# calling the handler, which the framework dispatches from a decorator. There is
# no static call edge from any test to any handler, so a call graph reports every
# route as untested — tested ones included. A 100% false-positive rate on exactly
# the check it was supposed to provide.
#
# The linkage that actually exists is the string. So that is what is checked.

#: Route registrations, across the frameworks in these repos. The path must be a
#: STRING LITERAL — a computed path cannot be matched against a test either, and
#: guessing at one is how a check starts inventing findings.
_ROUTE_RE = re.compile(
    r"""(?:^|\s|@)                              # decorator or call position
        (?:app|router|api|blueprint|bp|r)\s*
        \.\s*(get|post|put|patch|delete|route)\s*\(\s*
        (['"])(?P<path>/[^'"]*)\2""",
    re.X | re.I)

#: Where tests live. A path under any of these is a test, and a route mentioned
#: in one is covered for this check's purposes.
_TEST_DIR_RE = re.compile(r"(^|/)(tests?|spec|__tests__|e2e)(/|$)", re.I)

#: A path segment so generic that finding it in a test proves nothing, and NOT
#: finding it proves nothing either.
_TRIVIAL_ROUTE = re.compile(r"^/?(|health|healthz|ping|status|/)$")


def _added_lines(diff):
    """Lines this diff ADDS, per file. Only added routes are this PR's problem."""
    out, path = {}, None
    for line in (diff or "").splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
        elif path and line.startswith("+") and not line.startswith("+++"):
            out.setdefault(path, []).append(line[1:])
    return out


def new_routes(diff):
    """(file, path) for every route literal this PR adds outside a test file."""
    found = []
    for path, lines in _added_lines(diff).items():
        if _TEST_DIR_RE.search(path):
            continue
        for line in lines:
            m = _ROUTE_RE.search(line)
            if not m:
                continue
            route = m.group("path")
            if _TRIVIAL_ROUTE.match(route):
                continue
            if (path, route) not in found:
                found.append((path, route))
    return found


def _test_files(work):
    try:
        p = subprocess.run(["git", "-C", work, "ls-files"], capture_output=True,
                           text=True, timeout=60)
        listing = p.stdout if p.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return None                     # cannot tell — say nothing
    files = [f for f in listing.splitlines() if _TEST_DIR_RE.search(f)]
    return files or None


def route_without_test(work, diff):
    """A route this PR adds that no test file mentions by path.

    Returns nothing when the repository has no test directory at all — "you have
    no tests" is not a finding about this pull request.

    `medium`: it withholds approval, because a nit gets ignored and an untested
    new endpoint is worth one round-trip. Never `high` — the code may be
    perfectly correct, and this says only that nothing proves it.
    """
    routes = new_routes(diff)
    if not routes:
        return []
    tests = _test_files(work)
    if tests is None:
        return []
    out = []
    for file_path, route in routes:
        # The SUFFIX, because a router is mounted under a prefix: the decorator
        # says `/companies/settings` and the test calls
        # `/provisioning/companies/settings`. Matching the decorator's literal
        # inside the test's longer string is the correct direction.
        hit = False
        for test in tests:
            try:
                with open(os.path.join(work, test), encoding="utf-8",
                          errors="replace") as fh:
                    if route in fh.read():
                        hit = True
                        break
            except OSError:
                continue
        if hit:
            continue
        out.append({
            "severity": "medium",
            "file": file_path,
            "line": 0,
            "title": f"new route {route} is not named by any test",
            "detail": (
                f"This PR adds `{route}` in `{file_path}`, and no file under a "
                f"test directory mentions that path. Tests reach a route "
                f"through its URL, not by calling the handler — the framework "
                f"dispatches — so a test that exercises the service function "
                f"underneath still leaves the route's registration, its auth, "
                f"its request parsing and its response shape unverified. If a "
                f"neighbouring route is tested, that test is the shape to "
                f"copy. If this route is deliberately covered elsewhere (an "
                f"e2e suite in another repository, say), say so and this will "
                f"stop being raised."),
        })
    return out

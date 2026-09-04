#!/usr/bin/env python3
"""AI code review, driven by the hermes agent so it can READ THE REPOSITORY.

WHY AN AGENT AND NOT ONE CALL. The questions actually worth asking of a diff —
does this duplicate something we already have, does it re-implement logic a
module beside it owns, does it break a pattern the rest of the codebase keeps —
are all statements about code the diff does not contain. A single call can only
answer them from context somebody pre-selected for it, which means it answers
them from a guess. Three earlier one-shot runs produced nothing but hypotheticals
about unseen code ("if the parser normalizes severity differently…"), which is
the honest output of a reviewer that cannot look.

Measured, on a real repository: asked which scripts call one particular
helper, the agent grepped the tree and answered correctly. That traversal is the
whole reason this exists.

PR-Agent was evaluated first and is a good tool — MIT, GitHub Action, LiteLLM —
but its free tier extends diff HUNKS with surrounding lines rather than exploring
the repo; the repository-wide similar-code retrieval is the paid tier. That is
the same architecture we already had, so it would not have bought the one thing
we needed.

REASONING IS OFF, and that is measured rather than inherited. deepseek's thinking
shares the completion budget, so max_tokens is a cliff and not a cap. On a real
30k-char diff:

    none / 6k     6.8s     771 out  ->  6 findings
    low  / 6k    41.4s    6000 out  ->  NOTHING (finish=length, billed in full)
    low  / 20k   31.1s    3985 out  ->  1 finding
    high / 32k  235.3s   32000 out  ->  NOTHING
    max  / 32k  192.5s   32000 out  ->  NOTHING

Higher effort cost up to 15x and returned less, or nothing at all. The agent LOOP
is where deliberation belongs here — each tool result is grounded evidence, which
is worth more than unbounded thinking about a diff it has not looked past.

Usage:  pr-review.py <repo> <pr-number>          # posts the review
        DRY=1 pr-review.py <repo> <pr-number>     # prints it instead
"""
import base64
import hashlib
import json
from functools import lru_cache
import os
import pathlib
import re
import subprocess
import sys
import time
import tempfile
import urllib.error
import urllib.request

from . import agent
from . import checks
from . import context as ctx
from . import github
from . import llm
from . import status
from . import tracker
from .config import AGENT_TIMEOUT, MAX_DIFF, MAX_FINDINGS, ORG
from .errors import AgentFailed, PRClosed, ReviewError, Superseded


#: Generated files are volume without signal, and they dominate a diff by size.
SKIP = re.compile(
    r"(package-lock\.json|yarn\.lock|poetry\.lock|uv\.lock|bun\.lockb"
    r"|\.(png|jpe?g|gif|svg|ico|webp|woff2?|map|snap|pdf|zip|onnx|wasm)$"
    r"|/dist/|/\.output/|/node_modules/)")


def gh(path, method="GET", body=None, accept="application/vnd.github+json"):
    return github.request(path, method=method, body=body, accept=accept)


#: Paths the cap should spend itself on LAST. A test or a doc is worth
#: reviewing, but if only some of a PR fits, the source is where the defects
#: are — and a test file that fits while its subject does not produces a review
#: that assesses the proof and never sees the claim.
_LOW_PRIORITY = re.compile(
    r"(^|/)(tests?|spec|__tests__|e2e|docs?)(/|$)|\.(md|rst|txt|snap)$|"
    r"(^|/)CLAUDE\.md$", re.I)


class _Skipped(list):
    """The skipped paths, which still counts and formats as the number of them.

    `pr_diff`'s third value was an int, and three call sites — including the
    posted review's "N generated/binary files skipped" — read it as one. The
    paths are what a filter needs; the count is what the prose needs. This is
    both, so no caller had to change and none can silently get the wrong one.
    """

    def __index__(self):
        return len(self)

    def __format__(self, spec):
        return format(len(self), spec)

    def __eq__(self, other):
        if isinstance(other, int):
            return len(self) == other
        return list.__eq__(self, other)

    # NOT hashable, deliberately: it is a mutable list, and giving it a hash
    # while `__eq__` also matches an int would put `_Skipped([...])` and `2`
    # in a set as one key or two depending on the order they arrived.
    __hash__ = None


def pr_diff(repo, pr):
    """The reviewable part of the diff, and — by name — what was left out.

    Returns (diff, excluded, skipped): the diff shown to the model, the paths
    of files that did NOT fit under MAX_DIFF, and the count of generated files
    dropped on sight.

    WHOLE FILES ONLY. The old cap cut the concatenated diff at a character
    count, so the last file that fit arrived as half a hunk and the files after
    it were simply absent — the model was told "the diff was truncated" and
    never which files that meant, so it could not go and read them. Measured on
    caeli-marketing#212: 10 of 25 files reached the model, and the caveat said
    nothing about the other fifteen.

    Tingyi's call, 2026-09-02: for a big PR, review what fits and SAY WHICH
    FILES WERE NOT REVIEWED in the posted comment, rather than growing the cap
    or summarising. A separate review can be launched for the rest; the goal is
    a fast, reliable reviewer for day-to-day changes, and a partial review that
    names its gaps is honest where a summarised one is a guess.

    Source files first, then tests and docs, so the cap is spent where the
    defects are. Order within each group is git's.
    """
    raw = gh(f"/repos/{ORG}/{repo}/pulls/{pr}", accept="application/vnd.github.v3.diff")
    files, skipped_paths = [], []
    for i, chunk in enumerate(raw.split("\ndiff --git ")):
        blob = chunk if i == 0 else "diff --git " + chunk
        if not blob.strip():
            continue
        header = blob.split("\n", 1)[0]
        m = re.search(r"^\+\+\+ b/(.+)$", blob, re.M)
        path = m.group(1).strip() if m else header
        if SKIP.search(header):
            # THE PATH, not just a tally. A caller that knows only "3 files
            # were skipped" cannot tell that a changed `uv.lock` IS part of
            # this PR, and anything reasoning about which files the change
            # touches then treats it as untouched.
            skipped_paths.append(path)
            continue
        files.append((path, blob))
    files.sort(key=lambda f: bool(_LOW_PRIORITY.search(f[0])))
    kept, excluded, used = [], [], 0
    for path, blob in files:
        # Never split a file; a half hunk is worse than a named omission.
        if used + len(blob) + 1 > MAX_DIFF and kept:
            excluded.append(path)
            continue
        kept.append(blob)
        used += len(blob) + 1
    return "\n".join(kept), excluded, _Skipped(skipped_paths)


#: git stderr that means THE NETWORK, not the repository. Used only to word the
#: error — the retry below fires either way, because a permanent fault fails the
#: same way twice and costs one cheap attempt to say so.
#: How long a single fetch may take before it counts as hung.
FETCH_TIMEOUT = 300

_TRANSIENT_GIT = (
    "could not resolve host", "couldn't connect to server", "failed to connect",
    "connection reset", "connection timed out", "operation timed out",
    "temporary failure in name resolution", "the remote end hung up",
    "rpc failed", "empty reply from server", "ssl", "tls",
)


def _fetch_head(into, sha, env, attempts=2):
    """Fetch the PR head, retrying once, and NAME what went wrong.

    This is the most network-dependent step in the run, and it was the only one
    that neither retried nor explained itself. Measured on slack-app#367,
    2026-08-29 01:07:36:

        fatal: unable to access 'https://github.com/<org>/<repo>.git/':
        Failed to connect to github.com port 443 after 405 ms: Couldn't connect
        to server

    One TCP connect failed on a box whose network was fine two seconds either
    side — the reviewer fetch succeeded at 01:07:33 and the diff at 01:07:35.
    That cost the whole review.

    AND THE ALERT NAMED THE WRONG THING. `subprocess.CalledProcessError` escaped
    raw, so the page read "pr-review crashed: CalledProcessError ... exit status
    128" — which reads like a bug in the reviewer. git's stderr, the one line
    that explained it, went to the run log and never reached the alert. Two
    wrong theories (a deleted SHA, then a force-push) were chased before anyone
    read it.

    So: capture stderr, retry once, and raise a ReviewError that carries the
    reason. Everything else here already retries a transient — the agent reply,
    the judge call, the canary run — this was the omission.
    """
    last = "exit before any attempt"
    for attempt in range(attempts):
        try:
            p = subprocess.run(
                ["git", "-C", into, "fetch", "-q", "--depth", "1", "origin", sha],
                capture_output=True, text=True, timeout=FETCH_TIMEOUT, env=env)
        except subprocess.TimeoutExpired:
            # A HANG is the transient this retry most exists for, and the one
            # the first version missed: `subprocess.run` RAISES on timeout
            # rather than returning a non-zero result, so a loop inspecting
            # `returncode` never saw it. Uncaught it escapes as a bare
            # TimeoutExpired into guard_main's generic handler and is announced
            # "pr-review crashed" — the misleading page this function was
            # written to remove, reached by the other door. `_run_agent`
            # already guards its own subprocess call this way; same rule, one
            # call over.
            last = f"operation timed out after {FETCH_TIMEOUT}s waiting for github.com"
        else:
            if p.returncode == 0:
                return
            tail = (p.stderr or p.stdout or "").strip().splitlines()
            last = tail[-1].strip() if tail else f"exit {p.returncode}"
        if attempt + 1 < attempts:
            print(f"  fetch of {sha[:7]} failed ({last[:110]}) — retrying once",
                  flush=True)
            time.sleep(3)
    reason = (" — github.com was unreachable from this runner, which is a "
              "network fault rather than anything about this PR"
              if any(f in last.lower() for f in _TRANSIENT_GIT) else "")
    raise ReviewError(f"could not fetch {sha[:7]} after {attempts} attempts"
                    f"{reason}: {last[:220]}")


def checkout(repo, sha, into):
    """The PR's head, shallow. The agent needs the code AS PROPOSED — reviewing
    against main would have it 'find' the very changes under review.

    THE TOKEN NEVER TOUCHES THE URL. Embedding it (`https://x-access-token:{t}@…`)
    puts a write-scoped org credential in two places it must not be: the argv of
    `git remote add`, and then `<checkout>/.git/config`, verbatim, for as long as
    the review runs. That second one is the real problem — we hand THIS DIRECTORY
    to the agent and tell it to read whatever it needs, while the diff it is
    reasoning about is attacker-influenced content. A credential inside the
    thing being explored is a credential offered up.

    So authentication goes through `GIT_CONFIG_*`, which git reads from the
    environment: not in argv, and never written to the checkout. The remote is
    dropped afterwards regardless, so nothing about the fetch survives into the
    directory the agent sees.
    """
    token = github.token()
    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = {
        **os.environ,
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {auth}",
        # A prompt would hang until the timeout; fail fast instead.
        "GIT_TERMINAL_PROMPT": "0",
    }
    url = f"https://github.com/{ORG}/{repo}.git"
    subprocess.run(["git", "init", "-q", into], check=True, timeout=60)
    subprocess.run(["git", "-C", into, "remote", "add", "origin", url], check=True, timeout=60)
    try:
        _fetch_head(into, sha, env)
        subprocess.run(["git", "-C", into, "checkout", "-q", "FETCH_HEAD"],
                       check=True, timeout=120)
    finally:
        # Belt and braces: the extraheader lived only in the env, but a future
        # edit that reintroduces a credentialed URL should still not leave one
        # behind for the agent to read.
        subprocess.run(["git", "-C", into, "remote", "remove", "origin"],
                       check=False, timeout=60)


PROMPT = """You are reviewing a pull request in {repo}. You are IN a checkout of it
at {path} — read whatever you need. That is the point: the questions worth asking
cannot be answered from the diff alone.

THE CHANGE:
```diff
{diff}
```
{caveats}{context}
Review it. In priority order:

1. Correctness — logic errors, null/undefined, races, unhandled failure paths.
2. Security — injection, missing authz/ownership checks, secrets in code or logs,
   PII exposure.
3. Data integrity — money, rounding, migrations, anything writing state it cannot
   undo.
4. DUPLICATION AND REUSE — grep before you claim it. Does this add something the
   repo already has? Re-implement what a neighbouring module owns? NAME the
   existing thing and its path; a duplication finding without a path is a guess.
5. Consistency — the repository's own rules, quoted above, are AUTHORITATIVE.
   They record decisions that already have reasons. If the diff contradicts one,
   that is a finding and you should quote the rule. If the diff looks odd and a
   rule explains why, that is NOT a finding — say nothing.
   Intent counts here too: if a ticket is quoted above, a change that is correct
   and does not do what the ticket asked is a finding, and a more useful one
   than anything you will find in the code.
6. Tests — and check WHICH LAYER they exercise, not just that they exist.
   Three shapes, all real:
   - a behaviour change with no test at all;
   - a test that would pass with or without the fix (mutate the guard in your
     head: if the diff were reverted, would this test go red? if not, say so);
   - A TEST THAT CALLS BELOW THE CHANGE. If the diff changes what an ENTRY
     POINT accepts or forwards — a handler, a route, a message consumer, a
     published payload — a test that calls the inner helper directly proves the
     helper works and proves nothing about the contract that moved. Name the
     entry point that is untested and the regression that would slip through.
     This is the finding a diff-only reviewer makes and an exploring one
     forgets, because the test file it is reading looks thorough.

RULES:
- Point at code. Every finding needs a concrete failure: the input or state, and
  the wrong result. "Consider adding validation" is not a finding.
- VERIFY BEFORE YOU CLAIM. You can read the repo, so "if X is not defined" is not
  a finding — go and look. That was the failure mode of the reviewer this
  replaces.
- FOLLOW THE CALL; do not infer it from the name. Sibling functions often sound
  like one path and reach different services. Before asserting that code path A
  produces request B, read the body of the function that issues it.
- THE FIX IS A CLAIM TOO, and earns the same scrutiny as the defect. If your
  remedy names an existing helper, constant or pattern, OPEN IT and confirm it
  applies HERE — a guard written for a different surface is not the guard you
  want. If it moves a value between hosts, modules or branches, trace what else
  reads it. A wrong remedy is worse than none: it arrives with the authority of
  a correct diagnosis.
- SKETCH THE FIX IN AT MOST TWO LINES. Not the patch — the DIRECTION. "Re-read
  the doc inside the transaction, not before it", "guard with the same
  `_normalize` the sibling uses", "add the field to CompanyPayload so the
  provisioning path carries it". Name the helper, the field or the ordering.
  The author knows their codebase better than you do and will write the code
  faster from a direction than from a patch they have to check.
  WRITING IT OUT IN FULL IS WORSE, not better: a verbatim remedy costs the
  budget the next finding needs, cannot be applied in one click anyway (this is
  a fenced block, not a GitHub suggestion), and carries an authority it has not
  earned — this reviewer once proposed `generation_config` for a key the SDK
  calls `config`, confidently, and it read exactly like a correct fix.
  If you cannot see the direction, leave `fix` empty. "This is wrong, because
  X" is already a complete finding.
- SAY WHETHER YOU CHECKED IT. `fix_verified` is true only if you OPENED the
  helper, field or signature your sketch names and confirmed it at this head.
  Naming something from memory of how a library behaves is false.
- NAME WHAT CROSSES A WIRE. You have ONE repo. If this diff adds, removes or
  changes the MEANING of a field in an HTTP response or a message payload, list
  those field names in `wire_fields` — other repositories consume them and you
  cannot read those repositories, so the review cannot speak for them. Only
  fields whose contract actually moved; a renamed local variable is not one, and
  a padded list makes the caveat worthless. Empty is the normal answer.
- Severity describes the DEFECT, not your confidence. `high` means you can see it
  and it will bite. A doubt is at most `low`.
- No style, formatting or naming unless it changes behaviour. A linter owns those.
- No praise, no summary of the PR. The author wrote it.
- Finding nothing is a valid and useful result. An invented finding costs the
  next reader their trust in all of them.
{prior}
Reply with ONLY this JSON, no prose around it:
{{"wire_fields":["response_field_whose_contract_moved"],
"findings":[{{"file":"path","line":123,"severity":"high|medium|low",
"title":"one specific line","detail":"the concrete failure and why",
"fix":"at most two lines: the direction, naming the helper/field/ordering",
"fix_verified":true}}]}}"""


#: How much prior conversation the reviewer is shown. Generous on purpose: the
#: cost of forgetting is a reviewer that argues with its own earlier advice,
#: which is worse than any token bill and is what a 12-item cap actually bought.
CONVERSATION_BUDGET = int(os.environ.get("REVIEW_CONVERSATION_BUDGET", 120_000))


def conversation(repo, pr):
    """Its own past reviews and the author's replies. Without this it re-raises
    an answered point on every push — and tells an author who explained why
    something is deliberate that they were not heard."""
    # FOUR endpoints, because GitHub splits a PR conversation across three and
    # the fourth is not a conversation endpoint at all.
    # `/issues/{n}/comments` is only the top-level thread; a reply typed under a
    # diff line — the DEFAULT way an author answers a specific finding, and the
    # place they answer THIS tool — is a review comment and lives in
    # `/pulls/{n}/comments`. Reading only the first two presents the most common
    # "this is deliberate, here is why" as though nothing had been said, and the
    # point gets re-raised on the next push: exactly the loop this exists to end.
    #
    # COMMIT MESSAGES ARE THE FOURTH, and on some PRs they are the ONLY one.
    # An author who cannot comment as the repo owner — an agent working on
    # somebody's behalf — answers a finding in the commit that responds to it.
    # Measured on caeli-marketing#182 and tests#291: every rebuttal across five
    # rounds ("this fix would zero the funnel, here is why", "declined, with
    # four measurements") lived in a commit message, and this function returned
    # an empty conversation on every round. The block below then told the model
    # not to repeat itself while showing it nothing it had already been told.
    items = []
    for path, kind, cap in (
        # `per_page=100` ON ALL FOUR. Three of these were left at GitHub's
        # default of 30 because "they never exceed it" — but the default returns
        # the OLDEST 30, so the failure is silent and lands exactly on a
        # contested PR, where the rebuttal is comment 31.
        (f"/repos/{ORG}/{repo}/pulls/{pr}/reviews?per_page=100", "review", 1200),
        (f"/repos/{ORG}/{repo}/pulls/{pr}/comments?per_page=100", "inline", 800),
        (f"/repos/{ORG}/{repo}/issues/{pr}/comments?per_page=100", "comment", 800),
        (f"/repos/{ORG}/{repo}/pulls/{pr}/commits?per_page=100", "commit", 1500),
    ):
        try:
            for c in json.loads(gh(path)):
                if kind == "commit":
                    # A commit is shaped differently: the prose is under
                    # `commit.message`, and the author is the committer rather
                    # than a `user`. Its date must come from the commit too —
                    # `created_at` is absent, and defaulting to "" would sort
                    # every commit before every comment and scramble the order
                    # the block below depends on.
                    detail = c.get("commit") or {}
                    body = (detail.get("message") or "").strip()
                    who = ((detail.get("author") or {}).get("name")
                           or (c.get("author") or {}).get("login") or "?")
                    when = (detail.get("author") or {}).get("date") or ""
                    if body:
                        items.append((when, f"[{who} — commit]\n{body[:cap]}"))
                    continue
                body = (c.get("body") or "").strip()
                if not body:
                    continue  # a bare APPROVE carries no argument
                who = (c.get("user") or {}).get("login", "?")
                where = f" on {c['path']}" if kind == "inline" and c.get("path") else ""
                tag = c.get("state", kind) if kind == "review" else kind
                # `submitted_at` FIRST: a review object carries that and never
                # `created_at`, so keying on `created_at` alone gave every
                # review the empty string. They then sorted as one undated
                # block ahead of everything dated — measured on infra#106, the
                # model was handed six reviews followed by seven commits
                # instead of the argument they actually form. That destroys the
                # adjacency this whole function exists for: a finding and the
                # commit answering it end up in different halves of the text.
                # The comment endpoints carry no `submitted_at`, so the chain
                # is safe for them.
                items.append((c.get("submitted_at") or c.get("created_at") or "",
                              f"[{who} — {tag}{where}]\n{body[:cap]}"))
        except Exception as e:
            # One endpoint failing must not discard the other two — losing the
            # whole conversation is what makes the tool repeat itself.
            print(f"[pr-review] could not read {kind}s: {type(e).__name__}")
    # Chronological, then the most recent — an argument reads in order, and the
    # newest exchange is the one a re-review is answering.
    # A CHARACTER BUDGET, NOT AN ITEM COUNT — and it took a self-contradiction
    # to notice. caeli-marketing#212 had 54 conversation items (16 reviews, 6
    # inline, 16 comments, 16 commits) and this handed the model TWELVE. Round
    # 13 told the author to remove an install CTA that round 12 had told them to
    # widen; the words "Safari", "Firefox" and "same defect" from round 12's
    # advice were all outside the cut. The reviewer did not contradict itself
    # knowingly — it was never shown what it had said.
    #
    # Twelve items was ~12,000 characters against a model that holds 1,048,576
    # TOKENS. The scarce thing here is not context, it is coherence.
    #
    # Newest first while filling, because if anything must be dropped it is the
    # oldest — then re-sorted into order, because an argument reads forwards and
    # a finding must sit next to the commit that answered it.
    chosen, used = [], 0
    for stamp, text in sorted(items, key=lambda x: x[0], reverse=True):
        if used + len(text) > CONVERSATION_BUDGET and chosen:
            break
        chosen.append((stamp, text))
        used += len(text)
    dropped = len(items) - len(chosen)
    out = [text for _, text in sorted(chosen, key=lambda x: x[0])]
    if dropped:
        print(f"  conversation: {len(chosen)} of {len(items)} items "
              f"({used:,} chars); {dropped} older item(s) dropped", flush=True)
    if not out:
        return ""
    return ("\nALREADY SAID ON THIS PR — do not repeat yourself, and do not overrule\n"
            "the author. A reasoned rejection is a DECISION, not an open defect. If\n"
            "you still disagree, say so ONCE, acknowledge their reason, and say what\n"
            "it does not cover. Commit messages count: an author who cannot comment\n"
            "here answers you there, and a fix declined WITH A REASON is answered.\n\n"
            + "\n\n".join(out) + "\n")


#: The sha a past review says it READ, recovered from its own body.
#:
#: NOT `commit_id`: GitHub stamps that with the head at POST time, so a push
#: that lands mid-run re-attributes a review to code it never saw (measured on
#: slack-app#348). The body is the honest record — every review renders one of
#: these two forms — and `commit_id` is only the fallback for a review posted
#: before they existed.
_READ_AT = re.compile(r"(?:It read|read the change at) `([0-9a-f]{7,40})`")

#: Paths listed before the tail is summarised. Long enough for a normal round
#: of fixes, short enough that a rebase onto a moved base cannot flood the
#: prompt with the whole PR.
MAX_SINCE_PATHS = 40


def _last_review_read(mine):
    """The sha the most recent of our own reviews looked at, or ""."""
    if not mine:
        return ""
    last = max(mine, key=lambda r: r.get("submitted_at") or "")
    m = _READ_AT.search(last.get("body") or "")
    return m.group(1) if m else (last.get("commit_id") or "")


def changed_since_last_review(repo, pr, head_sha, pr_paths):
    """Which of this PR's files have moved since we last looked. Never raises.

    THE DIFF IS ALWAYS THE WHOLE PULL REQUEST. Round two is handed A, B, C and
    D with nothing saying that C and D arrived after round one — the model has
    to infer it from the order of the conversation block, and a truncated older
    review body takes even that away. So say it outright.

    It does not license skipping A and B: a later commit can break code that
    was fine when it was read, and the round still reviews everything. What it
    changes is where the attention goes first, and it stops the reviewer
    presenting an already-answered file as though it were new material.

    INTERSECTED WITH THE PR'S OWN PATHS. `compare` between two heads of the
    same branch also carries whatever arrived from the base when the author
    merged main — those files really did change in the tree, and they are
    nobody's new work. The PR diff is `merge-base(base, head)..head`, so they
    are absent from it, and intersecting removes exactly that noise.
    """
    try:
        revs = json.loads(gh(f"/repos/{ORG}/{repo}/pulls/{pr}/reviews?per_page=100"))
    except Exception as e:  # noqa: BLE001 — context, never a reason to stop
        print(f"[pr-review] could not read prior reviews for the since-list: "
              f"{type(e).__name__}")
        return ""
    me = _me()
    # `isinstance` BEFORE `.get`: this is context, and context may not kill a
    # review. A payload that is not a list of objects — an error body, a stub,
    # a future API shape — used to raise an AttributeError out of here, past
    # the narrow `except` around the fetch, and take the whole run with it.
    mine = [r for r in revs
            if isinstance(r, dict)
            and (r.get("user") or {}).get("login") == me] if me else []
    old = _last_review_read(mine)
    # A first review has no "since", and an abbreviated sha that prefixes the
    # head means the last review read this very commit.
    if not old or (head_sha or "").startswith(old):
        return ""
    try:
        cmp_ = json.loads(gh(f"/repos/{ORG}/{repo}/compare/{old}...{head_sha}"))
    except Exception as e:  # noqa: BLE001 — a force-push makes `old` unreachable
        print(f"[pr-review] could not compare {old[:7]}..{(head_sha or '')[:7]} "
              f"({type(e).__name__}) — reviewing without a since-list")
        return ""
    want = {os.path.normpath(p) for p in (pr_paths or []) if p}
    rows = []
    for f in (cmp_.get("files") if isinstance(cmp_, dict) else None) or []:
        if not isinstance(f, dict):
            continue
        name = f.get("filename") or ""
        if name and os.path.normpath(name) in want:
            rows.append((name, f.get("status") or "changed"))
    if not rows:
        return ""
    rows.sort()
    shown = ", ".join(f"`{n}` ({s})" for n, s in rows[:MAX_SINCE_PATHS])
    if len(rows) > MAX_SINCE_PATHS:
        shown += f", and {len(rows) - MAX_SINCE_PATHS} more"
    print(f"  since `{old[:7]}`: {len(rows)} of this PR's file(s) changed",
          flush=True)
    return ("\nNEW SINCE YOUR LAST REVIEW at `" + old[:7] + "` — the diff above is "
            "the WHOLE\npull request; these are the only parts of it that have moved "
            "since you\nlast looked: " + shown + ".\nEverything else you have already "
            "reviewed once. Read the new material first,\nthen re-check what it could "
            "have broken — a later commit can break code\nthat was correct when you "
            "read it.\n")


REVIEWER_SYSTEM = """You are a senior engineer reviewing a colleague's pull
request. You are sitting IN a checkout of the repository and you have three
tools: read_file, grep and list_files. Use them.

The reason you have them is that the questions worth asking of a diff are all
questions about code the diff does not contain: does this duplicate something
the repo already has, does it re-implement what a neighbouring module owns, does
it break a pattern this codebase keeps, is the helper it calls shaped the way
this caller assumes. A reviewer who cannot look those up can only guess, and a
guess dressed as a finding costs the author more time than it saves.

So: before you assert anything about code outside the diff, OPEN IT. When you
have finished looking, answer with the JSON you were asked for and nothing
else."""


def commit_messages(repo, pr):
    """Every commit message on the PR. Never raises.

    Read for the deterministic checks rather than for the model: an
    agent-attributed commit that links no session is a fact about the messages,
    not a judgement about the code.
    """
    try:
        return [((c.get("commit") or {}).get("message") or "")
                for c in json.loads(
                    gh(f"/repos/{ORG}/{repo}/pulls/{pr}/commits?per_page=100"))]
    except Exception as e:  # noqa: BLE001 — context, never a reason to stop
        print(f"[pr-review] could not read commits: {type(e).__name__}: {e}")
        return []


def build_context(repo, pr, meta, work, changed, diff, excluded=()):
    """Everything the reviewer is TOLD, as opposed to what it can go and find.

    The order is deliberate and it is the order a human would read in: the
    rules, then what was asked for, then what it connects to, then the map. The
    diff comes before all of it in the prompt — the change is the subject, and
    context that arrives before the subject reads as the subject.

    Nothing here may raise. Every source is optional and every one of them can
    be down; a review with less context is worth having, and a review that did
    not happen because a tracker 500'd is not.
    """
    title, body = meta.get("title") or "", meta.get("body") or ""
    ticket_section = ""
    keys = tracker.ticket_ids(title, body)[:tracker.MAX_TICKETS]
    if keys:
        print(f"  tickets: {', '.join(keys)}"
              + ("" if tracker.available() else " (tracker not configured)"),
              flush=True)
    tickets = [tracker.fetch(k) for k in keys]
    ticket_section = tracker.render([t for t in tickets if t])

    # Ticket text is searched for PR links too: the other half of a paired
    # change is named in the ticket at least as often as in the PR body.
    ticket_text = "\n".join(
        (t.get("description") or "") + " " +
        " ".join(c["body"] for c in t.get("comments") or [])
        for t in tickets if t)
    # The PR body gets both forms; ticket prose gets URLs only. `#3` in a Jira
    # description is a heading or an ordinal, not a pull request.
    refs = ctx.linked_pr_refs(repo, github_texts=[body],
                              url_only_texts=[ticket_text])
    # Never fetch OURSELVES as context for ourselves — `#<n>` appears in a PR's
    # own body more often than you would think, and the result reads as though
    # the change had a mysterious duplicate.
    linked_section = ctx.linked_prs(refs, gh, skip={(repo, int(pr))})
    if refs:
        print(f"  linked PRs: "
              + ", ".join(f"{r}#{n}" for r, n in refs[:ctx.MAX_LINKED_PRS]),
              flush=True)
    ci_section = ctx.check_results(repo, meta["head"]["sha"], gh,
                                   fetch_log=github.job_log)
    if ci_section:
        print("  ci: " + ("pending" if "still running" in ci_section else "results")
              + " on the head commit", flush=True)
    # BEFORE the greps, not after: up to MAX_XREF_NAMES subprocesses run here,
    # each with its own timeout, and a silent four minutes reads as a hang.
    names = ctx.xref_names(diff)
    if names:
        print(f"  cross-refs: searching {min(len(names), ctx.MAX_XREF_NAMES)} "
              f"name(s) used by this change", flush=True)
    xref_section = ctx.cross_references(work, diff, changed,
                                        also_changed=excluded)
    if xref_section:
        # The row marker, not every bullet: the "list cut" line is a bullet too,
        # and counting it reported one name more than the section carries.
        print(f"  cross-refs: {xref_section.count('` also in:')} name(s) also "
              "used outside the diff", flush=True)
    return (ctx.build(work, changed, ticket_section, linked_section, xref_section)
            + ci_section)


def run_agent(prompt, cwd, timeout=None):
    """The review agent: our own tool loop, not a subprocess.

    This used to shell out to `hermes -p cron -z <prompt>`, and the reason it no
    longer does is recorded in `cron/docs/native-reviewer-plan.md`. The short
    version: `-z` writes only the final answer, emits nothing on stderr and logs
    no tool calls, so when four reviews died at exactly the 901s cap on
    2026-09-01 there was nothing to read — while healthy runs on the same branch
    finished in 147-259s with nothing in between. A loop we own prints the turn
    it is on, so the same failure now names itself.
    """
    started = time.monotonic()
    try:
        return _run_agent(prompt, cwd, timeout or AGENT_TIMEOUT,
                          repo=_CURRENT["repo"], pr=_CURRENT["pr"])
    finally:
        print(f"  agent finished in {time.monotonic() - started:.0f}s", flush=True)


#: Set once in `main` so `_run_agent` can ask about its own PR without every
#: caller threading it through. A module global rather than a parameter because
#: `run_agent` is monkeypatched in the A/B harnesses, and adding required
#: arguments to it has broken those twice.
_CURRENT = {"repo": "", "pr": "", "wire_fields": [], "stats": {}}


#: The CALLER's filename. This is the identity that matters, and it is not the
#: obvious one: the reusable workflow (`pr-review.yml`, name "AI PR review")
#: gets NO run entry of its own. Only the caller appears, and it appears under
#: the reviewed repo. Verified against the live API rather than reasoned about —
#: every run of ours comes back as:
#:
#:     name=PR review  path=.github/workflows/pr-review-caller.yml
#:
#: Matched on PATH first because `name:` is cosmetic and one repo editing it
#: would silently stop that repo's supersession from being recognised. The name
#: is kept as a fallback for the same reason in reverse — a rename of the FILE
#: is the more deliberate act and more likely to come with a fix.
CALLER_PATH = ".github/workflows/pr-review-caller.yml"
CALLER_NAME = "PR review"


def _is_our_workflow(run):
    """Is this run OUR reviewer, rather than some other workflow on the repo?"""
    return (run.get("path") == CALLER_PATH) or (run.get("name") == CALLER_NAME)


#: How often to ask whether the PR is still there, while the agent runs.
#: Long enough to be free against a multi-minute review, short enough that a
#: merge does not buy many more minutes of a runner nobody is waiting on.
PR_STATE_POLL = 60


def _pr_is_gone(repo, pr):
    """"merged"/"closed" if there is no longer a PR to review, else None.

    UNANSWERABLE MEANS CARRY ON. A 5xx or an expired token must not abort a
    review in progress — the failure this predicate prevents is wasted work,
    and killing a good review to avoid wasted work is a worse trade than the
    waste. Same direction as every other guard in this file.
    """
    if not repo or not pr:
        return None
    try:
        m = json.loads(gh(f"/repos/{ORG}/{repo}/pulls/{pr}"))
    except Exception as e:  # noqa: BLE001
        print(f"[pr-review] could not check PR state ({type(e).__name__}) "
              "— continuing the review")
        return None
    if m.get("merged"):
        return "merged"
    if m.get("state") and m["state"] != "open":
        return m["state"]
    return None


def _superseding_run_exists(repo, pr):
    """Is a NEWER run of this workflow already queued or running for this PR?

    This is what separates "we were superseded" from "we were killed". Both
    arrive as -9. Only the first has someone else finishing the job.

    FAILS TOWARD ALERTING. If the question cannot be answered — no run id in the
    environment, the API refuses, anything unexpected — the answer is no, and the
    caller pages. Silence is the outcome this whole path exists to prevent, so an
    unknown must never buy it.
    """
    if not repo or not pr:
        return False
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not run_id or not run_id.isdigit():
        return False
    # A WILL-SKIP RUN IS NOT A SUCCESSOR.
    #
    # The route this used to describe is GONE. The group now carries the draft
    # clause, so a draft successor — `synchronize` or `review_requested` — gets
    # its own `run_id` group and cannot cancel an in-flight review at all. The
    # paragraph that stood here said the opposite ("0 mentions of draft in the
    # group"), which stopped being true the moment that clause landed.
    #
    # The guard stays, for the case the group cannot cover: our job is killed by
    # something that is NOT a superseding run — a manual cancel, an OOM, the
    # job-timeout ceiling — while the PR happens to be a draft. Treating that as
    # superseded would buy exactly the silence this predicate exists to prevent,
    # so it keeps failing toward the alert. It is now a backstop rather than the
    # front line, and that is the honest description of it.
    try:
        if bool(json.loads(gh(f"/repos/{ORG}/{repo}/pulls/{pr}")).get("draft")):
            print("[pr-review] PR is a draft, so any successor will skip the "
                  "job — not treating this as superseded")
            return False
    except Exception as e:  # noqa: BLE001 — unanswerable means "no successor"
        print(f"[pr-review] could not check draft state: {type(e).__name__}")
        return False
    try:
        runs = json.loads(gh(
            f"/repos/{ORG}/{repo}/actions/runs"
            f"?event=pull_request&per_page=40"))["workflow_runs"]
    except Exception as e:  # noqa: BLE001 — an unanswerable question is a "no"
        print(f"[pr-review] could not check for a superseding run: "
              f"{type(e).__name__}")
        return False
    mine = int(run_id)
    for r in runs:
        if r.get("id", 0) <= mine:
            continue                       # same run, or older
        if not _is_our_workflow(r):
            continue                       # a different workflow is not our successor
        if r.get("status") not in ("queued", "in_progress"):
            continue                       # finished ones cannot post for us
        if str(pr) not in [str(x.get("number")) for x in (r.get("pull_requests") or [])]:
            continue                       # a newer run on a DIFFERENT PR is irrelevant
        return True
    return False


def _run_agent(prompt, cwd, timeout=AGENT_TIMEOUT, repo="", pr=""):
    # The PR-merged check runs BETWEEN TURNS. Under hermes it had to be a poll
    # against a subprocess that was then killed, because `subprocess.run` blocks
    # until the timeout — a PR merged two minutes into a twelve-minute review
    # held the single self-hosted runner for the other ten and then posted onto
    # a merged PR. Owning the loop makes that a function call.
    # `None`, not 0.0: `time.monotonic()` counts from an arbitrary origin —
    # boot, in practice — so on a box up for less than PR_STATE_POLL the first
    # probe never fired. Found by the first CI run of this suite (SCRUM-1230):
    # green on Mini, up for hours; red on a fresh GitHub runner.
    last = {"at": None}

    def between_turns(turn):
        if not repo or not pr:
            return
        now = time.monotonic()
        if last["at"] is not None and now - last["at"] < PR_STATE_POLL:
            return
        last["at"] = now
        gone = _pr_is_gone(repo, pr)
        if gone:
            raise PRClosed(f"the PR was {gone} while the review ran")

    stats = {}
    try:
        text, _transcript = agent.run(
            REVIEWER_SYSTEM, prompt, cwd, deadline=timeout,
            on_turn=between_turns, stats=stats)
        # HOW MUCH LOOKING HAPPENED, kept where the verdict is decided. The
        # confirmation pass approves on the model's own account of what it
        # checked, and that account is worth exactly as much as the tool calls
        # behind it.
        # ACCUMULATE the opened paths, replace everything else. `stats` is
        # per-pass and the last pass wins, which is right for turn counts and
        # wrong for this: a file read during the first pass or the revision
        # would be reported as never opened because a later pass overwrote the
        # record of it.
        _CURRENT["stats"] = dict(stats)
        _merge_opened(stats)
        return text.strip()
    except agent.Timeout as e:
        _print_transcript(e.transcript)
        # A cancelled run is not a broken one. Under hermes this was inferred
        # from a signal; here the loop simply runs out of clock while GitHub is
        # already starting the review that supersedes us.
        if _superseding_run_exists(repo, pr):
            raise Superseded(str(e)) from e
        raise AgentFailed(str(e)) from e
    except agent.AgentError as e:
        _print_transcript(e.transcript)
        raise AgentFailed(str(e)) from e


def _print_transcript(transcript):
    """Put the turn log on the run page when the loop fails.

    The point of the rewrite. A failure that says only "timed out after 900s" is
    the state this replaced; a failure that says which turn, which tool and
    which arguments is a diagnosis. Best-effort — never let logging a failure
    become a second failure.
    """
    try:
        if not transcript:
            return
        print("  --- agent transcript ---", flush=True)
        for line in transcript[-60:]:
            print(f"  {line}", flush=True)
        print("  --- end transcript ---", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  (could not print transcript: {type(e).__name__}: {e})")


def parse_findings(reply):
    """The agent's reply as a dict, tolerating how an AGENT actually answers.

    `llm.parse_json_reply` is built for a json_mode completion, where the
    whole body is the object. An agent narrates: it reported what it had read
    before the JSON ("I've read the checkout. Key verification: …"), and its
    prose contained braces and quotes, so a whole-body parse died on a delimiter
    error 489 chars in — with a perfectly good object sitting at the end.

    So take the LAST object in the reply that actually answers. Last, not first,
    because the narration quotes fragments of the schema on the way past; the
    answer is what it finished with.

    THE SCAN USES THE REAL DECODER, not brace counting. Counting `{` and `}` to
    find a balanced span cannot tell a brace in CODE from a brace in a STRING,
    and this reviewer's whole job is quoting code back at you — the reply that
    broke it was a review of a workflow file, discussing `${{ … }}` expressions
    and a `concurrency.group` key. One unbalanced brace inside a quoted snippet
    and the span is wrong; the parse then fails on a reply that is perfectly
    valid JSON, and a completed review is thrown away.

    `json.JSONDecoder.raw_decode` already knows where strings end, because it is
    the parser. Trying it at each `{` costs a few hundred cheap failures on a
    30k reply and removes the entire class of bug.
    """
    try:
        return llm.parse_json_reply(reply)
    except ReviewError:
        pass
    decoder = json.JSONDecoder()
    answer = fallback = None
    # The FIRST failure at the reply's first `{` — i.e. the outermost object,
    # which is the one that was supposed to parse. Keeping it makes the error
    # say WHERE the JSON broke instead of only that it did, and that message is
    # what the retry hands back to the model.
    outer_err = None
    i, n = 0, len(reply)
    while i < n:
        if reply[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(reply, i)
        except ValueError as e:
            if outer_err is None:
                outer_err = e
            i += 1
            continue  # a brace in prose, or the start of something malformed
        # SKIP PAST what was just consumed, so only TOP-LEVEL objects are
        # candidates. Without this the scan walks back into the object it just
        # decoded, and a NESTED dict that happens to carry a `findings` key wins
        # the slot by being later — turning a real review into an empty one:
        #
        #   {"findings": [{… "meta": {"findings": []}}]}   -> APPROVE
        #
        # A silent approval on a review that did find something is the worst
        # outcome this module has, so nesting has to be excluded structurally
        # rather than hoped against.
        i = end
        if not isinstance(obj, dict):
            continue
        fallback = obj
        if "findings" in obj:
            answer = obj  # a later top-level one supersedes a quoted example
    if answer is not None:
        return answer
    if fallback is not None and outer_err is None:
        # The top-level object decoded FINE and simply is not an answer — `{}`,
        # `{"checked": [...]}`, the model's defeat-shrug. Hand it to
        # validate_findings, which declines naming what came back instead.
        return fallback
    if fallback is not None:
        # A DIFFERENT CASE, and it used to take the branch above. The outermost
        # object did NOT decode; `fallback` is a fragment scavenged from inside
        # it — one finding out of the list. Returning that produced "parsed but
        # carried no `findings` list (keys: ['detail', 'file', ...])", which
        # describes the fragment and says nothing about the syntax error that
        # cost us the real object. The 2026-08-31 caeli-marketing run failed
        # exactly here and the page never named the unescaped quote.
        at = getattr(outer_err, "pos", 0) or 0
        raise ReviewError(
            f"the agent's {len(reply)}-char reply has a malformed top-level "
            f"object: {outer_err.msg} at char {at}. Around there: "
            f"{reply[max(0, at - 90):at + 50]!r}")
    if outer_err is not None:
        at = getattr(outer_err, "pos", 0) or 0
        raise ReviewError(
            f"the agent's {len(reply)}-char reply is not valid JSON: "
            f"{outer_err.msg} at char {at}. Around there: "
            f"{reply[max(0, at - 90):at + 50]!r}")
    raise ReviewError(
        f"no parseable JSON object in the agent's {len(reply)}-char reply; "
        f"tail={reply[-200:]!r}")


def validate_findings(parsed):
    """The findings list out of a parsed reply, or raise.

    Pure, and separate from main(), because this is the one code path where a
    regression is silent AND wrong in the worst direction: an empty list means
    APPROVE, so anything that degrades to "no findings" turns a review that
    never happened into a formal approval.

    `.get("findings", [])` was the original bug — `{}`, a missing key, a null
    and a wrong-typed value all collapse to falsy, and three of those are the
    model failing to answer rather than the code being clean. Every one of them
    now declines instead.
    """
    if not isinstance(parsed, dict):
        raise ReviewError(
            f"the agent's reply parsed to {type(parsed).__name__}, not an object "
            "— refusing to post a review it did not make")
    if "findings" not in parsed or not isinstance(parsed["findings"], list):
        raise ReviewError(
            "the agent's reply parsed but carried no `findings` list "
            f"(keys: {sorted(parsed)[:6]}) — refusing to post a review it did not make")
    findings = parsed["findings"]
    # Every element must be an object. `findings: ["oops"]` clears the check
    # above and then dies in render() on `.get`, which guard_main turns into an
    # undifferentiated "pr-review crashed" — the right DIRECTION (nothing is
    # posted) reached by the wrong route, naming neither the cause nor the fix.
    # The same model demonstrably drops fields from findings, so a wrong SHAPE
    # is a live failure mode rather than a theoretical one.
    bad = [f for f in findings if not isinstance(f, dict)]
    if bad:
        raise ReviewError(
            f"{len(bad)} of {len(findings)} findings are not objects "
            f"(first: {bad[0]!r:.80}) — refusing to post a malformed review")
    # UNION, NOT REPLACE. An approval only ever comes from the CONFIRMATION
    # pass, so a replace made `_CURRENT["wire_fields"]` the confirmation's
    # answer alone — and the caveat went silent on exactly the verdict this
    # exists for. Adding the key to CONFIRM_PROMPT fixes the schema; the union
    # fixes the failure DIRECTION, because a second pass that simply forgets to
    # repeat a field would otherwise delete the caveat without a trace. A field
    # named once is worth disclosing; a lost one is silent, which is the thing
    # this module keeps being bitten by.
    _CURRENT["wire_fields"] = _merge_wire_fields(
        _CURRENT.get("wire_fields"), _wire_fields(parsed))
    return findings


def _merge_wire_fields(existing, new):
    """Everything either pass named, order-stable, deduped and capped."""
    out = list(existing or [])
    for name in new:
        if name not in out:
            out.append(name)
    return out[:8]


def _wire_fields(parsed):
    """Field names this diff moves across a wire, as the agent reported them.

    NEVER RAISES. A malformed or absent value means no caveat, because this is a
    footnote and `validate_findings` is the one path where declining costs a
    whole review. Anything that is not a list of non-empty strings is dropped.

    Capped and de-duplicated: the value is model-written and lands in markdown
    this bot posts under its own name, so it goes through `_code_span` at render
    rather than into the page.
    """
    raw = parsed.get("wire_fields")
    if not isinstance(raw, list):
        return []
    seen, out = set(), []
    for item in raw:
        name = str(item).strip() if isinstance(item, (str, int)) else ""
        if name and name not in seen:
            seen.add(name)
            out.append(name[:60])
    return out[:8]


#: Severity -> what the review DOES, not just what it says.
#:
#: The tool already classifies severity and then threw the distinction away,
#: posting COMMENT for a 🔴 and a nit alike. That is how a critical finding rode
#: along on a merge: COMMENT neither blocks nor withholds an approval, so it is
#: advisory by construction.
#:
#: With `required_approving_review_count: 1`, these become real:
#:   high    -> REQUEST_CHANGES, blocks until dismissed or superseded
#:   medium  -> COMMENT, withholds approval (blocks, but quietly)
#:   low     -> APPROVE, nits recorded in the body
#:
#: Note that medium ALSO blocks under that setting — it grants no approval. The
#: difference from high is visibility, not effect.
EVENT_BY_SEVERITY = {
    "high": "REQUEST_CHANGES",
    "medium": "COMMENT",
    "low": "APPROVE",
    # A word we were not expecting. It withholds approval, so it is NOT a nit,
    # and `normalize_severity` funnels every unrecognised value here so the
    # verdict and the rendering cannot disagree about what it means.
    "unknown": "COMMENT",
}

#: How each severity is drawn, and how findings sort. Keyed by the SAME
#: vocabulary as the events above — the two used to be independent, and drifted:
#: `review_event` treated an unknown severity as blocking while `render` drew it
#: with the blue nit icon, so a review that was withholding approval looked
#: exactly like one granting it.
ICON = {"high": "🔴", "medium": "🟡", "unknown": "⚠️", "low": "🔵"}
RANK = {"high": 0, "medium": 1, "unknown": 2, "low": 3}


def normalize_severity(value):
    """A finding's severity as one of the four words the rest of the code uses.

    One place decides what an unrecognised value means, because two places
    deciding independently is how the icon and the verdict came to disagree.
    """
    sev = str(value or "").strip().lower()
    return sev if sev in EVENT_BY_SEVERITY else "unknown"


CONFIRM_PROMPT = """A first review of this change found nothing.

That is a real possibility. It is also exactly what this model produces when it
gives up — measured on this prompt, seven of twelve replies were a bare
`{{"findings":[]}}`, several after fewer than ten output tokens. So this pass is
NOT a re-run, and answering it the same way is not useful.

Answer a different question: SHOW YOUR WORK. Open the files this change touches
and the ones around them, and record what you actually verified in each. Only
then say whether it is sound.

{original}

Reply with ONLY this JSON:
{{"wire_fields":["response_field_whose_contract_moved"],
"findings":[ …same shape as before, empty if genuinely none… ],
  "checked":[{{"file":"path","verified":"what you confirmed there"}}]}}

`checked` MUST NOT be empty. A review that cannot say what it looked at is not a
clean review, and will be rejected."""


#: The job's own cap — `timeout-minutes: 25` in pr-review.yml. Two agent runs at
#: AGENT_TIMEOUT each (900s) exceed it, so a retry started too late dies to the
#: JOB timeout instead of the parse: the same lost review, plus a long noisy run
#: holding the single runner. The retry is budgeted against what is left.
JOB_BUDGET = 25 * 60
JOB_SAFETY = 90
_STARTED = time.monotonic()


def _remaining_budget():
    return JOB_BUDGET - JOB_SAFETY - (time.monotonic() - _STARTED)


def _agent_timeout():
    """Never start a run that cannot finish inside the JOB's cap.

    Budgeting only the retry left the arithmetic wrong where it matters most: a
    review pass near its 900s cap that finds nothing, followed by a 900s
    confirmation, totals ~1800s against a 1410s budget — so Actions kills the
    process mid-confirmation and the `left < 120` gate is never reached, because
    nothing raises. A killed job loses the review AND holds the single runner,
    which is precisely what the budget was added to prevent.
    """
    return max(30, min(AGENT_TIMEOUT, _remaining_budget()))


def _usable(reply, need_evidence=False):
    """The parsed reply, or raise if it cannot be used as a review.

    `need_evidence` is the CONFIRMATION pass's extra bar: an empty result there
    must carry `checked`. It belongs here rather than after the call, because
    "unusable" has to mean the same thing to the retry as it does to the caller —
    otherwise the pass's single most common non-answer, a bare
    `{"findings": []}`, sails through as usable and is rejected a moment later
    with no second ask, while a rarer `{}` gets one.

    BOTH checks, deliberately. `parse_findings` is lenient by design — its
    fallback returns ANY top-level object and raises only when nothing decodes at
    all — so gating a retry on it alone catches "no JSON whatsoever" and nothing
    else. Measured on this module: `{}` and `{"checked": […]}` both parse, and
    both are unusable; only plain prose raises. `{}` is the model's defeat-shrug,
    i.e. the COMMON garbled shape, so a retry that skipped it would have covered
    the rare case and missed the frequent one.
    """
    parsed = parse_findings(reply)
    findings = validate_findings(parsed)
    if need_evidence and not findings:
        checked = parsed.get("checked")
        if not isinstance(checked, list) or not checked:
            raise ReviewError(
                "the confirmation approved without saying what it checked "
                f"(checked={checked!r})")
    return parsed


#: How much of a refused reply to keep from each end. The TAIL is the half that
#: matters for the commonest cause — a reply cut off mid-object — and the HEAD
#: says whether the wrapper was ever emitted. The middle is the findings the
#: review would have posted, and is the least useful part when diagnosing why
#: nothing could be read.
UNUSABLE_KEEP_CHARS = 4000


def _keep_unusable_reply(reply, reason, what=""):
    """Put a reply we REFUSED on the run page, so the next one is diagnosable.

    A refused reply used to be discarded entirely: the raise carried six key
    names and the text was gone. clientportal-prelaunch-site#33 failed twice in
    a row on 2026-08-28 — two independent agent runs, ~160s each, the same
    malformed shape — and it still could not be explained afterwards, because
    the only evidence had already been thrown away. The leading theory (a
    truncated reply, so only the inner finding objects parsed as top-level) fits
    the key set exactly and remains unproven.

    That is the same rule this module applies everywhere else, pointed at
    itself: `fix_verified` says whether the model checked, the footer names the
    commit it read and the consumers it could not, and an alert carries the
    artifact behind it. A guard that refuses without keeping what it refused
    asks the next person to re-derive the cause from nothing.

    THE RUN PAGE, not the PR. A reply that could not be parsed is an operator
    problem, and pasting a malformed model answer into a pull request tells the
    author nothing and costs them the thread.

    NEVER RAISES. It runs while a ReviewError is already in flight; a failure here
    would replace the real cause with an IO error and lose the diagnosis twice
    over.
    """
    body = str(reply or "")
    note = f"{what} reply was refused" if what else "reply was refused"
    if len(body) > 2 * UNUSABLE_KEEP_CHARS:
        omitted = len(body) - 2 * UNUSABLE_KEEP_CHARS
        body = (body[:UNUSABLE_KEEP_CHARS]
                + f"\n\n… [{omitted} chars omitted from the middle] …\n\n"
                + body[-UNUSABLE_KEEP_CHARS:])
    # Model-written text going into markdown, so the same fence discipline the
    # findings get: a reply full of code samples will contain backticks.
    fence = _fence_for(body)
    lines = [
        f"### ⚠ {note}",
        "",
        f"**Why:** {reason}",
        "",
        f"Reply was {len(str(reply or ''))} chars. Preserved so the cause can be "
        "read rather than inferred.",
        "",
        fence,
        body,
        fence,
        "",
    ]
    # TWO SURFACES, because they are not the same page. GITHUB_STEP_SUMMARY
    # renders on the run SUMMARY; stdout is what the JOB LOG shows — and the job
    # log is where a red check on the PR takes you. Preserving to the summary
    # alone means the text exists but not where anyone following the failure
    # actually lands.
    print(f"\n===== {note} — {len(str(reply or ''))} chars, verbatim below =====",
          flush=True)
    print(body, flush=True)
    print("===== end of refused reply =====\n", flush=True)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return False  # not in Actions; stdout above is the whole record
    try:
        with open(summary, "a") as f:
            f.write("\n".join(lines))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[pr-review] could not write the run-page copy: {type(e).__name__}",
              flush=True)
        return False


def _ask(prompt, work, need_evidence):
    """One ask, keeping the reply if it turns out to be unusable.

    The reply is bound to a NAME rather than passed inline. That is the whole
    change: `_usable(run_agent(...))` had nothing to preserve by the time it
    raised.
    """
    reply = run_agent(prompt, work, timeout=_agent_timeout())
    try:
        return _usable(reply, need_evidence)
    except ReviewError as e:
        _keep_unusable_reply(reply, e)
        raise


def _reply(prompt, work, what, need_evidence=False):
    """The agent's reply, asked again once if the first is unusable.

    A reply that cannot be parsed loses the whole review: the job exits non-zero,
    nothing is posted, and the PR shows a red check with no findings. Recovering
    costs a human noticing and re-requesting — which is exactly what happened to
    tests#289, where the review completed, produced 2145 characters, and was
    thrown away.

    It is a TRANSIENT, and measured as one: re-running the same review by hand
    parsed cleanly. The same content that breaks it also tends to be the content
    worth reading — reviews quoting code, which is most of them.

    So it asks once more, the same discipline `e2e_canary` uses for a failed run
    and the confirmation pass uses for an empty one. A second unreadable reply
    still raises: two in a row is not a blip, and a reviewer that cannot say what
    it found must be loud rather than silent.
    """
    try:
        return _ask(prompt, work, need_evidence)
    except AgentFailed:
        # hermes did not answer at all. Re-rolling that spends the remaining
        # budget on a provider that is down and mislabels the cause; let it out.
        raise
    except ReviewError as first:
        left = _remaining_budget()
        if left < 120:
            # Better to fail naming the real cause than to start a run that
            # cannot finish and lose the review to the job timeout instead.
            raise ReviewError(
                f"{what} reply was unusable ({first}) and only {left:.0f}s of the "
                f"job budget remained — not retrying") from first
        print(f"  {what} reply was unusable ({str(first)[:90]}) — asking once more "
              f"({left:.0f}s left)", flush=True)
        prompt = prompt + _correction(first)
    return _ask(prompt, work, need_evidence)


def _correction(err):
    """What to append to the prompt on the second ask.

    THE RETRY USED TO RE-SEND THE PROMPT UNCHANGED, which is a second draw from
    the same distribution rather than a correction — this file's own words about
    the confirmation pass, applied here. Measured on the 2026-08-31
    caeli-marketing scrum-1194 run: both draws failed the same way, and the
    second was worse. The model wrote

        "title": "... asserts "none qualify" for `check`-tier products ..."

    — an unescaped `"` inside a string value — in both replies, while escaping
    correctly inside the `fix` field of the same object. It is a mechanical
    slip, and nothing in the re-ask told it what had gone wrong, so it made the
    slip again.

    Handing back the parser's own message is the same grounded-correction the
    recipe-validator's retry loop relies on (workspace CLAUDE.md: "each retry
    feeds back a real execution error or validator diff"). It is also why the
    error above carries the decoder's position and surrounding text rather than
    just the tail: that text IS the correction.
    """
    return (
        "\n\n---\nYOUR PREVIOUS REPLY WAS REJECTED. Do not repeat it.\n\n"
        f"The parser reported: {str(err)[:700]}\n\n"
        "Send ONE JSON object and nothing else — no prose before or after, no "
        "``` fence. Inside a string VALUE, a double quote must be written as "
        '\\" and a newline as \\n. If you want to quote a phrase, prefer '
        "'single quotes' so the question does not arise. Every finding's text "
        "must survive `json.loads` unchanged.\n"
    )


#: Findings scoring at or below this are dropped. 0 is "this is wrong" in the
#: rubric below, so the default drops only what the model itself calls incorrect.
#: Raising it trades recall for noise and should be measured, not guessed.
REVISE = """Before I post that, go back over it — you have the files in front of
you, so this costs you nothing but attention.

Here is what you said, numbered:

{findings}

For EACH one, decide, and be willing to disagree with yourself:

  drop  — ONLY for these four, and OPEN THE FILE before you say so:
            (a) the code does not do what the finding says it does;
            (b) the symbol, path or field the finding names is not there;
            (c) the same case is already handled, and you can name where;
            (d) the thing the finding asks for is present at this head.
          "Deliberate", "intentional", "a product decision", "documented in a
          comment" are NOT reasons to drop — they mean the behaviour is real
          and chosen, which is `edit`. A drop says the finding is FALSE. Say
          what you opened to establish that; a drop with nothing opened will
          not be honoured.
  edit  — it is REAL but you said it wrong. The severity is off, the title
          overstates it, the mechanism is not quite what you described — or it
          is deliberate behaviour rather than a defect, in which case turn it
          into a QUESTION for the author ("is it intended that X? because Y")
          and lower it to `low`. Send back the corrected fields.
  keep  — it is right as written. Most findings that survived a careful first
          pass are; do not drop to look decisive.

Then, what did you NOTICE and NOT report? Findings get dropped for reasons that
have nothing to do with whether they are real — you were summarising, you had
written several already, one seemed small. Look again at:

  - a path in the diff that NOTHING under a test directory exercises, especially
    an entry point whose tests call the layer BELOW it;
  - a second writer to something this change writes, or a read that can now see
    a half-written state;
  - a promise in a docstring, comment or message the code no longer keeps;
  - a caller, config value or other repository this moves the ground under;
  - anything you checked, half-believed, and moved on from.

Adding nothing is a perfectly good answer to a complete first pass. Padding this
costs the author more than an empty list does.

Reply with ONLY this JSON:
{{"revisions":[{{"index":0,"action":"keep|drop|edit","why":"one line: what you
checked","severity":"high|medium|low","title":"...","detail":"...","fix":"..."}}],
"additions":[{{"file":"path","line":123,"severity":"high|medium|low",
"title":"one specific line","detail":"the concrete failure and why",
"fix":"at most two lines: the direction","fix_verified":true}}]}}

Every finding above needs exactly one revision entry, `index` matching its
number. On `edit`, send the fields you are changing; anything you omit keeps its
current value. On `keep` and `drop`, `why` alone is enough."""


def _dedupe_key(f):
    """What makes two findings the same finding.

    File and line alone are too coarse — two real defects share a line often
    enough — and the title is model prose that varies between passes, so it is
    reduced to its letters. Together they catch a restatement without merging
    genuinely different points at the same place.
    """
    title = re.sub(r"[^a-z0-9]+", "", str(f.get("title", "")).lower())[:60]
    return (str(f.get("file", "")), str(f.get("line", "")), title)


#: The revision's answer shape, enforced by the provider rather than asked for.
#:
#: `working` IS FIRST AND IT IS LOAD-BEARING. With a bare `{"revisions": [...]}`
#: schema the model answered with vacuous filler — "The finding is incorrect
#: based on the provided context or evidence" — and with a free-text field ahead
#: of the verdicts it produced 2,534 characters of real reasoning about React's
#: reconciler before deciding. Same call, same cost; the field is where the
#: thinking is allowed to happen.
#:
#: This also closes the failure that made the schema necessary. On
#: caeli-marketing#212 the revision reasoned CORRECTLY for 7,410 characters,
#: caught all three React claims the author later disputed by hand, and then
#: finished with a plain-text score list instead of JSON — so the whole
#: judgement was discarded and eight findings were posted where four should have
#: been. Reasoning in-band and a provider-enforced shape are not in tension;
#: they were only in tension because the shape was a request.
REVISION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "revision",
        "schema": {
            "type": "object",
            "properties": {
                "working": {
                    "type": "string",
                    "description": (
                        "Think it through HERE first, at whatever length you "
                        "need: for each finding, what you checked and what you "
                        "found. Write this before deciding anything below."),
                },
                "revisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "action": {"type": "string",
                                       "enum": ["keep", "drop", "edit"]},
                            "why": {"type": "string"},
                            "severity": {"type": "string"},
                            "title": {"type": "string"},
                            "detail": {"type": "string"},
                            "fix": {"type": "string"},
                        },
                        "required": ["index", "action", "why"],
                    },
                },
                "additions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string"},
                            "line": {"type": "integer"},
                            "severity": {"type": "string"},
                            "title": {"type": "string"},
                            "detail": {"type": "string"},
                            "fix": {"type": "string"},
                            "fix_verified": {"type": "boolean"},
                        },
                        "required": ["file", "line", "severity", "title",
                                     "detail"],
                    },
                },
            },
            "required": ["working", "revisions"],
        },
    },
}


def _listed(findings):
    return "\n".join(
        f"[{i}] ({f.get('severity')}) {f.get('file')}:{f.get('line')} — "
        f"{f.get('title')}\n    {str(f.get('detail'))[:600]}"
        for i, f in enumerate(findings))


def _merge_opened(stats):
    """Fold one pass's reading into the review's running record.

    Every pass reads files and every pass keeps its own `stats`, so this
    merges rather than replaces — the last pass winning would report a file
    opened by an earlier one as never opened.

    THE RANGES MERGE TOO, and that is the part that was missing: a file is
    marked opened only once its windows cover it, and the windows can land in
    DIFFERENT passes — the review pass reads lines 1-200, the revision reads
    the rest. Merging only the finished `opened` sets left that file reported
    as never opened, which is a false caveat. Found by this reviewer.
    """
    stats = stats or {}
    ranges = _CURRENT.setdefault("read_ranges", {})
    for path, seen in (stats.get("_read_ranges") or {}).items():
        into = ranges.setdefault(path, {"total": None, "covered": []})
        into["covered"].extend(seen.get("covered") or [])
        if seen.get("total") is not None:
            into["total"] = max(into["total"] or 0, seen["total"])
    opened = set(_CURRENT.get("opened") or set()) | set(stats.get("opened") or set())
    for path, seen in ranges.items():
        if agent.covers_whole_file(seen):
            opened.add(path)
    _CURRENT["opened"] = opened


def _revise(findings, work, repo):
    """One pass that can drop, correct and add — against the conversation that
    already read the code.

    THIS REPLACED TWO PASSES, and the reason is that both were asking the same
    model about the same evidence from different chairs. The old shape was a
    "second look" that could only ADD, then a separate `_reflect` that could
    only DELETE — and `_reflect` was a fresh agent loop with its own tools,
    spending ~84s re-reading files the review pass already had in context.

    Measured on caeli-marketing#212, which is why the third action exists: of
    eight findings posted, the author took one, disputed two with measurements,
    and escalated one as a PRODUCT CALL rather than a defect. Deleting that last
    one would have been wrong and keeping it as written was wrong too — it
    needed re-casting as a question. Neither old pass could do that.

    It RESUMES the review's conversation with the tools still on, so it can
    check a claim before dropping it — and because the prefix is unchanged, the
    whole thing is a cache hit.

    The risk this takes on is self-consistency: the model is judging findings it
    is invested in, without the fresh framing a separate pass gave it. The
    prompt names that directly ("be willing to disagree with yourself", "the
    action you will be most reluctant to take") because it cannot be designed
    away, only asked for and then measured.

    Never loses a review: any failure returns the originals untouched, and a
    revision list that does not cover every finding is refused wholesale — an
    unjudged finding must never be deleted.

    Returns (kept, withdrawn) with withdrawn as (finding, score, why) so the
    post can still show its work.
    """
    if not findings:
        return findings, []
    messages = (_CURRENT.get("stats") or {}).get("messages")
    if not messages:
        return findings, []
    # RESUMED WITH TOOLS ON, not asked in isolation. The most valuable action
    # here is `drop`, and "this is already handled in crud.py" or "nothing under
    # tests/ names this route" are lookups — a model that cannot look can only
    # drop what it can disprove from memory. The conversation already holds the
    # code the review pass read, so most revisions need no tool; the budget is
    # small because this is a revision, not a second review.
    stats = {}
    try:
        reply, _ = agent.resume(
            messages, REVISE.format(findings=_listed(findings)), work,
            deadline=min(300, max(60, _remaining_budget())), stats=stats,
            answer_schema=REVISION_SCHEMA)
    except (Superseded, PRClosed):
        raise
    except Exception as e:  # noqa: BLE001 — a revision must never lose a review
        print(f"  revision unavailable ({type(e).__name__}: {str(e)[:90]}) — "
              f"posting all {len(findings)} finding(s) as written", flush=True)
        _merge_opened(stats)
        return findings, []
    # BEFORE the reply is judged. `_run_agent` accumulates what its own pass
    # opened; this pass had its own `stats` and dropped them on every path,
    # including the three that return early — so a file the reviewer opened
    # while reconsidering was still reported as never opened. Whether the reply
    # parsed has nothing to do with whether the file was read.
    _merge_opened(stats)
    if not reply:
        return findings, []
    try:
        parsed = parse_findings(reply)
        revisions = parsed.get("revisions")
        if not isinstance(revisions, list):
            raise ReviewError(f"no revisions list (keys: {sorted(parsed)[:6]})")
        by_i = {}
        for r in revisions:
            if isinstance(r, dict) and isinstance(r.get("index"), int):
                by_i[r["index"]] = r
        missing = [i for i in range(len(findings)) if i not in by_i]
        if missing:
            # STRICT. A missing entry is a broken revision, not a verdict of
            # "keep" and certainly not one of "drop" — coercing either way
            # silently decides something the model never said.
            raise ReviewError(f"no revision for finding(s) {missing}")
    except ReviewError as e:
        print(f"  revision unusable ({str(e)[:100]}) — posting all "
              f"{len(findings)} finding(s) as written", flush=True)
        try:
            _keep_unusable_reply(reply, str(e), what="revision")
        except Exception as keep_error:  # noqa: BLE001
            print(f"  (could not preserve the revision reply: {keep_error})")
        return findings, []

    # A DROP NEEDS EVIDENCE BEHIND IT, the same rule the approval already has.
    # Measured on caeli-marketing#212 at the labelled commit, two runs: the
    # revision dropped 5 of 5 and then 10 of 10 findings — every one — with
    # ZERO tool calls, on reasons like "deliberate per the ticket comment" and
    # "correct as written". It judged from memory and abandoned the single
    # finding the author later took. The old separate pass, which re-read the
    # files, scored the same set 3 wrong and 4 real. A blanket drop with nothing
    # opened is the model looking decisive, not looking.
    looked = (stats.get("tool_calls") or 0) > 0
    kept, withdrawn = [], []
    for i, f in enumerate(findings):
        r = by_i[i]
        action = str(r.get("action") or "").strip().lower()
        why = str(r.get("why") or "")[:300]
        if action == "drop" and not looked:
            print(f"  drop of [{i}] NOT honoured — the revision opened nothing "
                  f"({why[:70]})", flush=True)
            action = "keep"
        if action == "drop":
            print(f"  dropped [{i}] {str(f.get('title'))[:60]} — {why[:90]}",
                  flush=True)
            withdrawn.append((f, 0, why))
            continue
        if action == "edit":
            # Only the fields it actually sent. An omitted field keeps its
            # current value, so a terse edit cannot blank a finding's detail.
            edited = dict(f)
            for key in ("severity", "title", "detail", "fix"):
                value = r.get(key)
                if isinstance(value, str) and value.strip():
                    edited[key] = value
            if edited != f:
                print(f"  edited [{i}] {str(f.get('title'))[:50]} — {why[:70]}",
                      flush=True)
            kept.append(edited)
            continue
        kept.append(f)

    added = []
    for a in (parsed.get("additions") or []):
        # `validate_findings` is deliberately lenient — it is built for the
        # review pass, where a finding missing a field is still worth showing.
        # An ADDITION with neither a title nor a detail is not a finding at all,
        # and would render as an empty bullet under the review's own name.
        if (isinstance(a, dict)
                and (str(a.get("title") or "").strip()
                     or str(a.get("detail") or "").strip())):
            added.append(a)
    if added:
        try:
            added = validate_findings({"findings": added})
        except ReviewError as e:
            print(f"  additions unusable ({str(e)[:80]}) — ignoring them",
                  flush=True)
            added = []
        seen = {_dedupe_key(f) for f in kept}
        added = [a for a in added if _dedupe_key(a) not in seen]
    print(f"  revision: {len(kept)} kept, {len(withdrawn)} dropped, "
          f"{len(added)} added ({stats.get('tool_calls', 0)} tool call(s))",
          flush=True)
    return kept + added, withdrawn


def _withdrawn_note(withdrawn):
    """What the reflection pass dropped, and why it says it was wrong.

    Shown rather than silently discarded: a finding that disappears between the
    review and the post is invisible to everyone, and if the scoring is wrong
    this is the only place a person can catch it.
    """
    # DEFANGED, like every other model-written string that reaches a post.
    # `_defang_links` exists for exactly this surface — the text is paraphrased
    # from a diff the PR author controls, and a link the model writes goes out
    # under the bot's identity (infra#110). render() applies it to every
    # finding's title and detail; this note reached the same body without it,
    # including the fresh second-pass rationale, which is newly generated text
    # nothing else had ever screened.
    rows = "\n".join(
        f"- ~~{_defang_links(str(f.get('title')))[:120]}~~ — scored {n}: "
        f"{_defang_links(str(why))[:220]}"
        for f, n, why in withdrawn)
    return (f"\n\n<details><summary>{len(withdrawn)} finding(s) withdrawn on "
            f"review</summary>\n\nRe-read in the checkout and scored 0 — the "
            f"code does not do this, the cited symbol is not there, or it is "
            f"already handled at this head.\n\n{rows}\n\n</details>")


def _apply_withdrawals(body, event, findings, withdrawn):
    """The posted body and event once reflection has dropped things.

    Pure, so the branch can be tested by calling it. The earlier version of
    this test grepped `inspect.getsource(main)` and failed on the word
    `approval_body` appearing in a COMMENT — which is exactly why this repo's
    own rule says never to assert on source text.
    """
    if not withdrawn:
        return body, event
    if findings:
        return body + _withdrawn_note(withdrawn), event
    # EVERY FINDING WITHDRAWN, ON EVIDENCE: that is an approval, and a better
    # founded one than an empty first pass. A drop is only honoured when the
    # revision opened at least one file (`looked`, above), so reaching here
    # means the model went back to the checkout and explained each finding
    # away in writing. That bar is LOWER than `review_findings`' one for an
    # empty first pass (which also demands a `checked` list): the difference is
    # accepted, because here there is a named finding per withdrawal to argue
    # with. This used to post a COMMENT headed "NOT a clean review" asking a
    # person to re-verify the retractions; on new-employer-portal#34 that meant
    # three correct withdrawals and a PR nobody approved, because nobody
    # re-verifies a reviewer's own retractions. The withdrawals stay in the
    # body so that the second pass can still be checked — and the headline says
    # what happened, rather than the plain "no findings" a clean first pass posts.
    #
    # The event is whatever the composition decided (APPROVE, or COMMENT when
    # files were excluded): the partial-review cap applies here as everywhere.
    n = len(withdrawn)
    # "verdict", not "approval": with files excluded the event is a COMMENT
    # and the partial-review note above says so — the body must not assert an
    # approval the event does not post (the review's 🔵 on this PR).
    verdict = "approval" if event == "APPROVE" else "verdict"
    body = body.replace(
        "### AI review — no findings\n\n",
        f"### AI review — no findings stood\n\n"
        f"This review raised {n} finding(s), went back to the checkout, and "
        f"could not stand behind any of them. The {verdict} rests on "
        f"that re-read, not on the first pass; the withdrawals are listed below "
        f"so it can be checked.\n\n", 1)
    return body + _withdrawn_note(withdrawn), event


def _finalize_review(findings, withdrawn, truncated=False, skipped=0,
                     head_sha="", repo="", wire_fields=(), diff="",
                     excluded=(), saw_every_change=None):
    """The exact body and event this review will post.

    The WHOLE composition, not just the withdrawal branch, because the previous
    seam left a source-grep as the only thing guarding the wiring — and a grep
    for `_apply_withdrawals(` passes while the call is fed the wrong arguments.
    Pulling the choice of approval-vs-render in here means a test can assert
    what actually gets posted for a given (findings, withdrawn) pair, which is
    the thing that matters.
    """
    unseen = False
    body = (approval_body(head_sha, repo=repo, wire_fields=wire_fields, diff=diff,
                          excluded=excluded)
            if not findings
            else render(findings, truncated, skipped, head_sha=head_sha,
                        repo=repo, wire_fields=wire_fields, diff=diff,
                        excluded=excluded))
    event = review_event(findings)
    # A PARTIAL REVIEW NEVER APPROVES. An approval is the one outcome that
    # carries authority, and one that covered 10 of 25 files reads exactly like
    # one that covered all of them. The files are named in the body either way;
    # the verdict has to stop short too, or the name is a footnote on a green
    # tick.
    # `saw_every_change` is the honest question: an agent that OPENED an
    # excluded file read it at head and still never saw the diff. Defaults to
    # "whatever `excluded` says" so every existing caller keeps its behaviour.
    if event == "APPROVE" and not (
            saw_every_change if saw_every_change is not None else not excluded):
        event = "COMMENT"
        unseen = True
    body, event = _apply_withdrawals(body, event=event,
                                     findings=findings, withdrawn=withdrawn)
    # LAST, so it rewrites whatever prose actually survived. With every
    # excluded file opened there is no unreviewed-files note to carry the news,
    # so the approval wording would otherwise stand above a COMMENT — the same
    # false-clean verdict as a refused approval, reached by a different route.
    # ONLY WHEN THERE IS NO OTHER NOTE. If files remain unopened,
    # `_unreviewed_files_note` is already at the top saying this is not an
    # approval, and two notes making overlapping claims read as a bug.
    if unseen and not excluded:
        body = _changes_unseen_note() + body.replace(
            "### AI review — no findings\n",
            "### AI review — no findings, and not an approval\n").replace(
            "**What this approval is.**", "**What this would have been.**")
    return body, event


def _one_pass(prompt, work, what):
    """A parsed, shape-checked set of findings from one agent pass."""
    return validate_findings(_reply(prompt, work, what))


#: claudelint's default (https://claudelint.com/rules/claude-md/claude-md-size).
#: The number matters because of what happens past it: Claude Code warns about
#: degraded performance and MAY TRUNCATE the file. An oversize CLAUDE.md is not
#: untidy, it is instructions that silently stop being read — which is the one
#: failure a repo's own rules file cannot report about itself.
def _diff_paths(diff):
    """Every path a unified diff touches."""
    return {m.group(1) for m in re.finditer(r"^\+\+\+ b/(.+)$", diff or "",
                                           re.M)} - {"dev/null"}


def review_findings(prompt, work, repo=""):
    """The findings for a change, with an EMPTY result confirmed by a second run.

    A DEGRADED REPLY MUST NOT LOOK LIKE A CLEAN ONE. `validate_findings` already
    refuses the malformed shapes — no `findings` key, null, wrong type, non-object
    elements. What it cannot refuse is a syntactically perfect empty answer, and
    that is the one that posts a formal APPROVE.

    The model produces exactly that on its own. Measured on this PR's own review
    prompt, n=3 per setting, 131k budget:

        none     5 findings | 0 | 1        <- best, and still 1-in-3 vacuous
        low      0 | 0 | 2
        high     unparseable | 0 | 0       (one run: 458s, 44k tokens, 0 findings)
        omitted  0 | unparseable | 0

    Seven of those twelve replies were the literal 15-character `{"findings":[]}`,
    several after only single-digit output tokens. That is a shrug, not a review,
    and nothing downstream can tell it from a genuinely clean one.

    So an empty result is asked again — the same discipline `e2e_canary` already
    applies, where a single failure is re-run before it is believed. It costs
    almost nothing: the empty case is the FAST case (3-13s), so the second pass is
    only paid when the first found nothing.

    THE SECOND PASS ASKS A DIFFERENT QUESTION, deliberately. Re-running the same
    prompt is a second draw from the same distribution, not a confirmation — and
    the draws are CORRELATED, because the shrug is a state the model gets into
    rather than independent noise. Two empties would then be a lowered-probability
    approval dressed up as a verified one.

    So the confirmation pass asks it to SHOW ITS WORK instead: name the files it
    opened and what it verified in each. A model that has given up cannot produce
    that list, while a model that genuinely read the change can. The `checked`
    evidence is the artifact a shrug cannot fake, and an empty one is refused.

    Anything unreadable RAISES rather than approving. One empty result plus one
    unreadable one is not evidence of clean code, and this repo's rule is that a
    scanner which cannot confirm coverage must be loud.
    """
    findings = _one_pass(prompt, work, "review")
    if findings:
        return findings

    print("  no findings — asking it to show its work before approving", flush=True)
    try:
        parsed = _reply(CONFIRM_PROMPT.format(original=prompt), work, "confirmation",
                        need_evidence=True)
        second = validate_findings(parsed)
    except AgentFailed:
        # hermes did not answer. Re-labelling that "could not be read" is the
        # content-vs-infra mix-up AgentFailed exists to stop, and the first-pass
        # path already gets it right — this one did not.
        raise
    except ReviewError as e:
        raise ReviewError(
            f"first pass found nothing and the second could not be read ({e}) — "
            "refusing to approve on an unconfirmed empty review") from e
    if second:
        print(f"  second pass found {len(second)} the first missed", flush=True)
        return second

    # AN APPROVAL HAS TO SHOW ITS WORK. This is the whole point of the second
    # pass being a different question: `findings: []` is producible by a model
    # that read nothing, `checked: [...]` is not.
    checked = parsed.get("checked")
    if not isinstance(checked, list) or not checked:
        raise ReviewError(
            "the confirmation pass approved without saying what it checked "
            f"(checked={checked!r}) — refusing to approve on an unevidenced "
            "empty review")

    # …AND `checked` IS A CLAIM, NOT EVIDENCE. It is worth exactly as much as
    # the tool calls behind it, and the model is not the authority on whether
    # those happened. Measured on slack-app#375: the confirmation pass made ZERO
    # tool calls, answered in 5.8s, and reported nine files examined. That
    # cleared the gate above word-for-word while examining nothing — a
    # fabricated evidence list is worse than no review, because it satisfies the
    # guard built to catch precisely this.
    #
    # The loop counts calls; this compares the claim against the count. Not
    # `>= len(checked)`, deliberately: one `grep` can legitimately answer for
    # several files, and a reviewer that had to open each one separately would
    # be penalised for being efficient. Zero is the honest bar — it separates
    # "looked and found nothing" from "did not look".
    calls = (_CURRENT.get("stats") or {}).get("tool_calls", 0)
    if not calls:
        raise ReviewError(
            f"the confirmation pass claims to have checked {len(checked)} "
            f"file(s) but made NO tool calls — it read nothing, so there is no "
            f"evidence behind the approval")
    print(f"  confirmed clean after examining {len(checked)} file(s) "
          f"({calls} tool call(s))", flush=True)
    return second


def severity_breakdown(findings):
    """"2 high, 1 unknown, 3 low" — worst first, for the job log.

    The line the job log was missing: `COMMENT: 3 finding(s)` says a review
    happened but not whether it found anything that matters, and those are very
    different runs to scroll past.
    """
    counts = {}
    for f in findings:
        sev = normalize_severity(f.get("severity"))
        counts[sev] = counts.get(sev, 0) + 1
    return ", ".join(f"{n} {sev}" for sev, n in
                     sorted(counts.items(), key=lambda kv: RANK[kv[0]])) or "none"


def review_event(findings):
    """The GitHub review event for a set of findings. Worst severity wins.

    AN UNRECOGNISED SEVERITY NEVER APPROVES. If the model answers "critical" or
    "blocker" — neither of which is in the vocabulary it was given — mapping the
    unknown to low would approve the most serious thing it ever found. Approval
    is the only outcome here that carries authority, so it is granted only for
    severities we actually understand, and anything else falls to COMMENT:
    withholds the approval, without blocking on a word we cannot interpret.
    """
    if not findings:
        return "APPROVE"
    events = {EVENT_BY_SEVERITY[normalize_severity(f.get("severity"))] for f in findings}
    for event in ("REQUEST_CHANGES", "COMMENT", "APPROVE"):
        if event in events:
            return event
    return "COMMENT"  # unreachable; a set built from the map is never empty


#: What GitHub says when it refuses a verdict on your own PR — "Can not approve
#: your own pull request" / "Can not request changes on your own pull request".
#: The refusals GitHub answers a verdict with, and which are ABOUT THE POSTER
#: rather than about the review. Both end the same way — the findings are worth
#: posting, the verdict is not available — so both fall back to a comment.
#:
#:   · your own pull request — a human token reviewing its author's PR;
#:   · a GitHub App's token cannot APPROVE at all, which is what the workflow's
#:     own GITHUB_TOKEN is. Measured on this repository's first hosted
#:     self-review: a clean diff, 0 findings, and the whole review lost to a
#:     422 because approving was the one thing it could not do.
SELF_REVIEW_REFUSAL = "your own pull request"
#: Each names the POSTER as the reason. "Review cannot be submitted" was here
#: too and came out: it is GitHub's generic wrapper, and matching it would send
#: any 422 carrying that phrase — a stale head, a body over the limit — down
#: the same path, which is precisely the silent downgrade this guard exists to
#: prevent. The App refusal is already named by its nested reason.
VERDICT_REFUSALS = (
    SELF_REVIEW_REFUSAL,
    "not permitted to approve",
    "cannot approve",
    "can not approve",
)


#: What the dismissal says. It is a claim about OUR earlier objection, not about
#: the PR — the medium findings in the review being posted alongside it still
#: stand, and a human's blocking review is never touched.
#: Deliberately narrower than the old wording, which claimed "the blocking
#: finding is no longer present at this head" — a fact nothing verified. This
#: says only what was actually established: the code moved, the whole diff was
#: read, and nothing blocking was found in it.
UNAPPROVE_MESSAGE = (
    "Withdrawn: this approval was given at {old}, and a later review of {new} "
    "found something. An approval is a claim about the code that was read, and "
    "GitHub does not retract one when a COMMENTED review follows it — so it "
    "would otherwise sit here, green, next to findings on newer code.\n\n"
    "Nothing is being asserted about severity — and this message CANNOT assert "
    "it. The withdrawal fires on a COMMENT verdict, which `review_event` also "
    "returns for a finding whose severity it did not recognise, precisely "
    "because an unknown word must never approve. Calling those findings "
    "non-blocking would claim the one thing that run declined to decide.\n\n"
    "They carry their own severity in the review that raised them. The point "
    "here is only that the approval no longer describes the head. Re-request a "
    "review once they are addressed and it will approve again if it approves."
)

DISMISS_MESSAGE = (
    "Superseded: this block was raised against {old}, the PR has since moved to "
    "{new}, and a full re-review of that head found nothing blocking.\n\n"
    "That is not a claim the original finding was fixed — it is a claim that it "
    "does not reproduce on the current code. If it was real and is still there, "
    "say so and it will be raised again.\n\n"
    "Any findings posted alongside this dismissal still stand; they are not "
    "severe enough to block."
)


@lru_cache(maxsize=1)
def _me():
    """The login this token posts as.

    `REVIEW_BOT_LOGIN` wins when set: the workflow's own GITHUB_TOKEN cannot
    call `/user` at all, and a self-hosted bot's login is known to the caller
    anyway — asking GitHub is the fallback, not the source of truth.
    """
    known = os.environ.get("REVIEW_BOT_LOGIN", "").strip()
    if known:
        return known
    try:
        return json.loads(gh("/user")).get("login") or ""
    except Exception as e:  # noqa: BLE001 — identity is best-effort
        print(f"[pr-review] could not resolve own login: {type(e).__name__}")
        return ""


def _withdraw_stale_approval(repo, pr, head_sha):
    """Retract OUR OWN earlier APPROVE once a later review has found something.

    GitHub changes a reviewer's state only on APPROVE or REQUEST_CHANGES — a
    COMMENTED review leaves an existing approval standing. So the reviewer can
    approve at one commit, find a real problem at the next, and leave a green
    "approved these changes" sitting beside its own findings. Seen on
    slack-app#363:

        20:25:32  APPROVED   @6b702ab
        20:31:03  COMMENTED  @df3365d   <- found something, newer code

    THE GUARDS ARE DELIBERATELY LOOSER THAN `_dismiss_stale_block`'s, because
    the risk points the other way. Clearing a block wrongly UNBLOCKS a merge, so
    that path demands the head moved and the diff was read whole. Withdrawing an
    approval wrongly only costs a re-request. An approval standing next to a
    finding is incoherent however it arose, so neither a truncated read nor an
    unmoved head is a reason to leave it there.

    NEVER TOUCHES A REVIEW IT DID NOT WRITE — the same and only real safety
    property. A human's approval is theirs to withdraw.

    Best-effort: the review is already posted by the time this runs, so a
    failure leaves the status quo rather than losing anything.
    """
    me = _me()
    if not me:
        return []
    try:
        reviews = json.loads(gh(f"/repos/{ORG}/{repo}/pulls/{pr}/reviews"))
    except Exception as e:  # noqa: BLE001
        print(f"[pr-review] could not read reviews to unapprove: {type(e).__name__}")
        return []

    withdrawn = []
    for r in reviews:
        if (r.get("user") or {}).get("login") != me:
            continue                      # someone else's approval, never ours
        if r.get("state") != "APPROVED":
            continue
        approved_at = r.get("commit_id") or ""
        try:
            gh(f"/repos/{ORG}/{repo}/pulls/{pr}/reviews/{r['id']}/dismissals",
               method="PUT",
               body={"message": UNAPPROVE_MESSAGE.format(
                         old=approved_at[:7] or "an earlier commit",
                         new=head_sha[:7] or "the current head"),
                     "event": "DISMISS"})
            withdrawn.append(r["id"])
        except Exception as e:  # noqa: BLE001
            print(f"[pr-review] could not withdraw approval {r.get('id')}: "
                  f"{type(e).__name__}")
    return withdrawn


def _dismiss_stale_block(repo, pr, event, head_sha, truncated):
    """Clear OUR OWN stale CHANGES_REQUESTED once the blocking finding is gone.

    A COMMENTED review does not clear a CHANGES_REQUESTED on GitHub — only an
    APPROVE or an explicit dismissal does. So once this reviewer blocks, every
    later review carrying even one medium finding leaves the PR blocked by an
    objection that no longer exists. Seen on slack-app#344: a high on
    `find_by_domain_match` was fixed, two later reviews found unrelated mediums,
    and the PR sat at CHANGES_REQUESTED with nothing left to change.

    Only fires on COMMENT. A REQUEST_CHANGES is still blocking and must stay;
    an APPROVE already supersedes on GitHub's side and needs no help.

    NEVER TOUCHES A REVIEW IT DID NOT WRITE. Dismissing a human's blocking review
    is not a judgement this tool gets to make — it would silently delete a
    colleague's objection because a model could not see the problem. The login
    check is the whole safety property here.

    Best-effort: a failure to dismiss is logged and swallowed. The review itself
    is already posted by then, so the worst case is the status quo — a stale
    block — rather than a lost review.
    """
    if event != "COMMENT":
        return []
    dismissed = []
    if truncated:
        # A truncated review never saw the whole diff, so "no high found" may
        # only mean "did not look there". THE reported failure mode: run 1 finds
        # a high in the tail, the author pushes something unrelated, run 2 again
        # stops short of that region and reports mediums — and the old code
        # cleared a live block on the strength of not having read it.
        print("[pr-review] not dismissing: this review was truncated, so a "
              "clean result is not evidence the earlier finding is gone")
        return []
    me = _me()
    if not me:
        return []
    try:
        reviews = json.loads(gh(f"/repos/{ORG}/{repo}/pulls/{pr}/reviews"))
    except Exception as e:  # noqa: BLE001
        print(f"[pr-review] could not read reviews to dismiss: {type(e).__name__}")
        return []

    for r in reviews:
        if (r.get("user") or {}).get("login") != me:
            continue                              # someone else's — never ours to clear
        if r.get("state") != "CHANGES_REQUESTED":
            continue                              # already dismissed, or not a block
        blocked_at = r.get("commit_id")
        if not blocked_at or blocked_at == head_sha:
            # The head has not moved since the block. Re-reading the same code
            # and reaching a different verdict is model nondeterminism, not a
            # fix — dismissing on that would let a re-request clear any block by
            # rolling the dice until it came up quiet.
            print(f"[pr-review] not dismissing {r.get('id')}: head unchanged "
                  f"since it was raised")
            continue
        try:
            gh(f"/repos/{ORG}/{repo}/pulls/{pr}/reviews/{r['id']}/dismissals",
               method="PUT",
               body={"message": DISMISS_MESSAGE.format(old=blocked_at[:7],
                                                        new=head_sha[:7]),
                     "event": "DISMISS"})
            dismissed.append(r["id"])
        except Exception as e:  # noqa: BLE001
            print(f"[pr-review] could not dismiss review {r.get('id')}: "
                  f"{type(e).__name__}")
    return dismissed


def _verdict_withheld(body, event):
    """The same findings, with the verdict's own prose taken back out.

    A refused APPROVE fell back to a comment carrying the APPROVAL body —
    which opens "**What this approval is.**" — so the posted comment claimed
    an approval GitHub had just declined to record. That is the false-clean
    verdict this module keeps being bitten by, arriving through the one door
    nobody had checked.
    """
    verdict = event.lower().replace("_", " ")
    note = (f"> **⚠️ GitHub would not record a `{verdict}` from this account.** "
            f"The findings below stand; the verdict does not. Nothing here has "
            f"been recorded as an approval.\n\n")
    cleaned = body.replace("### AI review — no findings\n",
                           "### AI review — no findings, verdict not recorded\n")
    cleaned = cleaned.replace(
        "**What this approval is.**",
        "**What this would have been.**")
    return note + cleaned


def _write_step_summary(repo, pr, event, body):
    """The run page's copy of what was POSTED. Never raises."""
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    try:
        with open(summary, "a") as fh:
            fh.write(f"## {event} — {repo}#{pr}\n\n{body}\n")
    except OSError as e:
        print(f"  (could not write step summary: {e})", flush=True)


def post_review(repo, pr, event, body, head_sha="", truncated=False):
    """POST the review; return the event actually posted.

    THE FALLBACK IS KEYED ON THE REASON, NOT ON THE STATUS. 422 is GitHub's
    catch-all for "unprocessable": the head moved between checkout and POST, the
    body is too long, the PR closed underneath us. Treating every 422 as
    self-review silently downgraded a blocking verdict to a comment AND printed
    a fabricated explanation for why — the failure disappears behind a sentence
    that is simply untrue.

    So only the refusal we can actually name is swallowed. Anything else raises,
    reaching `guard_main`, which alerts. A review that could not be posted is a
    review that did not happen, and this module's whole discipline is that those
    must be loud.
    """
    def post(ev, text=None):
        gh(f"/repos/{ORG}/{repo}/pulls/{pr}/reviews", method="POST",
           body={"event": ev, "body": text if text is not None else body})

    try:
        post(event)
        # AFTER the post, never before: dismissing first would leave a window
        # where the PR is unblocked with nothing said in its place.
        for rid in _dismiss_stale_block(repo, pr, event, head_sha, truncated):
            print(f"  dismissed our own stale CHANGES_REQUESTED ({rid})")
        # ONLY on COMMENT. A REQUEST_CHANGES already moves our state off
        # approved on GitHub's side, so the old approval is not misleading
        # there; a COMMENTED review is the one that leaves it standing.
        if event == "COMMENT":
            for rid in _withdraw_stale_approval(repo, pr, head_sha):
                print(f"  withdrew our own now-stale APPROVE ({rid})")
        return event
    except urllib.error.HTTPError as e:
        if e.code != 422 or event == "COMMENT":
            raise
        try:
            detail = e.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 — the body is best-effort context
            detail = ""
        low = detail.lower()
        if not any(r in low for r in VERDICT_REFUSALS):
            raise
        # The verdict is refused, the findings are not. Post them as a comment
        # and SAY which verdict was withheld, so a clean review is not read as
        # an approval that never happened.
        post("COMMENT", _verdict_withheld(body, event))
        # THE SAME RECONCILIATION A REAL COMMENT GETS. The fallback returned
        # early and skipped it, so a clean review on a newer head left our own
        # older CHANGES_REQUESTED standing — an approval would have superseded
        # it, and this comment is what that approval turned into.
        for rid in _withdraw_stale_approval(repo, pr, head_sha):
            print(f"  withdrew our own now-stale APPROVE ({rid})")
        why = ("own PR" if SELF_REVIEW_REFUSAL in low
               else "this token may not set a verdict")
        return f"COMMENT ({event.lower().replace('_', ' ')} refused — {why})"


#: A markdown link or image, and a bare autolink. `detail` and `title` are
#: model-written and land in the review body as RAW MARKDOWN, so these are the
#: constructs that turn model text into something a reader can click.
_MD_LINK = re.compile(r"(!?)\[([^\]]*)\]\(\s*<?([^)\s]*)[^)]*\)")
_MD_AUTOLINK = re.compile(r"<((?:https?|ftp)://[^>\s]+)>")


#: A bare URL, with its immediate neighbours captured so an already-backticked
#: one is left alone. Trailing punctuation is excluded from the match so a URL
#: ending a sentence does not swallow the full stop.
_BARE_URL = re.compile(r"(`?)((?:https?|ftp)://[^\s`<>\]]*[^\s`<>\].,;:!?)])(`?)")


def _bare(url):
    """Backticks inside a URL would close the span we are about to open."""
    return url.replace("`", "")


def _defang_links(text):
    """Neutralise clickable markdown in model-written prose.

    `render` emits `detail` and `title` as raw markdown, so a link the model
    writes becomes a live link posted under this bot's identity. That is a
    phishing surface: the text is paraphrased from a PR's diff, which its author
    controls, and readers trust the reviewer. Demonstrated for real — the review
    on infra#110 rendered `[click](https://attacker.invalid)` as a live anchor
    while describing a NARROWER version of the same bug in `fix`.

    DELIBERATELY NOT CODE-SPAN AWARE, after trying that first and measuring it.
    The obvious design preserves backticked text and defangs only the prose
    between spans. It does not hold: reproducing CommonMark's span pairing means
    matching the renderer exactly, and where the two disagree the escaper has a
    bypass. Verified against GitHub's own /markdown endpoint — a span-aware
    version left the infra#110 payload rendering as a live anchor, because
    earlier backtick runs in the same paragraph shift how the runs pair.

    A blanket rewrite costs almost nothing, because the link pattern needs the
    literal `](` sequence: `dict["key"]` and `items[0]` do not match and survive
    untouched inside or outside a span. What it does change is a link written
    INSIDE a code span, which renders as literal text either way — cosmetic, and
    the safe direction to be wrong in.

    The URL is kept as plain text in parentheses. Citing a doc URL is legitimate;
    the reader can still read it, they just cannot click it from a message they
    did not author.
    """
    text = _MD_LINK.sub(
        lambda m: f"{m.group(2)} (`{_bare(m.group(3))}`)" if m.group(3) else m.group(2),
        text)
    text = _MD_AUTOLINK.sub(lambda m: f"`{_bare(m.group(1))}`", text)
    # A URL left as PLAIN text is not defanged — GFM autolinks bare URLs, and
    # greedily: the first attempt here emitted `click (https://…)` and GitHub
    # turned the whole tail into one anchor, `https://attacker.invalid)\n```\nx=2`.
    # Backticks are what actually stop it, because a code span never autolinks.
    # Only wrap a URL that is not already adjacent to one, which is a cheap local
    # check rather than another attempt to model span pairing.
    return _BARE_URL.sub(
        lambda m: m.group(0) if m.group(1) or m.group(3) else f"`{_bare(m.group(2))}`",
        text)


#: A repo path, restricted hard enough that the result cannot break out of the
#: markdown link it goes into. `file` is MODEL-WRITTEN, so a `](` in it would
#: end the link early and let the rest render as live markdown under this bot's
#: name — the same class of hole `_defang_links` exists for. Every character
#: that survives here is also URL-safe, so the href needs no escaping.
_SAFE_PATH = re.compile(r"[A-Za-z0-9._/-]+")


def _code_span(text):
    """Wrap model-written text in a code span it cannot close early.

    A backtick in the text ends the span and everything after it renders as live
    markdown. MEASURED through GitHub's `/markdown`, with a path the model
    controls:

        `a.py](https://attacker.invalid) [x:3`   -> <code>…</code>, inert
        `a.py`](https://attacker.invalid)`:3`    -> a real <a href> to the host

    So the span is a real defence, but only while it stays intact. CommonMark
    closes it on the first backtick run of EQUAL length, so use one longer; the
    spaces are stripped by the renderer and stop a leading or trailing backtick
    in the text from fusing with the fence.
    """
    text = str(text)
    run = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence, pad = "`" * (run + 1), (" " if run else "")
    return f"{fence}{pad}{text}{pad}{fence}"


def _where_link(repo, path, line, head_sha):
    """`path:line` as a link to the exact blob the reviewer read.

    A review names files it never links to, so following one up means copying a
    path and hunting for it. Pinned to `head_sha` rather than a branch, for the
    same reason the footer names the commit: a link to `main` drifts, and would
    point at code that does not match the finding as soon as anything merges.

    FALLS BACK TO A PLAIN CODE SPAN whenever the link cannot be built honestly —
    no repo, no sha, no path, or a path this cannot vouch for. A wrong link is
    worse than none: it sends the reader somewhere confident and incorrect.
    """
    # A DETERMINISTIC, PR-LEVEL finding (the title names no ticket; an
    # agent-written commit links no session) has no file and no line, and there
    # is nothing dishonest about saying so. Rendering it as `?` implied the
    # reviewer had lost track of where it was.
    if not str(path or "").strip():
        return "_the pull request_"
    where = str(path or "?").strip()
    line = str(line or "").strip()
    # ASCII ONLY, not `.isdigit()` — and not `.isdecimal()` either. Both return
    # True for non-ASCII digits ('٣' passes both, '²' passes isdigit), which
    # would put raw non-ASCII in the href fragment and produce `#L٣`: a link
    # that goes nowhere, which is the one thing this function promises not to
    # emit. The path is allowlisted to ASCII; the line was the only value that
    # was not. The AI review's 🔵 on the PR that added this.
    if re.fullmatch(r"[0-9]+", line):
        where += f":{line}"
    else:
        line = ""
    clean = str(path or "").strip().lstrip("/")
    if not (repo and head_sha and clean
            and _SAFE_PATH.fullmatch(clean)
            and ".." not in clean.split("/")):
        return _code_span(where)
    anchor = f"#L{line}" if line else ""
    url = f"https://github.com/{ORG}/{repo}/blob/{head_sha}/{clean}{anchor}"
    # `where` passed _SAFE_PATH, so it carries no backtick and the plain span
    # inside the link text cannot break out.
    return f"[`{where}`]({url})"


def _unescape_backticks(text):
    """The model escapes backticks as if writing a shell or JS string, and GitHub
    renders `\\`` literally — so a finding that quotes code arrives with visible
    backslashes through the one thing it most needs to show. Seen on infra#106.

    One copy, used by both `detail` and `fix`. It was briefly two, which is how
    the pair drifts: the next escape the model invents gets handled in whichever
    site the fixer happened to be reading.
    """
    return str(text or "").replace("\\`", "`")


def _fence_for(body):
    """A fence long enough that `body` cannot close it.

    CommonMark closes a fenced block on the first run of backticks at least as
    long as the opening fence, so three backticks inside `fix` end the block
    early and everything after renders as LIVE MARKDOWN under this bot's name —
    links and images included. `fix` is derived from a PR's diff, which its
    author controls, so that is reachable rather than theoretical.

    The language token is allowlisted for the same reason; this is the content
    side of the same boundary, which the allowlist alone did not cover.
    """
    longest = max((len(m) for m in re.findall(r"`+", body)), default=0)
    return "`" * max(3, longest + 1)


#: Languages we will echo into a fenced block. An unknown value becomes a bare
#: fence rather than being interpolated — the model picks this string, and it
#: lands inside markdown the bot posts, so it is not allowed to invent syntax.
#: How many lines of `fix` survive rendering. The prompt asks for two.
FIX_SKETCH_LINES = 4

FIX_LANGUAGES = {
    "python", "py", "javascript", "js", "typescript", "ts", "tsx", "jsx",
    "json", "yaml", "yml", "bash", "sh", "swift", "go", "sql", "diff",
    "html", "css", "toml", "ini", "mjs", "cjs",
}


def _fix_block(finding):
    """The proposed change, fenced, with whether the model actually checked it.

    Two things are load-bearing here.

    The fence LANGUAGE comes from an allowlist. `fix_language` is model-written
    and lands in markdown this bot posts under its own name; an arbitrary string
    after the backticks is the model authoring markup rather than content. An
    unknown value degrades to a bare fence, which renders fine.

    The VERIFIED flag is rendered even when true, because the useful signal is
    the contrast. This reviewer proposed `generation_config` for a key that the
    SDK calls `config` (slack-app#341) and argued it confidently; the fix read
    exactly like a correct one. An author who cannot tell "I opened the file"
    from "I recall how this library works" has to re-derive every fix, which is
    the cost this block exists to remove.

    Absent or empty `fix` renders nothing at all. A finding without a remedy is
    still a finding — that is the standing rule and this does not overrule it.
    """
    fix = _unescape_backticks(finding.get("fix")).strip()
    if not fix:
        return []
    # A SKETCH, CAPPED. The prompt asks for two lines; a model that ignores that
    # and pastes a function body re-creates the truncation this shape exists to
    # avoid, and buries the direction inside code the author has to read anyway.
    lines = fix.splitlines()
    if len(lines) > FIX_SKETCH_LINES:
        fix = "\n".join(lines[:FIX_SKETCH_LINES]) + "\n…"
    lang = str(finding.get("fix_language") or "").strip().lower()
    if lang not in FIX_LANGUAGES:
        lang = ""
    verified = finding.get("fix_verified") is True
    label = ("**Suggested direction** — the named symbols were checked at this head"
             if verified else
             "**Suggested direction** — NOT checked against the code; confirm before applying")
    fence = _fence_for(fix)
    return [label, f"{fence}{lang}", fix, fence, ""]


def _unchecked_consumers_note(repo, wire_fields):
    """Say that the review could not speak for other repositories.

    THE REVIEWER HAS ONE REPO. It is cloned into a checkout of the PR's repo and
    nothing else, so a field that other services or clients branch on is
    reviewed from one side only. That is invisible in the output: a clean review
    of a wire change reads exactly like a clean review of an internal one.

    Real case, slack-app#363. The PR changed how "has this program been decided"
    is represented, around `unspecified_keys`. `browser-extension` branches on
    that field in its checkout CTA — 34 references, with tests pinning both the
    empty and non-empty cases — and the reviewer could not see any of it. It
    posted findings on the slack-app side and said nothing about the consumer,
    because it had no way to know one existed.

    This does not fix that; nothing here can read another repo. It makes the gap
    VISIBLE, which is the same move as `fix_verified`, the truncation note, and
    naming the commit that was read: the review states what it did not check
    instead of leaving silence to be read as coverage.
    """
    if not wire_fields:
        return ""
    names = ", ".join(_code_span(f) for f in wire_fields)
    return (f" It read only `{repo or 'this repo'}`, and {names} "
            f"{'cross' if len(wire_fields) > 1 else 'crosses'} a wire boundary — "
            "consumers in other repositories were NOT checked.")


def _changes_unseen_note():
    """Why a clean review is not an approval when the diff did not fit.

    The files were read — that is why there is no unreviewed-files note — but
    `read_file` returns the HEAD version, and their patches were never shown.
    Reading what the code says now is not seeing what the change did to it.
    """
    return ("> **⚠️ Not an approval.** Some changed files did not fit the diff "
            "budget. The agent opened them in the checkout, so nothing went "
            "unlooked-at — but it saw them AS THEY NOW STAND, not the changes "
            "made to them. A clean result on that basis is worth reading and "
            "is not an approval.\n\n")


def _unreviewed_files_note(excluded):
    """Name the files this review did not read. Loud, and first.

    Not a footer: a person deciding whether to merge needs to see it before the
    findings, because "no findings" means something different when fifteen
    files were never opened. The list is what makes a follow-up review possible
    — the reader can launch one naming exactly these.
    """
    if not excluded:
        return ""
    shown = [_code_span(p) for p in excluded[:30]]
    more = f", and {len(excluded) - 30} more" if len(excluded) > 30 else ""
    return (f"> **⚠️ Partial review — {len(excluded)} changed file(s) were NOT "
            f"opened** (they did not fit the diff budget and the agent did not "
            f"read them): {', '.join(shown)}{more}.\n"
            f"> This is not an approval of those files. Split the PR, or "
            f"request a follow-up review naming them.\n\n")


def render(findings, truncated, skipped, head_sha="", repo="", diff="",
           wire_fields=(), excluded=()):
    findings.sort(key=lambda f: RANK[normalize_severity(f.get("severity"))])
    lines = ["### AI review", "", _unreviewed_files_note(excluded)]
    for f in findings[:MAX_FINDINGS]:
        sev = normalize_severity(f.get("severity"))
        where = _where_link(repo, f.get("file"), f.get("line"), head_sha)
        # The model escapes backticks as if writing a shell or JS string, and
        # GitHub renders `\\`` literally — so a finding that quotes code arrives
        # with visible backslashes through the one thing it most needs to show.
        # Seen on infra#106, where every code span in the review was `\\`${X}\\``.
        detail = _defang_links(_unescape_backticks(f.get("detail")).strip())
        title = _defang_links(str(f.get("title", "")).strip())
        if not title:
            # The model dropped `title` on a real run and rendered "(untitled)"
            # four times; the heading is what makes a list scannable.
            title = re.split(r"(?<=[.!?])\s", detail)[0][:110] if detail else "(no detail)"
        lines += [f"{ICON[sev]} **{title}** — {where}", detail, ""]
        lines += _fix_block(f)
    if len(findings) > MAX_FINDINGS:
        lines.append(f"_…and {len(findings) - MAX_FINDINGS} more, not shown._")
    notes = []
    if truncated:
        notes.append("some changed files were over the diff budget and are "
                     "named above")
    if skipped:
        notes.append(f"{skipped} generated/binary files skipped")
    lines.append("")
    # NAME THE COMMIT THAT WAS ACTUALLY READ.
    #
    # GitHub stamps a review's `commit_id` with the head at POST time, not at
    # read time, so a push that lands mid-run silently re-attributes the review
    # to code it never saw. Measured on slack-app#348: the agent checked out
    # a0d780b at 22:24:38, a push landed, and the review posted at 22:26:53 was
    # recorded against fa9b51f. Every finding was correct for a0d780b and read
    # as flatly wrong against fa9b51f — including one asserting a fix was
    # missing that was, by then, present.
    #
    # There is no way to stop the race; a reviewer necessarily reads a snapshot.
    # What is fixable is the silence about which snapshot. Saying it here makes
    # a stale review self-evident instead of looking like a bad finding.
    read_at = f" It read `{head_sha[:7]}`." if head_sha else ""
    lines.append(f"_Automated review — agentic-review ({llm.model_label()}) with "
                 "read access to the repository at this PR's head." + read_at + " It did not run the tests"
                 + ("; " + ", ".join(notes) if notes else "") + "."
                 + _unchecked_consumers_note(repo, wire_fields) + "_")
    if diff:
        lines.append("")
        lines.append(_DIFF_MARK.format(fp=_diff_fp(diff)))
    return "\n".join(lines)


#: `{head}` is filled by `approval_body`. It matters MORE here than on the
#: findings path: an approval unblocks a merge, so a review silently
#: re-attributed to code it never read does its worst on the one verdict that
#: carries authority. GitHub's own staleness UI does not help — it flags a
#: review when the head moves AFTER posting, and the mid-run case is stamped
#: against the NEW head, so it renders as a current, genuine approval.
APPROVAL = (
    "### AI review — no findings\n\n"
    "Reviewed for correctness, security, data integrity, duplication, consistency "
    "with CLAUDE.md, and test coverage — with read access to the repository, not "
    "just the diff.\n\n"
    "**What this approval is.** An agent read the change at `{head}`, explored the "
    "code around it, and found no defect it could point at. It did NOT run the "
    "tests or open the app, and it stopped looking when it was satisfied — so this "
    "is a competent second pair of eyes, not proof the change is safe to ship.")


def approval_body(head_sha="", repo="", wire_fields=(), diff="", excluded=()):
    """APPROVAL with the commit it read, or without the claim if we cannot say.

    Same rule as the findings footer: absent beats wrong. A `{head}` left
    unfilled would render a literal brace to the author, and a blank one would
    assert a commit of "".
    """
    body = (APPROVAL.replace(" at `{head}`", "") if not head_sha
            else APPROVAL.format(head=head_sha[:7]))
    if excluded:
        # The heading says "no findings"; the note says in what. Both true.
        body = _unreviewed_files_note(excluded) + body
    # THE APPROVAL IS WHERE THIS MATTERS MOST. It is the verdict that unblocks a
    # merge, so a clean result on a change other repos consume is exactly the
    # one that should not read as full coverage.
    return (body + _unchecked_consumers_note(repo, wire_fields)
            + ("\n\n" + _DIFF_MARK.format(fp=_diff_fp(diff)) if diff else ""))



#: Fingerprint of the diff a review actually read, carried in its own body.
#:
#: An update-branch moves the head SHA without changing the PR's own diff, so a
#: SHA comparison alone cannot tell "new work" from "the base moved underneath
#: me". This is invisible in rendered markdown and absent from older reviews,
#: which simply means those fall through and get reviewed — the safe direction.
_DIFF_MARK = "<!-- caeli-review diff:{fp} -->"


def _diff_fp(diff):
    """A fingerprint of WHAT changed, independent of the order files arrive in.

    `pr_diff` packs source before tests, and packing order is a property of the
    reviewer, not of the pull request — a reviewer that reorders its input must
    not mistake that for the author having pushed. Hashing per-file blobs in
    sorted order means the same set of changes always fingerprints the same,
    whatever order they were shown in.
    """
    # Split at LINE STARTS so every blob keeps its `diff --git` header — a
    # `split("\ndiff --git ")` strips the prefix from all but the first, and
    # the same file then canonicalises differently depending on its position,
    # which is the exact order-dependence this exists to remove.
    blobs = [b for b in re.split(r"(?m)^(?=diff --git )", diff) if b.strip()]
    canonical = "\n".join(sorted(b.strip() for b in blobs))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _someone_replied_since(repo, pr, when):
    """Has anyone said anything since our last review at `when`?

    THE DISAGREEMENT PATH. `_already_reviewed` exists so a re-request on an
    unchanged commit does not re-run minutes of work for the same answer. But
    that also removed the only way to argue with a finding: an author who
    replies "this is deliberate, here is why" and asks for another look got
    silence, because the commit had not changed. Pushing an empty commit was
    the only remaining lever, which is a terrible thing to have to know.

    So a re-request is honoured whenever there is something new to read. The
    same FOUR endpoints `conversation()` already reads, because those are the
    places a rebuttal actually lands — a top-level comment, a reply under a
    diff line, another review, or a commit message.
    """
    if not when:
        return False
    newest = ""
    # FOUR endpoints, matching `conversation()`. I wrote "three" in the commit
    # and in CLAUDE.md and dropped the one its own docstring calls "the FOURTH,
    # and on some PRs the ONLY one": an author who cannot comment as the repo
    # owner answers in a COMMIT MESSAGE (measured on caeli-marketing#182 and
    # tests#291, where every rebuttal across five rounds lived there). Reading
    # the other three and not that one leaves those authors with no way to argue
    # at all — which is the precise failure this function was added to end.
    for base in (f"/repos/{ORG}/{repo}/issues/{pr}/comments",
                 f"/repos/{ORG}/{repo}/pulls/{pr}/comments",
                 f"/repos/{ORG}/{repo}/pulls/{pr}/reviews",
                 f"/repos/{ORG}/{repo}/pulls/{pr}/commits"):
        # PAGE THROUGH, because `per_page=100` is a ceiling and not pagination.
        # `gh()` does not paginate and ALL FOUR of these return OLDEST-first —
        # measured on infra#134, whose /commits page starts at 16:17 and ends at
        # 00:30 the next day. So the newest item, which is where a rebuttal
        # lands, is on the LAST page. Raising 30 to 100 moved the cliff rather
        # than removing it; a contested PR with 101 comments would fail exactly
        # as one with 31 did.
        #
        # A short page means the end. `page_cap` is a fuse against a pathological
        # thread, not a budget — at 100 an item it is 2,000 of them.
        page = 1
        while page <= 20:
            try:
                items = json.loads(gh(f"{base}?per_page=100&page={page}"))
            except Exception as e:  # noqa: BLE001 — unanswerable means "no reply"
                print(f"[pr-review] could not read {base.rsplit('/', 1)[-1]}: "
                      f"{type(e).__name__}")
                break
            for c in items:
                if base.endswith("/commits"):
                    # A MERGE COMMIT IS NOT A REPLY. update-branch is a
                    # `synchronize` whose merge commit has two parents and a
                    # prose body, so counting it would make every update-branch
                    # look like an argument and defeat the same-diff skip this
                    # guard was built around. Only a single-parent commit — the
                    # answer-only commit — counts.
                    if len(c.get("parents") or []) > 1:
                        continue
                    d = c.get("commit") or {}
                    who = ((c.get("author") or {}).get("login")
                           or (d.get("author") or {}).get("name") or "")
                    if who == _me() or not (d.get("message") or "").strip():
                        continue
                    newest = max(newest, (d.get("author") or {}).get("date") or "")
                    continue
                # OUR OWN posts do not count. The last review is by definition
                # newer than the one before it, so counting ours would make
                # every re-request look like a fresh argument and defeat the
                # guard entirely.
                if (c.get("user") or {}).get("login") == _me():
                    continue
                newest = max(newest, c.get("submitted_at") or c.get("created_at") or "")
            if len(items) < 100:
                break
            page += 1
    return bool(newest and newest > when)


#: The titles of the deterministic findings whose INPUT is not the diff. A
#: fix to one of these changes nothing the skip logic fingerprints — the PR
#: title is not in the diff, and a commit-message trailer changes the SHA but
#: not the diff — so without this the finding could never be cleared by
#: anything the author does. browser-extension#362: title fixed one minute
#: after the review, three later runs, all skipped as "this exact commit
#: already has a review", stale 🟡 forever.
_METADATA_FINDINGS = ("PR title does not name a ticket",
                      "agent-written commit with no session link")


def _metadata_findings_now(title, commits, body):
    return {f["title"].split(" (")[0] for f in
            checks.ticket_in_title(title) + checks.agent_session_url(commits, body)}


def _already_reviewed(repo, pr, head_sha, diff, title="", commits=(), body=""):
    """Has this exact state already been reviewed? Returns a reason, or None.

    TWO WAYS NOTHING IS NEW, and both were costing a full agent run:

      1. SAME COMMIT. A re-request while a review already exists for this head
         re-runs minutes of work to produce the same answer — and because the
         group cancels in progress, a push plus a re-request seconds apart also
         leaves a CANCELLED check that reads as a failing one.
      2. SAME DIFF. Update-branch is a `synchronize`, so merging main into an
         approved PR triggered a full re-review of a diff that did not change.
         `merge-base(base, head)..head` is identical afterwards; only the SHA
         moved.
    
    A FAILED review posts nothing, so it leaves no review at that SHA and this
    correctly declines to skip — the re-request arrow still works, which is the
    property `_release_review_request` exists to protect.
    """
    try:
        revs = json.loads(gh(f"/repos/{ORG}/{repo}/pulls/{pr}/reviews"))
    except Exception as e:  # noqa: BLE001 — unanswerable means "review it"
        print(f"[pr-review] could not read prior reviews ({type(e).__name__}) "
              "— reviewing rather than assuming nothing changed")
        return None
    me = _me()
    mine = [r for r in revs if (r.get("user") or {}).get("login") == me] if me else []
    if not mine:
        return None
    # APPROVED AT THIS COMMIT: ONLY A NEW COMMIT RE-REVIEWS. Tingyi's rule,
    # 2026-09-02, after infra#156 was approved and then re-reviewed because a
    # comment landed. A reply is "something new" when there is a finding to
    # argue with; after an approval there is nothing to argue with, and a
    # re-look can only add noise to a verdict already given. An edit, a
    # comment, a re-request — none of them change the code the approval was
    # for. A new commit does, and `synchronize` brings it here as a new head.
    if any(r.get("state") == "APPROVED" and r.get("commit_id") == head_sha
           for r in mine):
        return (f"approved at this commit ({head_sha[:7]}) — only a new "
                "commit is reviewed again")
    last_at = max((r.get("submitted_at") or "") for r in mine)
    if _someone_replied_since(repo, pr, last_at):
        print("[pr-review] someone has replied since the last review — "
              "reviewing again rather than skipping")
        return None
    last = mine[-1]
    # A METADATA FINDING THAT NO LONGER FIRES IS SOMETHING NEW. The two skips
    # below key on the commit and the diff, and neither moves when an author
    # fixes the PR title or adds a session trailer — so the finding that asked
    # for the fix could never be cleared by making it. Re-evaluate those checks
    # on the current metadata: if the last review raised one and it is now
    # satisfied, that is exactly the re-review the author was told to expect.
    posted = [m for m in _METADATA_FINDINGS if m in (last.get("body") or "")]
    if posted:
        now = _metadata_findings_now(title, commits, body)
        cleared = [m for m in posted if not any(m in n for n in now)]
        if cleared:
            print(f"[pr-review] metadata finding(s) since fixed: "
                  f"{'; '.join(cleared)} — reviewing again rather than skipping")
            return None
    if any(r.get("commit_id") == head_sha for r in mine):
        return f"this exact commit ({head_sha[:7]}) already has a review"
    want = _DIFF_MARK.format(fp=_diff_fp(diff))
    if want in (last.get("body") or ""):
        return (f"the diff is unchanged since {(last.get('commit_id') or '')[:7]} "
                "— the base moved, this PR's own changes did not")
    return None


def main():
    repo, pr = sys.argv[1], sys.argv[2]
    # BEFORE the first gh call. `_release_review_request` reads these on the way
    # out of ANY failure, and the earliest failures are the likeliest: the meta
    # fetch below is the first network call in the process, so a 5xx, an expired
    # token or a flaky DNS lookup lands here. Setting them later left exactly
    # that class releasing nothing and staying deadlocked — the state this
    # release exists to clear. Found by the AI review on the PR that added it.
    _CURRENT["repo"], _CURRENT["pr"] = repo, pr
    if not ORG:
        raise ReviewError("REVIEW_ORG is not set (and GITHUB_REPOSITORY is absent) "
                          "— the reviewer does not guess whose repository this is")
    llm.reset_usage()
    _CURRENT["wire_fields"] = []
    _CURRENT["stats"] = {}
    _CURRENT["opened"] = set()
    _CURRENT["read_ranges"] = {}
    meta = json.loads(gh(f"/repos/{ORG}/{repo}/pulls/{pr}"))
    if meta.get("draft"):
        print("draft — not reviewing")
        return
    # Cheapest possible check, and it costs nothing: the meta is already here.
    # Catches a run that starts AFTER the merge — queued behind another review
    # on the single runner, or a `synchronize` that lands as the branch merges.
    if meta.get("merged") or meta.get("state") != "open":
        print(f"PR is {'merged' if meta.get('merged') else meta.get('state')} "
              "— nothing to review")
        return
    print(f"{repo}#{pr}: {meta.get('title', '')!r} by {meta['user']['login']} "
          f"@ {meta['head']['sha'][:12]}", flush=True)
    # Known from here, so a failure anywhere below can still mark the head —
    # the pending status itself waits until the nothing-new guard has passed.
    _CURRENT["head"] = meta["head"]["sha"]
    diff, excluded, skipped = pr_diff(repo, pr)
    # Files the PR touches: what the diff shows, what the budget cut, and what
    # was skipped as generated. Anything reasoning about "does the change touch
    # this file" needs all three or it will call a changed file untouched.
    # `skipped` is a count to everything that formats it and a path list to
    # this; a stub or an older caller may hand over a bare int, so ask.
    pr_paths = list(excluded or []) + (list(skipped)
                                       if isinstance(skipped, list) else [])
    truncated = bool(excluded)
    if not diff.strip():
        print(f"nothing reviewable ({skipped} generated files skipped)")
        return
    print(f"  diff: {len(diff)} chars"
          + (f", TRUNCATED at {MAX_DIFF}" if truncated else "")
          + (f", {skipped} generated file(s) skipped" if skipped else ""), flush=True)

    # NOTHING NEW, NOTHING TO SAY — checked here, before the checkout and the
    # multi-minute agent run, because the whole point is not to spend them.
    nothing_new = _already_reviewed(repo, pr, meta["head"]["sha"], diff,
                                    title=meta.get("title") or "",
                                    commits=commit_messages(repo, pr),
                                    body=meta.get("body") or "")
    if nothing_new:
        print(f"nothing new to review: {nothing_new}")
        return

    # ON THE PR PAGE from here on. The merge box shows only the newest run of
    # a workflow, and Copilot's automatic request starts a no-op one a second
    # after ours — so the page said "all checks have passed" five minutes into
    # a real review (2026-09-03). A status context is shown regardless; a dry
    # run sets nothing (`status.set_status` enforces that for every path).
    status.pending(repo, meta["head"]["sha"])

    # So `_run_agent` can ask whether anything is superseding it without every
    # caller threading the pair through.

    with tempfile.TemporaryDirectory() as work:
        checkout(repo, meta["head"]["sha"], work)
        print(f"  agent exploring the checkout (timeout {AGENT_TIMEOUT}s)…", flush=True)
        caveats = ""
        if excluded:
            # BY SHAPE, not just by name. A list of paths was something the
            # model could open and mostly did not; how long each file is and
            # what it declares is something it can DECIDE on.
            caveats += ctx.skeletons(work, excluded)
        if skipped:
            caveats += f"[{skipped} generated/binary files omitted]\n"
        changed = _diff_paths(diff)
        # THE PROMPT GETS THE EXPANDED DIFF; everything else keeps the canonical
        # one. `_diff_paths` and the `<!-- caeli-review diff:… -->` fingerprint
        # both read `diff`, and the fingerprint is what decides whether anything
        # has changed since the last review — expanding it would make every open
        # PR look freshly changed the first time this shipped.
        shown = ctx.expand_hunks(diff, work, max_chars=int(MAX_DIFF * 1.6))
        prompt = PROMPT.format(repo=repo, path=work, diff=shown, caveats=caveats,
                               context=build_context(repo, pr, meta, work, changed,
                                                     diff, pr_paths),
                               prior=conversation(repo, pr)
                               + changed_since_last_review(
                                   repo, pr, meta["head"]["sha"],
                                   list(changed) + pr_paths))
        findings = review_findings(prompt, work, repo)
        # ONE pass that drops, corrects and adds — against the conversation that
        # already read the code, so it costs a single call and no traversal.
        findings, withdrawn = _revise(findings, work, repo)
        # After the agent's, so the model never sees them and cannot be
        # influenced into repeating or contradicting one.
        findings += checks.run_all(work, changed, title=meta.get("title") or "",
                                   commits=commit_messages(repo, pr),
                                   pr_body=meta.get("body") or "", diff=diff)

    # UNREVIEWED MEANS UNOPENED. The agent can read anything in the checkout,
    # so a file the diff had no room for is not unreviewed if the agent went
    # and read it — only the ones it never touched deserve the caveat.
    # Every pass, normalised the same way the loop records them.
    opened = _CURRENT.get("opened") or set()
    unopened = [p for p in (excluded or [])
                if os.path.normpath(p) not in opened]
    if excluded and len(unopened) < len(excluded):
        print(f"  the agent opened {len(excluded) - len(unopened)} of "
              f"{len(excluded)} file(s) the diff could not show", flush=True)
    # THREE DIFFERENT QUESTIONS, AND THEY WERE RIDING ON ONE FLAG.
    #
    #   · the caveat asks "which files did nobody look at" — `unopened`;
    #   · the stale-block dismissal asks "is the old finding still in the
    #     code" — answerable from the file AT HEAD, so `unopened` again;
    #   · the approval cap asks "did this review see every CHANGE" — and
    #     `read_file` shows the head version, not the diff. An agent that
    #     read an excluded file knows what the code says and not what the
    #     change did to it, so this one stays keyed on `excluded`.
    #
    # Collapsing the third into the first would let a review approve changes
    # it never saw, which is the exact overclaim the cap exists to prevent.
    # Found by this reviewer, on this PR.
    truncated = bool(unopened)
    saw_every_change = not excluded

    head_sha = meta["head"]["sha"]
    wire_fields = _CURRENT.get("wire_fields") or []
    body, event = _finalize_review(findings, withdrawn, truncated, skipped,
                                   head_sha=head_sha, repo=repo,
                                   wire_fields=wire_fields, diff=diff,
                                   excluded=unopened,
                                   saw_every_change=saw_every_change)

    # LAST CHECK BEFORE POSTING. The agent poll aborts a review whose PR merges
    # mid-run, but the window between the agent finishing and the POST is not
    # covered by it — and a review landing on a merged PR is noise nobody reads.
    gone = _pr_is_gone(repo, pr)
    if gone:
        raise PRClosed(f"the PR was {gone} before the review could be posted")

    # BEFORE the DRY return, so a dry run reports exactly what a real one does.
    # Putting these after it meant the two modes disagreed about their own
    # output, which is the sort of thing that makes a rehearsal worthless.
    print(f"  {len(findings)} finding(s): {severity_breakdown(findings)}", flush=True)
    print(f"  usage: {llm.usage_line()}", flush=True)

    # The findings, rendered, on the RUN page — so a run is readable without
    # opening the PR, and a review that was posted and then dismissed still has
    # a record. Best-effort: there is no summary file outside Actions, and
    # failing to write a nicety must never cost the review.
    if os.environ.get("DRY"):
        _write_step_summary(repo, pr, event, body)
        print(f"--- would post {event} ---\n{body}")
        return

    # AFTER the POST, because the POST can change both. A refused approval
    # becomes a comment carrying different text, and writing the summary first
    # left the run page reporting a clean approval GitHub had declined — the
    # very claim the fallback exists to retract.
    posted = post_review(repo, pr, event, body,
                         head_sha=head_sha, truncated=truncated)
    _write_step_summary(repo, pr, posted,
                        _verdict_withheld(body, event)
                        if posted.startswith("COMMENT (") else body)
    event = posted
    print(f"{event}: {len(findings)} finding(s)")
    status.done(repo, head_sha, event,
                f"{len(findings)} finding(s): {severity_breakdown(findings)}")


def _release_review_request(repo, pr):
    """Stop a FAILED review from deadlocking the PR.

    A run that fails posts nothing, so GitHub never drops us from
    `requested_reviewers` — and it emits no `review_requested` event when the
    reviewer is ALREADY requested. So `gh pr edit --add-reviewer` is a silent
    no-op. Pushing a commit DOES now trigger a run — the caller listens to
    `synchronize` — so that recovery path is open again on a non-draft PR, and
    on a draft it resolves to a no-op by design. This function is still what
    makes the re-request arrow work, and it is still the only recovery when the
    head is not going to move.

    MEASURED ON portal-api#150. A run failed at 21:22 and left the request in
    place; commits at 21:21 and 21:53 triggered nothing; the last review had
    read 64687d6 while the head moved to c3bfb52. The PR sat with a red check,
    two unreviewed commits and no way to ask again short of a DELETE-then-POST
    by hand. Nothing in the failure said so.

    Releasing the request makes the state HONEST — nobody is reviewing this —
    and makes the ordinary gesture work again, because the re-request arrow only
    fires while we are absent from the list.

    NOT called when superseded: that run posts the review, and dropping the
    request would cancel a review that is about to happen.

    Best-effort and quiet. It runs while the real error is on its way out and
    must never replace it.
    """
    if not repo or not pr:
        return False
    me = _me()
    if not me:
        return False
    try:
        gh(f"/repos/{ORG}/{repo}/pulls/{pr}/requested_reviewers",
           method="DELETE", body={"reviewers": [me]})
        print(f"[pr-review] released our review request on {repo}#{pr} so it can "
              "be asked again — this run posted nothing", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[pr-review] could not release the review request: "
              f"{type(e).__name__}", flush=True)
        return False


def _main_unless_superseded():
    """`main`, with cancellation demoted from an alert to a log line.

    Caught HERE rather than inside `main` so every path that can be interrupted
    is covered — the agent call, the confirmation pass, the POST — without
    threading the case through each of them.

    Exits 1 so the Actions job stays red, which is honest: this run produced no
    review. What it does not do is page anyone, because the superseding run is
    already doing the work and a human has nothing to act on.
    """
    try:
        main()
    except PRClosed as e:
        # Exit 0: there is no PR left to be red about, and a spurious red check
        # is a complaint about nothing. No release either — `requested_reviewers`
        # on a merged PR is not a deadlock anyone can hit.
        print(f"[pr-review] {e} — nothing to post, not alerting")
        raise SystemExit(0)
    except Superseded as e:
        print(f"[pr-review] superseded: {e} — the run that replaced this one "
              f"posts the review; not alerting")
        raise SystemExit(1)
    except BaseException:
        # No review was posted, so leaving ourselves in `requested_reviewers`
        # closes every way of asking again. Release it before the error goes out.
        #
        # NOT UNDER `DRY`. That mode's whole contract is that it prints what it
        # would do instead of doing it, and `CRON_DRY_RUN` is side-effect-free
        # everywhere else. A rehearsal that happened to fail would
        # otherwise DELETE a real review request on a real PR.
        if not os.environ.get("DRY"):
            _release_review_request(_CURRENT.get("repo"), _CURRENT.get("pr"))
        else:
            print("[pr-review] DRY: would release our review request", flush=True)
        raise

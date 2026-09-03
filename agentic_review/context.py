"""What the reviewer is TOLD, as opposed to what it can go and find.

The agent has tools and can read the whole checkout, so in principle none of
this is necessary. In practice the difference between a good review and a
mediocre one is almost entirely whether the model bothered to look — and the
things it skips are exactly the ones with no textual hook in the diff. Nothing
in a two-line change to a handler says "there is a CLAUDE.md forbidding this",
so the model never greps for one.

Measured against GitHub Copilot on four of our PRs: ours out-found it everywhere
(9 findings to 2, 4 to 0, 1 to 0) EXCEPT on constraints that live outside the
diff, where a tool that had been handed the repository's conventions beat one
that had to think of asking. That gap is deterministic, so it is closed
deterministically here rather than by asking the model more nicely.

Every section is budgeted, and every truncation SAYS SO in the text the model
reads, followed by where to find the rest. A silent truncation teaches the model
that it has seen everything.
"""
import json
import os
import re
import subprocess

from .config import CONVENTION_DOCS, ORG

#: Per-document and total caps for the convention docs. A repository's CLAUDE.md
#: can be 100KB; pasting it whole would crowd out the diff it is meant to
#: explain, and the agent can read the rest on demand.
MAX_DOC = 14_000
MAX_DOCS_TOTAL = 30_000

#: The repository map is orientation, not an inventory.
MAX_MAP_ENTRIES = 220

#: Linked PRs: title, state and touched files. Bodies are capped hard because a
#: linked PR's description is background on background.
MAX_LINKED_PRS = 4
MAX_LINKED_BODY = 700

#: `#123` in prose, and the full URL form. The URL form carries the repo, so a
#: cross-repo link is recognised as one rather than fetched from the wrong repo.
_PR_URL_RE = re.compile(
    r"https?://github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)")
_PR_HASH_RE = re.compile(r"(?<![\w/])#(\d+)\b")


def _git(work, *args):
    try:
        p = subprocess.run(["git", "-C", work, *args], capture_output=True,
                           text=True, timeout=60)
        return p.stdout if p.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _read(work, rel, limit=MAX_DOC):
    path = os.path.join(work, rel)
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            body = fh.read()
    except OSError:
        return None
    if len(body) <= limit:
        return body
    return (body[:limit] +
            f"\n\n[... {len(body) - limit} more chars — read `{rel}` for the rest]")


def convention_docs(work, changed_paths):
    """The repository's own rules, root-level and per-directory.

    BOTH, because the per-directory ones are where the specific rules live and
    they are the easiest to miss: a monorepo's `services/api/CLAUDE.md` binds
    exactly the code this PR touches and nothing in the diff points at it.

    Ordered by CONVENTION_DOCS, so a repo that keeps its rules in AGENTS.md or
    CONTRIBUTING.md needs configuration and not a fork.
    """
    seen, out, total = set(), [], 0
    # Root first: it is the one a human would read first, and if the budget runs
    # out it is the one worth having.
    dirs = [""]
    for path in changed_paths:
        d = os.path.dirname(path)
        while d and d not in dirs:
            dirs.append(d)
            d = os.path.dirname(d)
    for directory in dirs:
        for name in CONVENTION_DOCS:
            rel = os.path.join(directory, name) if directory else name
            if rel in seen:
                continue
            seen.add(rel)
            if total >= MAX_DOCS_TOTAL:
                continue
            body = _read(work, rel, min(MAX_DOC, MAX_DOCS_TOTAL - total))
            if body is None or not body.strip():
                continue
            total += len(body)
            out.append((rel, body))
    if not out:
        return ""
    blocks = "\n\n".join(f"--- {rel} ---\n{body}" for rel, body in out)
    return ("\nTHE REPOSITORY'S OWN RULES. These are AUTHORITATIVE and they record\n"
            "decisions that already have reasons. If the diff contradicts one,\n"
            "that is a finding and you should QUOTE the rule. If the diff looks\n"
            "odd and one of these explains why, that is NOT a finding — say\n"
            "nothing. A rule you cannot find here is not a rule; do not invent\n"
            "conventions from the code's general shape.\n\n" + blocks + "\n")


def repo_map(work, max_entries=MAX_MAP_ENTRIES):
    """Directories and how much is in them — orientation, not an inventory.

    The question this answers is the one behind every duplication finding:
    "where would this already live if it existed?" A model that cannot see the
    shape of the tree greps for the name it expects, fails to find it, and
    concludes nothing exists — which is how a duplication finding gets MISSED
    and, worse, how "this should be extracted to a shared helper" gets proposed
    for a helper that is already there under another name.
    """
    listing = _git(work, "ls-files")
    if not listing.strip():
        return ""
    counts = {}
    for path in listing.splitlines():
        directory = os.path.dirname(path) or "."
        counts[directory] = counts.get(directory, 0) + 1
    rows = sorted(counts.items())
    total_dirs = len(rows)
    if total_dirs > max_entries:
        # Keep the LARGEST directories: a tree pruned alphabetically loses
        # whatever sorts late, which is arbitrary. Re-sorted by path afterwards
        # so it still reads as a tree.
        rows = sorted(sorted(counts.items(), key=lambda kv: -kv[1])[:max_entries])
    lines = [f"{d}/  ({n} file{'s' if n != 1 else ''})" for d, n in rows]
    note = ("" if total_dirs <= max_entries else
            f"\n[{total_dirs - max_entries} smaller directories omitted — "
            f"use list_files to explore]")
    return ("\nTHE REPOSITORY, BY DIRECTORY. Use it to decide WHERE something\n"
            "would already live before claiming it does not exist.\n\n"
            + "\n".join(lines) + note + "\n")


def linked_pr_refs(repo, github_texts=(), url_only_texts=()):
    """(repo, number) for every pull request mentioned, de-duplicated.

    TWO KINDS OF SOURCE, and conflating them produced garbage on the first live
    run. `#123` is GITHUB syntax: in a PR description it is an autolink and
    means a pull request. In a Jira description it means nothing — it is a
    heading, an ordinal, a version, a channel — and reading slack-app#381's
    three tickets that way produced `#4`, `#3` and `#121`, none of which had
    anything to do with the change.

    So a PR body is a `github_text` and gets both forms; ticket prose is a
    `url_only_text` and contributes only what is unambiguously a link.

    A bare `#123` means THIS repository; a URL carries its own. Getting that
    wrong is not cosmetic — it fetches a real, unrelated PR from another repo
    and presents it as context for this one.
    """
    if isinstance(github_texts, str):
        github_texts = [github_texts]
    if isinstance(url_only_texts, str):
        url_only_texts = [url_only_texts]
    github_texts = [t or "" for t in github_texts]
    url_only_texts = [t or "" for t in url_only_texts]
    refs, seen, claimed = [], set(), set()
    # URLs first, ACROSS ALL TEXTS, because a number a URL has already claimed
    # must not then be read as a bare `#n` in this repo. The case is ordinary
    # markdown: `[#378](https://github.com/org/slack-app/pull/378)` names one
    # pull request and would otherwise produce two — the real one, and a
    # same-numbered PR in the repo under review that is somebody else's change
    # entirely. Fetching THAT and calling it context is worse than missing a
    # link, because it is wrong rather than absent.
    for text in github_texts + url_only_texts:
        for owner, name, number in _PR_URL_RE.findall(text):
            claimed.add(int(number))
            if owner.lower() != ORG.lower():
                continue  # somebody else's repo; we cannot read it anyway
            key = (name, int(number))
            if key not in seen:
                seen.add(key)
                refs.append(key)
    for text in github_texts:
        for number in _PR_HASH_RE.findall(_PR_URL_RE.sub(" ", text)):
            if int(number) in claimed:
                continue
            key = (repo, int(number))
            if key not in seen:
                seen.add(key)
                refs.append(key)
    return refs


def linked_prs(refs, fetch, skip=(), limit=MAX_LINKED_PRS):
    """Fetch each linked PR's title, state and touched files.

    `fetch(path)` is injected rather than imported so this is testable without a
    network, and so a caller can cache. Failures are skipped silently for the
    same reason the tracker's are: a dead link is one less piece of evidence,
    never a reason to abandon the review.

    The FILES are the point. "Has somebody already changed this?" and "does this
    conflict with the other half of the same work?" are both answered by a path
    list and neither is answered by a description.
    """
    out = []
    for repo, number in refs:
        if len(out) >= limit:
            break
        if (repo, number) in skip:
            continue
        try:
            meta = json.loads(fetch(f"/repos/{ORG}/{repo}/pulls/{number}"))
            files = json.loads(
                fetch(f"/repos/{ORG}/{repo}/pulls/{number}/files?per_page=100"))
        except Exception as e:  # noqa: BLE001 — a dead link is not a failure
            print(f"[review] could not read {repo}#{number}: {type(e).__name__}: {e}")
            continue
        state = ("merged" if meta.get("merged") else meta.get("state") or "?")
        out.append({
            "repo": repo, "number": number, "state": state,
            "title": meta.get("title") or "",
            "body": (meta.get("body") or "")[:MAX_LINKED_BODY],
            "files": [f.get("filename", "") for f in files][:40],
        })
    if not out:
        return ""
    blocks = []
    for pr in out:
        lines = [f"### {pr['repo']}#{pr['number']} [{pr['state']}] — {pr['title']}"]
        if pr["body"].strip():
            lines.append(pr["body"])
        if pr["files"]:
            lines.append("files: " + ", ".join(pr["files"]))
        blocks.append("\n".join(lines))
    return ("\nPULL REQUESTS THIS ONE REFERS TO. A MERGED one is the state of the\n"
            "world this change lands on; an OPEN one is work in flight that may\n"
            "conflict with it. If this PR is the second half of a pair, check\n"
            "that the halves actually agree — the field names, the order of\n"
            "deploy, the default when only one side has shipped.\n\n"
            + "\n\n".join(blocks) + "\n")


def build(work, changed_paths, tracker_section="", linked_section=""):
    """Assemble the whole context block, in the order a human would read it."""
    parts = [convention_docs(work, changed_paths), tracker_section,
             linked_section, repo_map(work)]
    return "".join(p for p in parts if p)


# --------------------------------------------------------------------------
# Dynamic context expansion
# --------------------------------------------------------------------------
# A diff hunk carries three lines either side, which is enough to apply a patch
# and not enough to judge one. The reviewer then spends tool calls rediscovering
# the function it is standing in — and tool calls are the scarce resource: on
# slack-app#380 the agent made 64 of them, filled its transcript, and had no room
# left to think.
#
# So each hunk is grown UP TO ITS ENCLOSING DEFINITION before the diff is shown.
# pr-agent does the same thing (`allow_dynamic_context`,
# `max_extra_lines_before_dynamic_context = 10`) and calls it dynamic context;
# the idea is theirs. Ours is cheaper and more accurate for one reason: they
# fetch the surrounding lines from the provider's raw-file API, and we already
# have the checkout on disk, so there is no network call and no chance of
# reading a different revision than the one under review.
#
# THE EXPANDED DIFF IS FOR THE PROMPT ONLY. `pr_diff`'s output stays canonical
# for `_diff_paths` and for the `<!-- caeli-review diff:… -->` fingerprint that
# decides whether anything has changed since the last review — expanding that
# would make every review look like a new diff the first time this shipped, and
# would re-review every open PR once.

#: How far up to look for the enclosing definition.
MAX_EXTRA_BEFORE = 12

#: Fixed padding below the hunk. Small: what follows a change is usually the
#: rest of the same statement, and the lines that matter are above it.
EXTRA_AFTER = 3

#: Lines that plausibly open a definition, across the languages in these repos.
#: Deliberately NOT `const`/`let`/`var` — those match half of any JS file and
#: would anchor to a neighbouring assignment instead of the enclosing function.
_DEF_RE = re.compile(
    r"^\s*(?:@|"                                   # a decorator opens a def
    r"(?:async\s+)?def\s|class\s|"                 # python
    r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s|"
    r"(?:export\s+)?(?:abstract\s+)?class\s|"
    r"(?:export\s+)?interface\s|(?:export\s+)?type\s|"
    r"func\s|fn\s|impl\s|"                         # go / rust
    r"(?:public|private|protected|static)\s|"      # java / c#
    r"describe\s*\(|it\s*\(|test\s*\("             # test suites
    r")")

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def _indent(line):
    return len(line) - len(line.lstrip())


def _enclosing_start(lines, hunk_start_idx, limit=MAX_EXTRA_BEFORE):
    """How many lines above `hunk_start_idx` reach the enclosing definition.

    Returns 0 when nothing definition-like is found, rather than padding
    blindly: an arbitrary ten lines of a neighbouring function is noise that
    costs the same budget as the real thing.
    """
    body = None
    for i in range(hunk_start_idx, min(hunk_start_idx + 6, len(lines))):
        if lines[i].strip():
            body = _indent(lines[i])
            break
    if body is None:
        return 0
    for back in range(1, min(limit, hunk_start_idx) + 1):
        line = lines[hunk_start_idx - back]
        if not line.strip():
            continue
        # Less indented AND declaration-shaped. Indentation alone anchors to a
        # closing brace; the pattern alone anchors to a nested helper.
        if _indent(line) < body and _DEF_RE.match(line):
            return back
    return 0


def expand_hunks(diff, work, max_chars=None):
    """Grow every hunk up to its enclosing definition. Never raises.

    Falls back to the original diff whenever a file cannot be read (deleted in
    this PR, binary, or simply absent) or the result would blow the budget — a
    diff that is merely un-expanded is a working diff.
    """
    if not diff.strip():
        return diff
    out, cache = [], {}
    path, lines = None, None
    i, src = 0, diff.splitlines()
    while i < len(src):
        line = src[i]
        if line.startswith("+++ b/"):
            path = line[6:].strip()
            if path not in cache:
                try:
                    with open(os.path.join(work, path), encoding="utf-8",
                              errors="replace") as fh:
                        cache[path] = fh.read().splitlines()
                except OSError:
                    cache[path] = None
            lines = cache[path]
            out.append(line)
            i += 1
            continue
        m = _HUNK_RE.match(line) if line.startswith("@@") else None
        if not m or not lines:
            out.append(line)
            i += 1
            continue
        old_start, old_count, new_start, new_count, tail = m.groups()
        old_start, new_start = int(old_start), int(new_start)
        old_count = int(old_count) if old_count is not None else 1
        new_count = int(new_count) if new_count is not None else 1
        # Collect the hunk body so it can be re-emitted after the new header.
        j = i + 1
        while j < len(src) and not src[j].startswith(("@@", "diff --git", "--- ",
                                                      "+++ ")):
            j += 1
        body = src[i + 1:j]

        idx = new_start - 1                      # 0-based index of the first line
        before = _enclosing_start(lines, idx) if 0 <= idx < len(lines) else 0
        before = min(before, old_start - 1, new_start - 1)
        end = new_start - 1 + new_count          # 0-based, one past the hunk
        after = max(0, min(EXTRA_AFTER, len(lines) - end))

        pre = [" " + lines[idx - before + k] for k in range(before)]
        post = [" " + lines[end + k] for k in range(after)]
        out.append(f"@@ -{old_start - before},{old_count + before + after} "
                   f"+{new_start - before},{new_count + before + after} @@{tail}")
        out.extend(pre)
        out.extend(body)
        out.extend(post)
        i = j
    expanded = "\n".join(out)
    if max_chars and len(expanded) > max_chars:
        # The un-expanded diff was already sized to fit. Growing it past the cap
        # would truncate the LAST files entirely, which is strictly worse than
        # showing every file with less context.
        return diff
    return expanded


# --------------------------------------------------------------------------
# CI results on the commit under review
# --------------------------------------------------------------------------
# Tingyi's suggestion, 2026-09-02. The author's own test suite has usually run
# on the same commit by the time the review does, and its verdict is the
# cheapest evidence there is: a failing test is a finding-grade fact the model
# would otherwise have to rediscover, and a passing suite that exercises the
# changed code is a reason not to invent a doubt about it.
#
# THE ACTIONS API, NOT CHECK-RUNS. A fine-grained PAT reads `/actions/runs` and
# `/actions/runs/{id}/jobs` with `Actions: Read-only`; the check-runs endpoint
# wants a `Checks` permission that the token's UI did not even offer. Measured
# on the bot's fine-grained token: check-runs and status 403, actions/runs 200. Jobs carry
# name, status and conclusion, which is everything the section needs, and a
# failed job's log is where the failing test names are.
#
# TIMING IS THE CATCH. The review and the unit job are triggered by the same
# push and run side by side, so at review start the suite is often still in
# progress. That is reported as PENDING, never as absent — "no test results"
# and "tests still running" mean different things to a reviewer.

#: Our own workflow and its jobs, plus anything else that is a review rather
#: than a test. Matched on the WORKFLOW name too: the reviewer's job is called
#: `review (Caeli)` and its workflow `PR review`.
_NOT_A_TEST_RE = re.compile(r"review|copilot|codeql|dependabot|label|lint$", re.I)

#: What a failure actually looks like in a runner log, MEASURED on
#: caeli-marketing job 100095120836 (vitest) rather than guessed. The first
#: pattern extracted nothing from it, for two reasons the synthetic test could
#: not show: the output is ANSI-coloured, so `×` never sits at a line start;
#: and the reliable signal is the runner's own `##[error]` annotation, which the
#: pattern did not allow for. Stripped and matched after:
#:
#:     ##[error]TypeError: answerRef.current?.scrollIntoView is not a function
#:      ❯ components/answer/answer-home.tsx:241:28
#:      Errors  1 error            (vitest summary; also "Tests  2 failed")
#:     ⎯⎯ Uncaught Exception ⎯⎯
#:     FAILED tests/test_x.py::test_a   (pytest)     not ok 4 - name  (node:test)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_STAMP_RE = re.compile(r"^\S+T\S+Z ")
_FAIL_LINE_RE = re.compile(
    r"^\s*(?:"
    r"##\[error\](?!Process completed).+"          # runner annotation, not the exit
    r"|(?:Test Files|Tests|Errors)\s+\d+ (?:failed|errors?)\b.*"   # vitest/jest summary
    r"|⎯+ *(?:Uncaught Exception|Unhandled Rejection|Failed Tests?) *⎯+.*"
    r"|(?:FAIL|×|✗|✕)\s+\S.+"                        # a failed test line
    r"|FAILED \S.+|ERROR \S.+"                        # pytest
    r"|not ok \d+ .+"                                  # node:test TAP
    r"|(?:Assertion|Type|Reference|Range|Syntax)Error: .+"
    r")$", re.M)
#: The location frame that follows an error — worth one line, not a stack.
_FRAME_RE = re.compile(r"^\s*❯ (\S+:\d+(?::\d+)?)\s*$", re.M)

#: How much failure evidence the model sees per job.
MAX_FAIL_LINES = 12
MAX_LOG_TAIL = 200_000


def failure_lines(log):
    """The lines of a runner log that say what failed, cleaned and bounded."""
    tail = (log or "")[-MAX_LOG_TAIL:]
    clean = "\n".join(_ANSI_RE.sub("", _STAMP_RE.sub("", l))
                      for l in tail.splitlines())
    seen, out, want_frame = set(), [], False
    for line in clean.splitlines():
        m = _FAIL_LINE_RE.match(line)
        if m:
            text = " ".join(line.split())[:220]
            if text not in seen:
                seen.add(text)
                out.append(text)
            # An error line is followed by its location; keep the first frame.
            want_frame = "Error" in text or text.startswith("##[error]")
        elif want_frame:
            f = _FRAME_RE.match(line)
            if f:
                loc = f"    at {f.group(1)}"
                if loc not in seen:
                    seen.add(loc)
                    out.append(loc)
                want_frame = False
        if len(out) >= MAX_FAIL_LINES:
            break
    return out


def _job_failures(repo, job_id, fetch_log):
    """Failing-test lines from a job's log, or []. Never raises."""
    try:
        log = fetch_log(f"/repos/{ORG}/{repo}/actions/jobs/{job_id}/logs")
    except Exception as e:  # noqa: BLE001
        print(f"[review] could not read job {job_id} log: {type(e).__name__}: {e}")
        return []
    return failure_lines(log)


def check_results(repo, sha, fetch, fetch_log=None):
    """CI jobs on `sha`, as a prompt section. Never raises.

    `fetch(path)` returns JSON text; `fetch_log(path)` returns a job's log as
    text (it lives behind a redirect to blob storage, which is why it is a
    separate callable). Both injected so this is testable without a network. A
    failed fetch is one less piece of evidence, not a reason to abandon the
    review.
    """
    try:
        data = json.loads(fetch(f"/repos/{ORG}/{repo}/actions/runs"
                                f"?head_sha={sha}&per_page=30"))
        runs = data.get("workflow_runs") or []
    except Exception as e:  # noqa: BLE001
        hint = ""
        if "403" in str(e):
            hint = " — the review token needs the `Actions: Read-only` permission"
        print(f"[review] could not read workflow runs: {type(e).__name__}: {e}{hint}")
        return ""
    lines, pending = [], []
    for run in runs:
        wf = run.get("name") or "?"
        if _NOT_A_TEST_RE.search(wf):
            continue
        try:
            jobs = json.loads(fetch(f"/repos/{ORG}/{repo}/actions/runs/"
                                    f"{run.get('id')}/jobs")).get("jobs") or []
        except Exception as e:  # noqa: BLE001
            print(f"[review] could not read jobs for run {run.get('id')}: {e}")
            continue
        for job in jobs:
            name = job.get("name") or "?"
            if _NOT_A_TEST_RE.search(name):
                continue
            label = f"{wf} / {name}"
            if job.get("status") != "completed":
                pending.append(label)
                continue
            conclusion = job.get("conclusion") or "?"
            line = f"- {label}: {conclusion}"
            if conclusion in ("failure", "timed_out", "cancelled") and fetch_log:
                for fl in _job_failures(repo, job.get("id"), fetch_log):
                    line += f"\n    {fl}"
            lines.append(line)
    if not lines and not pending:
        return ""
    head = ("\nCI ON THIS COMMIT. A failing test in the author's own suite is a "
            "finding — name the test and what it proves. A passing suite that "
            "exercises the changed code is evidence against a doubt you "
            "cannot substantiate; do not invent a failure it would have "
            "caught.\n\n")
    body = "\n".join(lines)
    if pending:
        body += ("\n- still running when this review started: "
                 + ", ".join(pending)
                 + " — treat their coverage as unknown, not as absent.")
    return head + body + "\n"

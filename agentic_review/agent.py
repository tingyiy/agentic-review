"""A small, observable tool loop for reviewing a checkout.

This replaces the one-shot `hermes -p cron -z <prompt>` the PR reviewer used to
shell out to. The reasons are recorded in `cron/docs/native-reviewer-plan.md`;
the operative one is that hermes is opaque BY CONSTRUCTION here — `-z` writes
only the final answer, emits nothing on stderr, and logs no tool calls — so when
four reviews burned exactly 901s on 2026-09-01 and died at the cap, there was
nothing to read. Not a slow tail: healthy runs on the same branch finished in
147-259s and nothing landed in between. Stuck, and undiagnosable.

So the contract of this module is OBSERVABILITY FIRST. Every turn prints its
number, the tool, the arguments, the latency and the finish reason before the
next one starts, and the whole transcript is available afterwards on the
exception as well as on success. A hang here names the turn it hung on.

The tools are deliberately three and read-only: `read_file`, `grep`,
`list_files`. A reviewer needs to answer "does this already exist", "what does
this helper actually do" and "who else reads this field". It never needs to
write, run or fetch anything, and every capability past that is a way for a PR
under review to reach the machine reviewing it — which on a self-hosted runner
is Mini itself.
"""
import json
import os
import os.path
import re
import subprocess
import time

from . import llm
from .errors import ReviewError

#: How much of one tool's output the model may see. A `grep` across a large repo
#: can return thousands of lines, and the failure it causes is not an error: it
#: is a context window quietly consumed by one match list, leaving no room for
#: the file the model actually needed. Truncation is reported IN the result, so
#: the model knows to narrow rather than concluding it has seen everything.
#:
#: Halved from 12,000 after slack-app#380 measured what the old value did: 64
#: tool calls returning 187,526 chars, median 2,231 and max 12,099 — the cap
#: itself, hit by whole-file reads. The agent filled the transcript budget at
#: turn 29, was forced to answer, and returned 77 characters after reading every
#: changed file. It drowned rather than ran out.
MAX_TOOL_CHARS = 6_000

#: Guard on the whole conversation, not one reply. Tool results accumulate in
#: `messages` and are re-sent every turn, so an unbounded loop grows its own
#: prompt quadratically: the cost of turn N includes the sum of turns 1..N-1.
#: This is a budget, not a limit — when it is exceeded the model is asked to
#: answer now rather than cut off.
#:
#: 240,000 WAS A GUESS AND IT WAS WRONG BY MORE THAN TENFOLD. deepseek-v4-flash
#: reports `context_length: 1048576` TOKENS; 240,000 chars is roughly 60,000
#: tokens, so the loop was being forced to answer with 94% of the window unused.
#: That is the direct cause of the two worst results measured: slack-app#380
#: round 4 answered in 77 characters after reading every changed file, and both
#: passes on caeli-marketing#212 were forced at turns 19 and 22.
#:
#: NOT RAISED TO THE CEILING, because the ceiling is not the constraint that
#: matters. Every turn re-sends the conversation, so a transcript twice as long
#: costs more than twice as much across a loop — #212 spent 1.75M prompt tokens
#: at the OLD budget. 600,000 chars (~150k tokens, ~15% of context) clears every
#: transcript observed so far with room to spare; raising it further should be
#: paid for by a measurement, not by the fact that the model would allow it.
MAX_TRANSCRIPT_CHARS = int(os.environ.get("REVIEW_MAX_TRANSCRIPT", 600_000))

#: Room for the final answer. The reviewer's answer is a JSON array of findings,
#: each with a two-line fix sketch — larger than a typical completion, and 8192
#: truncated a real review of a 4-file PR at turn 11 after ten good turns of
#: exploration. Truncation is fatal (see `llm.chat`), so a budget that is too
#: small does not degrade the review, it destroys it.
MAX_TOKENS = int(os.environ.get("REVIEW_MAX_TOKENS", 16_384))

#: A defensive cap on the loop itself. The real bound is the deadline; this
#: catches a model that has stopped making progress but is still calling tools
#: quickly (re-reading the same file, walking a directory tree one level at a
#: time) — which burns the deadline without needing anything like the whole of
#: it. 40 turns is ~3x what a healthy review used.
MAX_TURNS = 40

#: Per-request ceiling. The loop deadline bounds the whole review; this bounds
#: ONE call, so a single hung HTTP request cannot eat the entire budget in
#: silence. Deliberately shorter than any sane loop deadline.
REQUEST_TIMEOUT = 180


class AgentError(ReviewError):
    """The loop could not produce an answer.

    Carries `transcript` — the per-turn log — because the whole point of this
    module is that a failure says what it was doing. The caller prints it; it
    must never be necessary to reproduce a hang to find out where it was.
    """

    def __init__(self, message, transcript=()):
        super().__init__(message)
        self.transcript = list(transcript)


class Timeout(AgentError):
    """Wall clock exhausted. A distinct type because the caller's response
    differs: a timeout is worth retrying with a smaller task, a malformed reply
    is not."""


class Workspace:
    """The checkout, and the only thing the tools can see.

    Containment is `realpath`-based and checked on the RESOLVED path, not the
    argument. Rejecting `..` textually is not enough: a checkout can contain a
    symlink (ours do — `node_modules` links, and any repo may add one) whose
    target is outside the tree, and `open()` follows it happily. The argument
    `docs/link/etc/passwd` contains no `..` at all.
    """

    def __init__(self, root):
        self.root = os.path.realpath(root)

    def resolve(self, path):
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path is required")
        if os.path.isabs(path):
            raise ValueError(f"absolute paths are not allowed: {path}")
        full = os.path.realpath(os.path.join(self.root, path))
        # `commonpath` rather than `startswith`: a sibling directory named
        # `/tmp/work-evil` starts with `/tmp/work` as a string and is nowhere
        # near it as a path.
        if full != self.root and os.path.commonpath([full, self.root]) != self.root:
            raise ValueError(f"path escapes the checkout: {path}")
        return full


def _record_opened(stats, name, raw, result):
    """Track how much of a file `read_file` has actually shown, in `stats`.

    A path counts as OPENED once the windows the agent asked for cover the
    whole file. Four ways this has been got wrong, each found by review:

      · a read that FAILED still counted, so the caveat dropped a file
        nobody had seen;
      · the raw spelling was stored, so `./src/big.py` never matched
        `src/big.py`;
      · a 200-line WINDOW counted as the file, which let a 255-line file
        clear the caveat after four fifths of it — and the footer that says
        so is appended, so `_truncate` can cut it off;
      · then the correction went too far: the normal way to read a 255-line
        file is two windows, `read_file(p)` then `read_file(p, offset=201)`,
        and neither one alone is the file. Requiring a single top-to-end read
        made a fully-read file report as unopened, which is a FALSE caveat on
        every large file — the opposite failure, and just as wrong.

    So the ranges are unioned. The footer names the total, which is what makes
    "have we covered it all" answerable at all.
    """
    text = str(result or "")
    if name != "read_file" or text.startswith(TOOL_ERROR_PREFIX):
        return
    # A CLIPPED RESULT SHOWS LESS THAN IT SAYS. `_truncate` keeps the head and
    # drops the tail, footer included, so the window's own end is unknown.
    if _CLIPPED_RESULT.search(text):
        return
    try:
        args = json.loads(raw or "{}") or {}
        path = args.get("path")
        offset = int(args.get("offset") or 1)
    except (ValueError, TypeError):
        return
    if not isinstance(path, str) or not path.strip() or offset < 1:
        return
    path = os.path.normpath(path.strip())

    seen = stats.setdefault("_read_ranges", {}).setdefault(
        path, {"total": None, "covered": []})
    window = _PARTIAL_READ.search(text)
    if window:
        start, end, total = (int(g) for g in window.groups())
        seen["total"] = total
        seen["covered"].append((start, end))
    else:
        # No footer: this window ran to the end of the file.
        seen["covered"].append((offset, None))

    total = seen["total"]
    if total is None:
        # Never partial and started at the top — the whole file in one read.
        if any(start == 1 and end is None for start, end in seen["covered"]):
            stats.setdefault("opened", set()).add(path)
        return
    reach = 0
    for start, end in sorted((s, e if e is not None else total)
                             for s, e in seen["covered"]):
        if start > reach + 1:
            return                      # a gap: something in the middle is unseen
        reach = max(reach, end)
    if reach >= total:
        stats.setdefault("opened", set()).add(path)


def _truncate(text, limit=MAX_TOOL_CHARS):
    if len(text) <= limit:
        return text
    # Head, not tail. A grep's first matches are as good as any, and a file's
    # head carries the imports and signatures that place everything else. The
    # note is inside the returned string so the model reads it as data.
    return (text[:limit]
            + f"\n\n[... truncated: {len(text) - limit} more chars. "
              f"Narrow the pattern, or read a specific range with offset/limit.]")


#: Lines returned when the caller does not ask for a range.
#:
#: A WINDOW, NOT THE WHOLE FILE — the ACI lesson, and the fix for the failure
#: above. SWE-agent measured 18.0% vs 11.0% on SWE-bench Lite from interface
#: design alone, and names the principle: "environment feedback should be
#: informative but concise… without unnecessary details". A 400-line module
#: returned whole is 12,000 characters of which the model needed forty lines,
#: and it pays that toll on every read until it has no room left to think.
DEFAULT_READ_LINES = 200


def _read_file(ws, path, offset=None, limit=None):
    full = ws.resolve(path)
    if os.path.isdir(full):
        raise ValueError(f"{path} is a directory — use list_files")
    with open(full, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    start = max(int(offset or 1), 1)
    count = int(limit) if limit else DEFAULT_READ_LINES
    chunk = lines[start - 1:start - 1 + count]
    if not chunk:
        return f"{path} has {len(lines)} lines; nothing at offset {start}"
    # Numbered, because every finding must carry a line and a model counting
    # newlines itself gets it wrong — and a finding whose line is wrong sends
    # the author to the wrong code and reads as a hallucination.
    body = "".join(f"{start + i}\t{line}" for i, line in enumerate(chunk))
    shown = start + len(chunk) - 1
    if shown < len(lines):
        # SAY WHAT IS LEFT, and how to get it. A window with no footer is
        # indistinguishable from a whole file, and a model that thinks it has
        # read the file stops looking — which is a worse failure than the cost
        # of the extra call.
        body += (f"\n[showing lines {start}-{shown} of {len(lines)}. "
                 f"Call again with offset={shown + 1} for the rest.]")
    return _truncate(body)


#: Lines of surrounding code returned with each match.
#:
#: A BARE MATCH LINE COSTS A SECOND ROUND-TRIP, and the second trip is worse
#: than the first: the model has a line number and no way to know how much of
#: the function around it matters, so it pulls a whole window — 200 lines to see
#: five. That is how slack-app#380 accumulated 242,000 characters of transcript
#: across 64 calls and then had no room left to think.
#:
#: Two is deliberately small. The purpose is to answer "does this match mean
#: what I think it means" — a signature, the `if` above a `return`, the line a
#: value is assigned on — not to substitute for reading the function.
GREP_CONTEXT = 2

#: Matches returned WITH context. Lower than the bare-line cap, because each
#: match now costs ~5 lines instead of 1: 40 x 5 is the same budget 200 bare
#: lines used, spent on evidence rather than on an index.
GREP_MAX_MATCHES = 40


def _grep(ws, pattern, path=None, context=GREP_CONTEXT,
          max_matches=GREP_MAX_MATCHES):
    """Search, and return each hit WITH the lines around it.

    The pattern comes from watching how a human — and Claude Code's own Read
    tool — actually works: locate, then look at the few lines either side, and
    only open the whole region when those lines say it matters. Returning a bare
    `path:line:text` forces a guess about where the interesting region starts.

    `-C` also makes `git grep` group hits per file with `--` separators, which
    is the shape the model needs anyway: several matches in one file read as one
    piece of evidence rather than as several unrelated ones.
    """
    where = ws.resolve(path) if path else ws.root
    ctx = max(0, min(int(context), 10))
    # `git grep` first: it searches TRACKED files only, so it skips
    # `node_modules`, build output and the `.git` directory without a list of
    # exclusions to maintain. On this repo that is the difference between a
    # readable answer and 40,000 lines of vendored JavaScript.
    for cmd in (["git", "grep", "-n", "-I", "--no-color", f"-C{ctx}",
                 "-e", pattern, "--"],
                ["grep", "-rn", "-I", f"-C{ctx}", "--exclude-dir=.git",
                 "--exclude-dir=node_modules", "-e", pattern]):
        try:
            args = list(cmd)
            if cmd[0] == "git":
                args.append(os.path.relpath(where, ws.root) or ".")
            else:
                args.append(where)
            p = subprocess.run(args, cwd=ws.root, capture_output=True, text=True,
                               timeout=60)
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"grep failed: {type(e).__name__}: {e}"
        # 1 is "no matches" for both tools and is a real answer, not a failure.
        if p.returncode in (0, 1):
            out = p.stdout.strip()
            if not out:
                return f"no matches for {pattern!r}"
            # Count MATCHES, not lines. With context on, a line count would cap
            # at eight hits and call it "200 results", which tells the model to
            # narrow a pattern that was already fine. A match line carries `:`
            # after the number; a context line carries `-`.
            blocks, matches = [], 0
            for line in out.splitlines():
                if line == "--":
                    blocks.append(line)
                    continue
                if re.match(r"^[^:]+:\d+:", line):
                    matches += 1
                    if matches > max_matches:
                        break
                blocks.append(line)
            body = "\n".join(blocks)
            if matches > max_matches:
                body += (f"\n[... more than {max_matches} matches — narrow the "
                         f"pattern or pass a path]")
            return _truncate(body)
        # git grep exits 128 outside a repository; fall through to plain grep.
    return f"grep failed: {p.stderr.strip()[:300]}"


def _list_files(ws, directory="."):
    full = ws.resolve(directory)
    if not os.path.isdir(full):
        raise ValueError(f"{directory} is not a directory")
    entries = []
    for name in sorted(os.listdir(full)):
        if name == ".git":
            continue
        kind = "/" if os.path.isdir(os.path.join(full, name)) else ""
        entries.append(f"{name}{kind}")
    if not entries:
        return f"{directory} is empty"
    return _truncate("\n".join(entries))


TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": (f"Read a file from the checkout, with line numbers. "
                        f"Returns {DEFAULT_READ_LINES} lines from `offset`; the "
                        f"result says when there is more and where to continue."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string",
                     "description": "Path relative to the repository root."},
            "offset": {"type": "integer", "description": "First line (1-based)."},
            "limit": {"type": "integer",
                      "description": f"How many lines to read "
                                     f"(default {DEFAULT_READ_LINES})."},
        }, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep",
        "description": (f"Search the repository for a regular expression. "
                        f"Returns each match with {GREP_CONTEXT} lines of "
                        f"surrounding code, so you can usually judge it without "
                        f"a separate read. Tracked files only."),
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "A POSIX regex."},
            "path": {"type": "string",
                     "description": "Optional file or directory to limit the search to."},
            "context": {"type": "integer",
                        "description": f"Lines of context each side "
                                       f"(default {GREP_CONTEXT}, max 10)."},
        }, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List the entries of a directory in the checkout.",
        "parameters": {"type": "object", "properties": {
            "directory": {"type": "string",
                          "description": "Path relative to the repository root."},
        }, "required": []}}},
]

_DISPATCH = {"read_file": _read_file, "grep": _grep, "list_files": _list_files}


#: `_read_file`'s footer when a window did not reach the end of the file, and
#: `_truncate`'s note when the result itself was clipped. Both pair with the
#: code that writes them — change one, change both.
#:
#: BOTH ARE NEEDED, and the second is the one that is easy to miss: the footer
#: is APPENDED and `_truncate` keeps the HEAD, so a window whose numbered lines
#: run past MAX_TOOL_CHARS loses the footer entirely. That is exactly the large
#: file where a partial read matters, and checking the footer alone would have
#: called it complete. A clipped result is not a complete read either way.
_PARTIAL_READ = re.compile(r"\[showing lines (\d+)-(\d+) of (\d+)\.")
_CLIPPED_RESULT = re.compile(r"\[\.\.\. truncated: \d+ more chars\.")

#: How `_call_tool` spells a failure. A tool result is a string either way, so
#: anything reading results — `_record_opened`, for one — has to be able to
#: tell the two apart without re-deriving the wording.
TOOL_ERROR_PREFIX = "error:"


def _call_tool(ws, name, raw_args):
    """Run one tool call. Never raises — a bad call is a RESULT.

    A model that asks for a path outside the checkout, or sends malformed JSON
    arguments, must be told so and allowed to try again. Raising instead kills a
    review over a recoverable mistake, and the mistakes are common: an absolute
    path from a diff header is the obvious one.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"error: no such tool {name!r}. Available: {', '.join(_DISPATCH)}"
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except (ValueError, TypeError) as e:
        return f"error: could not parse arguments ({e}). Send a JSON object."
    try:
        return fn(ws, **args)
    except TypeError as e:
        return f"error: wrong arguments for {name}: {e}"
    except (ValueError, OSError) as e:
        return f"error: {type(e).__name__}: {e}"


#: What an assistant message may carry back into the next request.
#:
#: VERBATIM ECHO BROKE THE FAILOVER. infra#155, 2026-09-02: turn 1 timed out on
#: Fireworks and failed over to OpenRouter, whose reply carried `refusal: null`
#: and `reasoning: null`. Appended as-is, turn 2 went back to Fireworks with
#: those keys in `messages[2]` and got `400 Extra inputs are not permitted`.
#: The review died on the one path that exists to save it.
#:
#: So the echo keeps exactly what the conversation needs — the role, the text,
#: the tool calls it must answer, and `reasoning_content`, which is the chain of
#: thought when reasoning is on — and drops provider decoration and nulls. Each
#: provider accepts the keys IT returns; neither accepts the other's.
_ECHO_KEYS = ("role", "content", "tool_calls", "reasoning_content", "name")


def _for_echo(reply, drop_tool_calls=False):
    """The assistant message as it may be resent, to either provider."""
    out = {k: v for k, v in reply.items()
           if k in _ECHO_KEYS and v is not None
           and not (drop_tool_calls and k == "tool_calls")}
    out.setdefault("role", "assistant")
    if "content" not in out:
        out["content"] = ""
    return out


def _arg_summary(raw_args, width=120):
    """The arguments as one short line for the turn log."""
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        text = ", ".join(f"{k}={v!r}" for k, v in args.items())
    except (ValueError, TypeError):
        text = str(raw_args)
    return text[:width]


#: Room a resumed pass gets ON TOP of the conversation it inherits.
#:
#: The review pass's conversation is, by construction, AT the transcript budget
#: whenever the review was forced by it. So a resume that shares the same
#: absolute cap is forced on turn 1 with zero tool calls — measured on
#: caeli-marketing#212 at 240k: "turn 1: final answer forced (transcript
#: budget), 0 tool call(s)", and a revision that could neither check a drop nor
#: add much. The old separate pass never had this problem because it started
#: from empty. This is the allowance that gives the resumed pass its own room.
RESUME_HEADROOM = int(os.environ.get("REVIEW_RESUME_HEADROOM", 150_000))


def resume(messages, question, root, deadline=300, max_turns=8, **kw):
    """Ask an existing conversation one more thing — WITH THE TOOLS STILL ON.

    Tingyi's correction, 2026-09-02, to a simplification of mine that went too
    far. I had moved the revision pass onto `ask_again` (tools off, one call) to
    stop it re-exploring, and in doing so removed its ability to CHECK anything.
    That guts the most valuable action it has: "this is already handled in
    crud.py" and "nothing under tests/ names this route" are both lookups, and a
    model that cannot look can only drop findings it can disprove from memory.

    Resuming the conversation rather than starting one keeps both halves. The
    code the review pass read is still in context, so most revisions need no
    tool at all; when one does, it is there. And the prefix is unchanged, so the
    whole conversation is still a cache hit — this pass is nearly free in prompt
    tokens despite carrying the entire review with it.

    Deliberately small budgets. This is a revision, not a second review: eight
    turns and five minutes is room to check a handful of claims, not room to
    start again.
    """
    inherited = sum(len(m.get("content") or "") for m in messages)
    return run(None, None, root, deadline=deadline, max_turns=max_turns,
               _messages=list(messages) + [{"role": "user", "content": question}],
               _transcript_cap=inherited + RESUME_HEADROOM, **kw)


def run(system, user, root, model=None, deadline=900, max_turns=MAX_TURNS,
        max_tokens=MAX_TOKENS, log=print, on_turn=None, stats=None,
        _messages=None, answer_schema=None, _transcript_cap=None):
    """Run the loop against a checkout and return the model's final text.

    `deadline` is WALL CLOCK for the whole loop, checked before every request and
    again before every tool call, so the caller's timeout is a property of this
    function rather than an outer `kill` that leaves no trace of what died.

    `on_turn(turn)` is called before every request and may raise to abort — the
    PR reviewer uses it to notice that the PR merged mid-review. It runs BETWEEN
    turns rather than on a timer thread, which is the whole advantage of owning
    the loop: the old subprocess had to be polled and killed.

    `answer_schema`, if given, is a `response_format` applied ONLY on the turn
    where tools are off and an answer is required. It cannot be sent on an
    exploring turn: alongside a live tool choice the model echoed the tool
    schema back (`{"type": "object"}`) instead of answering. On the answer turn
    it is the difference between a judgement and a parse error.

    `stats`, if given, is filled in with `turns` and `tool_calls` — how much
    looking actually happened. A caller deciding whether to APPROVE needs that:
    a reply claiming to have checked nine files is only worth something if nine
    files were opened, and the model is not the authority on whether they were.

    Returns (text, transcript). Raises `Timeout` or `AgentError`, both carrying
    the transcript.
    """
    stats = {} if stats is None else stats
    stats.update(turns=0, tool_calls=0)
    # The conversation is left on `stats` so a caller can ASK AGAIN without
    # paying for the exploration a second time. See `ask_again`.
    stats["messages"] = None
    ws = Workspace(root)
    # `_messages` resumes an existing conversation (see `resume`). Appending to
    # it rather than rebuilding keeps the cached prefix intact.
    messages = (list(_messages) if _messages else
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}])
    transcript = []
    tool_calls_made = [0]
    started = time.monotonic()
    shortened = force_next = False
    # RECOVERY TURNS ARE NOT EXPLORATION TURNS. A truncated answer and a refused
    # `tool_choice: "none"` are both retried, and both used to consume a turn
    # from the same budget — so on a loop already at its cap the retry fell out
    # of the bottom and raised "loop ended without an answer", which is a worse
    # report than either failure it was recovering from. They get their own
    # small allowance instead, which cannot be starved and cannot be looped on.
    extra = 0
    MAX_EXTRA = 2
    kwargs = {"max_tokens": max_tokens, "timeout": REQUEST_TIMEOUT}
    if model:
        kwargs["model"] = model

    def left():
        return deadline - (time.monotonic() - started)

    def note(line):
        transcript.append(line)
        if log:
            log(f"  [agent] {line}")

    turn = 0
    while turn < max_turns + extra:
        turn += 1
        stats["turns"] = turn
        # Two reasons to stop asking for tools rather than to fail: the clock and
        # the transcript. Both give the model one last turn with `tool_choice`
        # off, because a review that has read six files and is out of budget
        # still has something worth saying — throwing it away is strictly worse
        # than a slightly under-researched answer.
        if on_turn is not None:
            on_turn(turn)
        size = sum(len(m.get("content") or "") for m in messages)
        out_of_room = size > (_transcript_cap or MAX_TRANSCRIPT_CHARS)
        last_turn = turn == max_turns + extra
        if left() <= 0:
            raise Timeout(
                f"deadline of {deadline:.0f}s exhausted after {turn - 1} turn(s)",
                transcript)
        forced = force_next or out_of_room or last_turn or left() < REQUEST_TIMEOUT
        if forced and not force_next:
            why = ("transcript budget" if out_of_room
                   else "turn cap" if last_turn else "deadline")
            # AN AGENT THAT NEVER LOOKED HAS NOT REVIEWED. Being forced on the
            # FIRST turn by the clock means the deadline was already spent
            # before this run started, so the model answers from the prompt
            # alone — which is precisely the one-shot reviewer this replaced,
            # except now wearing an agent's authority.
            #
            # Measured on slack-app#375: forced on turn 1, answered in 1.3s with
            # `{"findings":[]}`, and the confirmation pass — also forced on turn
            # 1 — reported "confirmed clean after examining 9 file(s)". It had
            # examined nothing. A fabricated evidence list is worse than no
            # review, because it satisfies the gate built to catch exactly this.
            #
            # The turn CAP is different and stays allowed: `max_turns=1` is a
            # deliberate configuration, not a budget that ran out.
            if turn == 1 and not last_turn and not out_of_room:
                raise Timeout(
                    f"only {left():.0f}s left at the start — not enough to read "
                    f"anything, so there is nothing to review with", transcript)
            note(f"turn {turn}: final answer forced ({why})")
            messages.append({"role": "user", "content": (
                "Stop reading and answer now with the required JSON only. "
                f"Reason: {why} reached. Report what you have actually verified; "
                "do not invent findings to fill the gap.")})

        t0 = time.monotonic()
        try:
            reply = llm.chat_with_tools(
                messages, TOOLS, tool_choice="none" if forced else "auto",
                response_format=answer_schema if forced else None, **kwargs)
        except ReviewError as e:
            # A TRUNCATED FINAL ANSWER IS RECOVERABLE, and it is the one failure
            # here that throws away work rather than reporting it: measured on
            # slack-app#377, the model explored for 90 seconds across ten turns
            # and then wrote past the budget, so a complete review became an
            # alert. Asking once for a shorter answer costs one call and keeps
            # the exploration.
            #
            # ONCE. A second truncation means the budget is genuinely too small
            # for this PR, which is an operator problem and must be reported as
            # one rather than looped on.
            if ("finish_reason=length" in str(e) and not shortened
                    and extra < MAX_EXTRA):
                shortened = force_next = True
                extra += 1
                note(f"turn {turn}: answer truncated at max_tokens — "
                     f"asking once for a shorter one")
                messages.append({"role": "user", "content": (
                    "Your answer was cut off at the token limit. Send it again, "
                    "shorter: keep only the findings you are most confident in, "
                    "and where a `fix` was long, replace it with the one or two "
                    "lines that change and leave `fix_verified` false. Do not "
                    "drop a finding's `detail` — a title with no explanation is "
                    "not usable. JSON only.")})
                continue
            raise AgentError(f"turn {turn}: {e}", transcript) from e
        force_next = False
        latency = time.monotonic() - t0
        calls = reply.get("tool_calls") or []
        content = (reply.get("content") or "").strip()

        if not calls:
            note(f"turn {turn}: answered in {latency:.1f}s "
                 f"({len(content)} chars, {size} char transcript)")
            # HOW MUCH LOOKING HAPPENED is part of the answer, not a detail.
            # A caller deciding whether to APPROVE has to be able to tell a
            # review that read nine files from one that read none and said it
            # read nine.
            note(f"turn {turn}: {tool_calls_made[0]} tool call(s) total")
            # THE ANSWER TURN IS USUALLY NOT THE FORCED ONE. A model that is
            # finished simply stops calling tools, so a `response_format`
            # attached only to the forced turn would almost never fire. And it
            # cannot be attached to every turn: alongside a live tool choice the
            # model has to choose between calling a tool and matching the shape,
            # and it echoed the schema back instead of answering.
            #
            # So when a schema is required and the model has just answered
            # freely, its answer is re-asked ONCE with tools off and the schema
            # on. The conversation already holds its reasoning, the prefix is
            # unchanged so the call is a cache hit, and what comes back is the
            # same judgement in a shape that cannot fail to parse — which is the
            # whole failure this exists to end: on caeli-marketing#212 the
            # revision reasoned correctly for 7,410 characters and then finished
            # with a plain-text list, and the judgement was thrown away.
            if answer_schema and not forced:
                note(f"turn {turn}: re-asking the answer against the schema")
                try:
                    shaped = llm.chat_with_tools(
                        messages + [_for_echo(reply)], TOOLS, tool_choice="none",
                        response_format=answer_schema, **kwargs)
                except ReviewError as e:
                    note(f"turn {turn}: schema pass unavailable ({e}); "
                         f"keeping the free-form answer")
                else:
                    shaped_content = (shaped.get("content") or "").strip()
                    if shaped_content:
                        stats["messages"] = messages + [_for_echo(reply)]
                        return shaped_content, transcript
            stats["messages"] = messages + [_for_echo(reply)]
            if not content:
                # An empty non-tool reply is the one case with nothing to
                # salvage and nothing to retry against. Name it precisely: the
                # old reviewer's `{"findings":[]}` shrug taught us that a
                # content-free answer must never be read as a clean review.
                raise AgentError(
                    f"turn {turn}: model returned neither content nor tool calls",
                    transcript)
            return content, transcript

        if forced:
            # WE ASKED FOR AN ANSWER AND GOT A TOOL CALL. `tool_choice: "none"`
            # is a request, not a guarantee — providers differ, and this one
            # ignored it once already. Honouring the call would restart the
            # exploration the force exists to end.
            #
            # The assistant message is DROPPED rather than appended: `role:tool`
            # replies must attach to a call, so keeping it without executing the
            # call would leave the conversation malformed and 400 on the next
            # request. Never appended means never inconsistent.
            if content:
                note(f"turn {turn}: answered alongside {len(calls)} ignored "
                     f"tool call(s) in {latency:.1f}s")
                # REBUILT WITHOUT `tool_calls`, and only that. Keeping the
                # refused calls would leave the conversation malformed (a
                # `role:tool` reply must attach to a call), but rebuilding it
                # from `content` alone silently drops everything else the model
                # sent — `reasoning_content` above all, which is where the whole
                # chain of thought lives when reasoning is enabled. This is the
                # message list the revision pass RESUMES from, so a drop here
                # makes it start from a conversation whose last turn has been
                # quietly emptied of its thinking. Invisible from the outside;
                # it shows up only as a slower, slightly worse review.
                stats["messages"] = messages + [
                    _for_echo(reply, drop_tool_calls=True)]
                return content, transcript
            if extra >= MAX_EXTRA:
                raise AgentError(
                    f"turn {turn}: kept calling tools after being told not to, "
                    f"and never answered", transcript)
            note(f"turn {turn}: asked for tools after being told not to; "
                 f"asking again")
            force_next = True
            extra += 1
            messages.append({"role": "user", "content": (
                "You may not call tools now. Reply with the required JSON and "
                "nothing else, using only what you have already read.")})
            continue

        note(f"turn {turn}: {len(calls)} tool call(s) in {latency:.1f}s")
        # The assistant message goes back with its tool_calls — or the
        # follow-up `role:tool` messages have no call to attach to and the
        # provider 400s — and with its reasoning, but WITHOUT the other
        # provider's decoration. See `_for_echo`.
        messages.append(_for_echo(reply))
        for call in calls:
            fn = (call.get("function") or {})
            name = fn.get("name", "?")
            raw = fn.get("arguments")
            if left() <= 0:
                raise Timeout(
                    f"deadline of {deadline:.0f}s exhausted during turn {turn}",
                    transcript)
            t1 = time.monotonic()
            tool_calls_made[0] += 1
            stats["tool_calls"] = tool_calls_made[0]
            # WHICH FILES IT ACTUALLY OPENED. A partial review that names every
            # unshown file as unreviewed is wrong about the ones the agent went
            # and read; only the untouched ones deserve the caveat.
            result = _call_tool(ws, name, raw)
            _record_opened(stats, name, raw, result)
            note(f"  {name}({_arg_summary(raw)}) -> {len(result)} chars "
                 f"in {time.monotonic() - t1:.2f}s")
            messages.append({"role": "tool",
                             "tool_call_id": call.get("id", ""),
                             "content": result})

    # The `last_turn` branch forces an answer on the final turn, and that turn
    # returns or raises — so this is reachable only if a future edit breaks that
    # invariant. Kept, and named, so the break is visible rather than silent.
    raise AgentError(f"loop ended without an answer after {turn} turns",
                     transcript)


def ask_again(messages, question, model=None, max_tokens=MAX_TOKENS,
              timeout=REQUEST_TIMEOUT, log=print):
    """One more question against a conversation that already happened. NO TOOLS.

    Tingyi's suggestion, and it is aimed at the right step. Across three runs of
    slack-app#381 at a fixed commit, run 1 reported the paired-PR wire mismatch
    and not the untested forwarding, run 2 reported the untested forwarding and
    not the wire mismatch. Same code, same prompt. The model was not failing to
    FIND things — it had read the files in both runs — it was failing to REPORT
    all of what it found.

    So the cheap fix is not a second exploration. It is one more question to the
    conversation that already has the files in it, with tools off: a single
    call, no traversal, and the model is answering about evidence it is still
    looking at.

    Returns "" on any failure. A completeness pass that can cost the review it
    is completing is worth less than no completeness pass.
    """
    if not messages:
        return ""
    kwargs = {"max_tokens": max_tokens, "timeout": timeout}
    if model:
        kwargs["model"] = model
    try:
        reply = llm.chat_with_tools(
            list(messages) + [{"role": "user", "content": question}],
            TOOLS, tool_choice="none", **kwargs)
    except ReviewError as e:
        if log:
            log(f"  [agent] second look unavailable: {e}")
        return ""
    return (reply.get("content") or "").strip()

"""Chat completions against an OpenAI-shaped endpoint, with provider failover.

Extracted from Caeli's `cronlib` so this package stands on its own. Everything
here is deliberately small and dependency-free: `urllib`, no SDK. A reviewer
that cannot be read in an afternoon is a reviewer nobody will trust with a
blocking check.

TWO ENTRY POINTS, and the split is not cosmetic:

  `chat`            — the answer is `content`, and `finish_reason == "length"`
                      is FATAL (a truncated JSON body reaches the parser as a
                      meaningless offset error).
  `chat_with_tools` — the answer is often `tool_calls` with EMPTY content, and
                      `finish_reason == "tool_calls"` is the healthy case.

Folding the second into the first as a flag would make one guard mean two
things, which is how the failures this package was written to fix happened.
"""
import http.client
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.request

from . import env
from .errors import ReviewError

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = os.environ.get(
    "REVIEW_MODEL", "accounts/fireworks/models/deepseek-v4-flash-0731")

#: The same model on a second provider, used ONLY when the first one is the
#: broken party. Fireworks served 503 "service overloaded" on 13 of 15 probes on
#: 2026-08-11; three retries two seconds apart against a sustained overload is
#: not a recovery strategy, it is the same failure three times.
FAILOVER_MODEL_MAP = {
    "accounts/fireworks/models/deepseek-v4-flash-0731": "deepseek/deepseek-v4-flash-0731",
    "accounts/fireworks/models/minimax-m3": "minimax/minimax-m3",
    "accounts/fireworks/models/glm-5p2": "z-ai/glm-5.2",
}

#: LOAD-BEARING, and the reason a model swap is not a one-line change.
#:
#: deepseek-v4-flash is a reasoning model whose thinking shares the `max_tokens`
#: budget, so the budget is a CLIFF and not a cap. Measured on a real ~11.5k
#: prompt:
#:
#:   effort    wall    completion   finish    content
#:   unset     126s    24000 (cap)  length    EMPTY   ← 94k chars of reasoning
#:   "low"      35s     3847        stop      781 ch
#:   "none"      4s      618        stop      2071 ch
#:
#: Unset is not "a bit slower", it is a guaranteed failure on every call. In an
#: AGENT the deliberation belongs in the loop anyway: each tool result is
#: grounded evidence, which is worth more than unbounded thinking about a diff
#: the model has not looked past.
DEFAULT_REASONING_EFFORT = "none"


#: What this review has spent, accumulated across every call in the process.
#:
#: TOKENS ALWAYS, MONEY ONLY IF SOMEBODY CONFIGURED A PRICE. A cost printed from
#: a guessed rate is worse than no cost: it gets quoted, it gets budgeted
#: against, and nothing about it is true. `REVIEW_PRICE_PER_MTOK` (input,
#: cached-input, output — comma separated, USD per million) turns the estimate
#: on and names where the numbers came from.
USAGE = {"calls": 0, "prompt": 0, "cached": 0, "completion": 0, "by_provider": {}}


def _price():
    raw = os.environ.get("REVIEW_PRICE_PER_MTOK", "")
    try:
        parts = [float(x) for x in raw.split(",")]
    except ValueError:
        return None
    if len(parts) == 2:
        # input, output — cached input assumed at Fireworks' documented 50%
        # DEFAULT. Pass all three when you know the real rate: on
        # deepseek-v4-flash serverless it is $0.007 against $0.22, a 97%
        # discount, and the two-value form would overstate the bill fivefold.
        parts = [parts[0], parts[0] / 2, parts[1]]
    return parts if len(parts) == 3 else None


def reset_usage():
    global _PREFER_FAILOVER
    USAGE.update(calls=0, prompt=0, cached=0, completion=0, by_provider={})
    # A new review starts on the cheap provider again.
    _PREFER_FAILOVER = False


def _record(provider, model, body, headers):
    """Tally one call. Never raises — accounting must not cost a review."""
    try:
        u = body.get("usage") or {}
        prompt = int(u.get("prompt_tokens") or 0)
        completion = int(u.get("completion_tokens") or 0)
        # FIREWORKS REPORTS CACHE HITS IN HEADERS, not in `usage`. Its caching is
        # automatic and prefix-matched, and cached prompt tokens are discounted
        # (50% by default on serverless) — so on an agent loop, where every turn
        # re-sends the whole conversation, this is most of the bill.
        cached = 0
        for key in ("fireworks-cached-prompt-tokens", "x-cached-prompt-tokens"):
            if headers and headers.get(key):
                cached = int(headers.get(key))
                break
        # OpenAI-shaped providers put it here instead.
        if not cached:
            cached = int(((u.get("prompt_tokens_details") or {})
                          .get("cached_tokens")) or 0)
        USAGE["calls"] += 1
        USAGE["prompt"] += prompt
        USAGE["cached"] += min(cached, prompt)
        USAGE["completion"] += completion
        per = USAGE["by_provider"].setdefault(f"{provider}:{model_label(model)}",
                                              {"calls": 0, "prompt": 0,
                                               "cached": 0, "completion": 0})
        per["calls"] += 1
        per["prompt"] += prompt
        per["cached"] += min(cached, prompt)
        per["completion"] += completion
    except Exception:  # noqa: BLE001 — accounting is never worth a review
        pass


def usage_line():
    """One line for the job log. Names the cache rate, because that is the
    number a person can act on — a low rate means the prefix is being
    disturbed, which is a bug, not a bill."""
    u = USAGE
    if not u["calls"]:
        return "no model calls"
    fresh = u["prompt"] - u["cached"]
    rate = (u["cached"] / u["prompt"] * 100) if u["prompt"] else 0.0
    line = (f"{u['calls']} call(s), {u['prompt']:,} prompt tokens "
            f"({u['cached']:,} cached, {rate:.0f}%), "
            f"{u['completion']:,} completion")
    price = _price()
    if price:
        cost = (fresh * price[0] + u["cached"] * price[1]
                + u["completion"] * price[2]) / 1_000_000
        saved = u["cached"] * (price[0] - price[1]) / 1_000_000
        line += f" — ${cost:.4f} (caching saved ${saved:.4f})"
    return line


class _Retryable(Exception):
    """A provider-side failure worth trying the other provider for."""


#: A CDN refusing us, wearing a 4xx. Cloudflare's own error codes appear in the
#: body as "error code: NNNN"; 1010 is a browser-signature ban and 1020 an
#: access-rule denial. Neither is anything about the request we sent.
_EDGE_BLOCK = re.compile(r"error code:\s*1\d{3}\b|cloudflare|attention required",
                         re.I)


def _is_edge_block(code, body):
    return code in (403, 401) and bool(_EDGE_BLOCK.search(body or ""))


def model_label(model=None):
    """The bare model name, for report text that must not go stale.

    A function rather than an f-string at the call site: this runs on Python
    3.11 in places, where an f-string may not reuse the outer quote character.
    """
    return (model or DEFAULT_MODEL).split("/")[-1]


def _reasoning_payload(effort, failover=False):
    """The providers disagree on the wire format. Fireworks takes a flat
    `reasoning_effort` string; OpenRouter takes a `reasoning` object and spells
    "off" as `{"enabled": false}` rather than an effort value."""
    if effort is None:
        return {}
    if failover:
        return {"reasoning": {"enabled": False} if effort == "none"
                else {"effort": effort}}
    return {"reasoning_effort": effort}


def _post(url, key, payload, timeout, provider, model):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 method="POST")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            headers = {k.lower(): v for k, v in r.headers.items()}
            body = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:300]
        # 5xx and 429 are the provider. A 4xx is normally OUR payload and would
        # fail identically on the second provider, so failing over just spends
        # twice.
        #
        # THE EXCEPTION IS AN EDGE BLOCK. Measured on slack-app#378,
        # 2026-09-02: turn 10 came back `403: error code: 1010` — a Cloudflare
        # signature ban, from the CDN in front of the API, after nine turns of
        # the same credentials and the same payload shape had succeeded. It is
        # not our request; it is the edge refusing us, and the other provider is
        # behind a different edge. Treating it as a payload error threw away a
        # review that was nine turns in.
        #
        # Matched on the body, not the status: a genuine 403 from the API
        # (revoked key, model not enabled) must still fail loudly rather than
        # spend a second provider's budget discovering the same thing.
        if e.code >= 500 or e.code == 429 or _is_edge_block(e.code, detail):
            raise _Retryable(f"{provider} {model} -> {e.code}: {detail}")
        raise ReviewError(f"{provider} {model} -> {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise _Retryable(f"{provider} {model} unreachable: {e}")
    except (TimeoutError, socket.timeout) as e:
        raise _Retryable(f"{provider} {model} timed out after {timeout}s: {e}")
    except (http.client.IncompleteRead, ConnectionError) as e:
        # THE CONNECTION DIED MID-BODY. `URLError` only wraps failures at
        # connect time; a response cut off while being read arrives as
        # `IncompleteRead` (or a bare `ConnectionResetError`), and neither is
        # a subclass of it. agentic-review#2, 2026-09-03 06:01: a router
        # restart cut a reply at 341 bytes and the review was announced as
        # "crashed: IncompleteRead" — a transient wearing the crash label the
        # other clauses exist to remove.
        raise _Retryable(f"{provider} {model} connection dropped mid-response: "
                         f"{type(e).__name__}: {e}")
    _record(provider, model, body, headers)
    choices = body.get("choices") or []
    if not choices:
        raise _Retryable(f"{provider} returned no choices: {json.dumps(body)[:200]}")
    return choices[0]


#: STICKY FAILOVER. Once Fireworks has failed in this process, later calls go
#: to OpenRouter FIRST and try Fireworks only as the fallback.
#:
#: Measured live, 2026-09-02 22:00-22:20 UTC, two reviews at once: Fireworks'
#: `/models` answered in 0.27s while every chat completion hit the 180s
#: timeout, and the failover was per call — so EVERY turn paid 180s waiting on
#: Fireworks before OpenRouter answered in ~40s. Turns cost ~3.5 minutes,
#: infra#156's review pass took 865s, and both runs were heading for the
#: 25-minute job ceiling with a BROKEN page at the end of it. A provider that
#: has just stalled is the one least likely to answer the next call in time.
#:
#: Process-scoped, deliberately: one review is one process, and the next review
#: starts fresh with Fireworks — it is the cheaper provider and usually fine.
_PREFER_FAILOVER = False


def _with_failover(payload, timeout, reasoning_effort, extract):
    global _PREFER_FAILOVER
    key = env.get("FIREWORKS_API_KEY")
    if not key:
        raise ReviewError("FIREWORKS_API_KEY not set")
    model = payload["model"]
    alt_key = env.get("OPENROUTER_API_KEY")
    alt_model = FAILOVER_MODEL_MAP.get(model)

    def fireworks():
        return extract(_post(FIREWORKS_URL, key, payload, timeout,
                             "fireworks", model), "fireworks", model)

    def openrouter():
        alt = dict(payload, model=alt_model)
        alt.pop("reasoning_effort", None)
        alt.update(_reasoning_payload(reasoning_effort, failover=True))
        return extract(_post(OPENROUTER_URL, alt_key, alt, timeout,
                             "openrouter", alt_model), "openrouter", alt_model)

    can_fail_over = bool(alt_key and alt_model)
    if _PREFER_FAILOVER and can_fail_over:
        try:
            return openrouter()
        except _Retryable as first:
            print(f"openrouter failed ({first}); trying fireworks",
                  file=sys.stderr)
            try:
                return fireworks()
            except _Retryable as second:
                raise ReviewError(f"both providers failed — openrouter: {first}"
                                  f" | fireworks: {second}")
    try:
        return fireworks()
    except _Retryable as first:
        if not can_fail_over:
            why = ("OPENROUTER_API_KEY not set" if not alt_key
                   else f"no failover mapping for {model}")
            raise ReviewError(f"{first} — no failover ({why})")
        print(f"fireworks failed ({first}); failing over to {alt_model} "
              f"for the rest of this run", file=sys.stderr)
        _PREFER_FAILOVER = True
        try:
            return openrouter()
        except _Retryable as second:
            raise ReviewError(f"both providers failed — fireworks: {first} | "
                              f"openrouter: {second}")


def chat(messages, model=DEFAULT_MODEL, max_tokens=8192, temperature=0.2,
         timeout=180, json_mode=False,
         reasoning_effort=DEFAULT_REASONING_EFFORT):
    """One stateless completion. Returns the assistant `content` string."""
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature}
    payload.update(_reasoning_payload(reasoning_effort))
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    def extract(choice, provider, mdl):
        if choice.get("finish_reason") == "length":
            # Not retryable: the other provider serves the same model and would
            # truncate at the same budget. Raised here so the alert names the
            # real problem rather than surfacing as a JSON parse error at some
            # meaningless character offset.
            raise ReviewError(
                f"{provider} {mdl} truncated at max_tokens={max_tokens} "
                f"(finish_reason=length) — raise the budget, lower "
                f"reasoning_effort, or shrink the input")
        return choice["message"].get("content") or ""

    return _with_failover(payload, timeout, reasoning_effort, extract)


def chat_with_tools(messages, tools, model=DEFAULT_MODEL, max_tokens=8192,
                    temperature=0.2, timeout=180,
                    reasoning_effort=DEFAULT_REASONING_EFFORT,
                    tool_choice="auto", response_format=None):
    """One turn of a tool-using conversation. Returns the whole assistant MESSAGE.

    Truncation still raises: a tool call cut off mid-arguments is not
    recoverable by retrying elsewhere.
    """
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature, "tools": tools,
               "tool_choice": tool_choice}
    payload.update(_reasoning_payload(reasoning_effort))
    if response_format:
        payload["response_format"] = response_format
    # THE SCHEMAS GO WITH `tool_choice: "none"` TOO, and dropping them to save
    # tokens is what broke it. Without `tools` in the payload the `tool_choice`
    # field is meaningless and was dropped alongside it, so a request meant to
    # forbid tool calls became an ordinary chat request — and deepseek, with a
    # history full of tool calls, simply made more. Measured on slack-app#381:
    # the "answer now" turn called `read_file` and the loop carried on for four
    # more turns before running out of budget.

    def extract(choice, provider, mdl):
        if choice.get("finish_reason") == "length":
            raise ReviewError(
                f"{provider} {mdl} truncated at max_tokens={max_tokens} "
                f"(finish_reason=length)")
        return choice["message"]

    return _with_failover(payload, timeout, reasoning_effort, extract)


def parse_json_reply(text):
    """Extract a JSON object from a reply, tolerating fences and prose.

    Tries the most LITERAL reading first and only then salvage heuristics — a
    reply that is already valid JSON must never be mangled by one. That ordering
    is the fix for a real failure: the old version stripped a fenced block
    whenever ``` appeared anywhere, so a payload whose string values legitimately
    contained ```bash blocks was replaced by the fragment between the first two
    fences and reported as "no JSON object", while the error printed the
    original, perfectly valid JSON.
    """
    t = (text or "").strip()
    candidates = [t]

    def add_brace_slice(s):
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j > i:
            candidates.append(s[i:j + 1])

    add_brace_slice(t)
    if t.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
        if m:
            inner = m.group(1).strip()
            candidates.append(inner)
            add_brace_slice(inner)

    last = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            last = e
    raise ReviewError(
        f"unparseable JSON in reply ({last}); reply was {len(text or '')} chars, "
        f"head={(text or '')[:120]!r} tail={(text or '')[-120:]!r}")

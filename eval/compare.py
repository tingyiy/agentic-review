"""Run the reviewer against pull requests that already have another review, and
report what each side found that the other did not.

THIS IS THE ONLY HONEST WAY TO JUDGE A CHANGE TO THE REVIEWER. A prompt edit
always reads better than the prompt it replaces; the question is whether the
findings changed, and in which direction. So a reviewer change is accepted or
rejected on this output, never on how the prompt sounds.

    python -m eval.compare --repo slack-app --prs 381 378 377 376

It reviews at the PR's ORIGINAL HEAD, so the comparison is against what the
other reviewer actually saw. It never posts: `main()` is bypassed entirely and
nothing here holds a write path.

A merged PR is the right subject. The other reviews are already in, the outcome
is known, and — most usefully — the review comments that were ACTED ON are
distinguishable from the ones that were argued down.
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_review import checks, context as ctx, llm, review, tracker  # noqa: E402
from agentic_review.config import ORG  # noqa: E402

#: Who else's reviews we compare against. The bot below is our own, so its
#: reviews are the INCUMBENT — the thing being replaced — and Copilot is the
#: independent baseline.
#: EACH REVIEWER IS A SET OF LOGINS, and that is not defensiveness — Copilot
#: genuinely posts under two. Its review BODIES come from
#: `copilot-pull-request-reviewer[bot]`; its inline comments, which is where its
#: actual findings live, come from `Copilot`. Matching one spelling found the
#: overview and none of the findings, so on slack-app#378 the baseline scored
#: three summary bullets and zero of the two real points it made — one of which
#: is the CLAUDE.md size problem we now catch deterministically.
#:
#: The `[bot]` suffix is a third spelling of the same thing: REST returns it,
#: GraphQL (`gh pr view --json reviews`) strips it.
COPILOT = ("copilot-pull-request-reviewer", "copilot")
INCUMBENT = tuple(x for x in os.environ.get("REVIEW_BOT_LOGIN", "").split(",") if x)


def _is(login, logins):
    return (login or "").lower().removesuffix("[bot]") in logins


def baseline_commit(repo, pr, logins=COPILOT):
    """The commit the BASELINE reviewer actually read.

    THE MEASUREMENT DEPENDS ON THIS AND IT WAS WRONG UNTIL 2026-09-02. Copilot
    reviews early — usually the opening commit — the author then fixes what it
    found, and this harness was reviewing the FINAL head. So findings Copilot
    made against code that no longer existed were being scored as recall
    failures on our side. Measured across three PRs, it was wrong on all three:

        slack-app#380  head b38342c19bb1  copilot read fbb7d98e3cc1
        slack-app#381  head 4cbdcd624581  copilot read 7ab545c1d74d
        slack-app#378  head 0850df8f0c79  copilot read 040170f39d37

    On #380 that scored "the new route has no handler-level coverage" against
    us while the route had four handler tests at the commit we read.

    Returns None when the reviewer left no review, in which case the caller
    falls back to head — comparing against nothing is still a valid run.
    """
    try:
        for r in json.loads(review.gh(
                f"/repos/{ORG}/{repo}/pulls/{pr}/reviews?per_page=100")):
            if _is((r.get("user") or {}).get("login"), logins):
                return r.get("commit_id")
    except Exception as e:  # noqa: BLE001
        print(f"  ! could not read the baseline commit: {type(e).__name__}: {e}")
    return None


def _reviews(repo, pr, logins):
    """Every review body and inline comment left by one reviewer."""
    out = []
    for path, kind in (
            (f"/repos/{ORG}/{repo}/pulls/{pr}/reviews?per_page=100", "review"),
            (f"/repos/{ORG}/{repo}/pulls/{pr}/comments?per_page=100", "inline")):
        try:
            for c in json.loads(review.gh(path)):
                if not _is((c.get("user") or {}).get("login"), logins):
                    continue
                body = (c.get("body") or "").strip()
                if body:
                    out.append({"kind": kind, "state": c.get("state", ""),
                                "path": c.get("path", ""), "body": body})
        except Exception as e:  # noqa: BLE001
            print(f"  ! could not read {kind}s: {type(e).__name__}: {e}")
    return out


#: A Copilot review body is prose with bolded points; ours is a bulleted list
#: with a severity icon. Neither is machine-readable, so the split is a
#: heuristic and the output SAYS SO — the numbers here are for orientation, and
#: the judgement is made by reading the two lists side by side.
_POINT = re.compile(r"^\s*(?:[-*]|\d+\.|🔴|🟡|🔵|⚠️)\s+", re.M)

#: Everything in a review body that is not a finding. Each of these produced a
#: wrong count before it was stripped.
#:
#:   <details>      Copilot's overview, file table and "review details" block —
#:                  full of list markers and table rows, which the splitter
#:                  counted as points while LOSING the finding above them.
#:   ```…```        Our own `fix` blocks. Replacement code is full of lines
#:                  starting with `-`, so a six-line fix became six findings.
#:   the overview   On a PR where Copilot puts its summary in the body rather
#:                  than in <details>, "Adds `Company.benefit_tip`…" is a
#:                  description of the change, not a review of it.
_DETAILS = re.compile(r"<details>.*?</details>", re.S | re.I)
_FENCE = re.compile(r"```.*?```", re.S)
_OVERVIEW = re.compile(
    r"#{1,3}\s*Pull request overview.*?(?=\n#{1,4}\s|\Z)"
    r"|#{1,3}\s*Reviewed changes.*?(?=\n#{1,4}\s|\Z)",
    re.S | re.I)
_BOILERPLATE = re.compile(
    r"^\s*(?:\*?Once you.{0,60}request another.*"
    r"|💡.*"
    r"|\|.*\|\s*"                       # a markdown table row
    r"|#{1,4}\s*(?:🟡|🔴|🔵)?\s*(?:Changes recommended|No major issues|"
    r"Looks good|AI review|Reviewed changes).*"
    r"|_Automated review.*)$",
    re.M | re.I)

#: NOT A REVIEW AT ALL. Copilot answers a request it cannot serve with a review
#: object that says so, and counting that as a finding would credit the baseline
#: for turning up. Measured: it said this on three of the five PRs sampled,
#: which is a fact about the comparison and has to be REPORTED, not silently
#: dropped — a reviewer that could not run is not a reviewer that found nothing.
_DECLINED = re.compile(r"unable to review|reached (?:their|the) .{0,40}quota"
                       r"|rate limit", re.I)


def _points(bodies):
    """One line per distinct point a reviewer made, and whether it ran at all.

    Returns (points, declined_reason). A heuristic — Copilot writes prose with
    bolded points, ours writes a severity-iconed list, and neither is
    machine-readable. The numbers are for orientation; the judgement is made by
    reading the two lists side by side.
    """
    out, seen, declined = [], set(), ""
    for item in bodies:
        raw = item["body"]
        if _DECLINED.search(raw) and len(raw) < 400:
            declined = raw.strip()[:160]
            continue
        body = _OVERVIEW.sub("", _DETAILS.sub("", _FENCE.sub("", raw)))
        body = _BOILERPLATE.sub("", body).strip()
        if not body:
            continue
        parts = _POINT.split(body)
        # A body with no list markers is ONE point — an inline comment usually
        # is. Splitting it into paragraphs would inflate every prose review.
        chunks = [p.strip() for p in parts[1:]] if len(parts) > 1 else [body]
        for chunk in chunks:
            first = chunk.split("\n")[0].strip()
            if len(first) <= 12:
                continue
            # A reviewer that ran six times repeats itself, and counting the
            # same point once per round says nothing about recall. Digits are
            # out of the key because a size finding's byte count moves between
            # rounds while the finding does not.
            key = re.sub(r"[\W\d_]+", "", first.lower())[:80]
            if key in seen:
                continue
            seen.add(key)
            out.append({"where": item.get("path", ""), "text": first[:200]})
    return out, declined


def run_ours(repo, pr, at=None):
    """The full pipeline minus posting. Returns findings + timings."""
    # RESET THE JOB BUDGET PER PR. In production each review is its own process,
    # so `_STARTED` is the start of that review. Here one process reviews five
    # PRs in sequence, and by the fifth the budget was spent — `_agent_timeout`
    # handed down its 30s floor, the loop was forced to answer on turn one, and
    # slack-app#375 produced a one-line review in 10.8s where the incumbent
    # found nine things. Without this the harness measures its own exhaustion
    # and calls it recall.
    review._STARTED = time.monotonic()
    llm.reset_usage()
    meta = json.loads(review.gh(f"/repos/{ORG}/{repo}/pulls/{pr}"))
    head = at or meta["head"]["sha"]
    diff, excluded, skipped = _diff_at(repo, pr, head, meta)
    truncated = bool(excluded)
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as work:
        review.checkout(repo, head, work)
        changed = review._diff_paths(diff)
        caveats = ""
        if excluded:
            caveats += ("\n[NOT INCLUDED — these changed files did not fit the "
                        "diff budget. They ARE in the checkout — read_file works on them; "
                        "open any the change depends on: "
                        + ", ".join(excluded) + "]\n")
        if skipped:
            caveats += f"[{skipped} generated/binary files omitted]\n"
        # `diff` is REQUIRED, and that is deliberate: it was optional, this
        # call omitted it, and the cross-reference section silently switched
        # itself off in the very harness that measures whether it helps.
        # Every path the PR touches: shown, cut for budget, and skipped as
        # generated. `excluded` alone leaves a changed lockfile looking
        # untouched — and this harness diverging from `main()` is exactly how
        # the cross-reference section once measured itself as absent.
        # `skipped` counts as a number and carries the paths; a stub may hand
        # over a bare int, so ask before iterating it (same guard as `main`).
        skipped_paths = list(skipped) if isinstance(skipped, list) else []
        context = review.build_context(repo, pr, meta, work, changed, diff,
                                       list(excluded or []) + skipped_paths)
        prompt = review.PROMPT.format(
            repo=repo, path=work, diff=diff, caveats=caveats, context=context,
            # NOT the real conversation. Every one of these PRs already carries
            # our own past reviews, and feeding them back would let the reviewer
            # score itself against its own answer sheet — the findings would
            # look excellent and mean nothing.
            prior="")
        findings = review.review_findings(prompt, work)
        findings, withdrawn = review._revise(findings, work, repo)
        findings += checks.run_all(work, changed, title=meta.get("title") or "",
                                   commits=review.commit_messages(repo, pr),
                                   pr_body=meta.get("body") or "")
    return {
        "repo": repo, "pr": pr, "head": head, "title": meta.get("title", ""),
        "at_baseline": bool(at) and at != meta["head"]["sha"],
        "seconds": round(time.monotonic() - started, 1),
        "usage": dict(llm.USAGE, by_provider=dict(llm.USAGE["by_provider"])),
        "context_chars": len(context),
        "findings": findings, "withdrawn": withdrawn,
    }


def _print(result):
    """One PR, three columns."""
    print(f"\n{'=' * 78}\n{result['repo']}#{result['pr']}", flush=True)
    print(f"  {result.get('title', '')[:70]}")
    u = result.get("usage") or {}
    cache = (f", {u['cached'] / u['prompt'] * 100:.0f}% cached"
             if u.get("prompt") else "")
    print(f"\n  OURS ({result.get('seconds', '?')}s, "
          f"{result.get('context_chars', '?')} chars of context, "
          f"{len(result.get('withdrawn') or [])} withdrawn, "
          f"{u.get('calls', 0)} call(s), {u.get('prompt', 0):,} prompt tok"
          f"{cache}):")
    for f in result.get("findings") or []:
        icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(
            review.normalize_severity(f.get("severity")), "⚠️")
        print(f"    {icon} {f.get('file', '?')}:{f.get('line', '?')}  "
              f"{str(f.get('title', ''))[:110]}")
    if not result.get("findings"):
        print("    (none)")

    for label, key in (("COPILOT", "copilot"), ("INCUMBENT (hermes)", "incumbent")):
        declined = result.get(f"{key}_declined")
        print(f"\n  {label} ({len(result.get(key) or [])} points, split "
              f"heuristically" + (" — DID NOT RUN" if declined else "") + "):")
        if declined:
            # Reported, never silently dropped: a reviewer that could not run is
            # not a reviewer that found nothing, and the difference decides
            # whether this PR belongs in a recall claim at all.
            print(f"    ! {declined}")
        for p in result.get(key) or []:
            print(f"    · {p['where'] + '  ' if p['where'] else ''}{p['text'][:110]}")
        if not result.get(key) and not declined:
            print("    (none)")


def others(result):
    """Fill in the two baseline columns. No model calls — re-runnable for free.

    Split out so a scoring bug can be fixed and every past run re-scored without
    paying for the reviews again. There have been four such bugs, all of them
    undercounting the baseline.
    """
    repo, pr = result["repo"], result["pr"]
    result["copilot"], result["copilot_declined"] = \
        _points(_reviews(repo, pr, COPILOT))
    result["incumbent"], result["incumbent_declined"] = \
        _points(_reviews(repo, pr, INCUMBENT))
    return result


def _diff_at(repo, pr, sha, meta):
    """The PR's diff AS OF `sha`. Against head this is just `pr_diff`.

    Otherwise it is `base...sha`, which is what the PR looked like when the
    baseline reviewer read it — the same three-dot form GitHub uses for a PR,
    so a base that moved underneath does not show up as this PR's work.
    """
    if sha == meta["head"]["sha"]:
        return review.pr_diff(repo, pr)
    base = meta["base"]["sha"]
    raw = review.gh(f"/repos/{ORG}/{repo}/compare/{base}...{sha}",
                    accept="application/vnd.github.v3.diff")
    keep, skipped = [], 0
    for i, chunk in enumerate(raw.split("\ndiff --git ")):
        blob = chunk if i == 0 else "diff --git " + chunk
        if review.SKIP.search(blob.split("\n", 1)[0]):
            skipped += 1
            continue
        keep.append(blob)
    # Same whole-file, source-first packing as `review.pr_diff`, so the
    # harness reviews what production would.
    files = []
    for blob in keep:
        m = re.search(r"^\+\+\+ b/(.+)$", blob, re.M)
        files.append((m.group(1).strip() if m else "?", blob))
    files.sort(key=lambda f: bool(review._LOW_PRIORITY.search(f[0])))
    kept, excluded, used = [], [], 0
    for path, blob in files:
        if used + len(blob) + 1 > review.MAX_DIFF and kept:
            excluded.append(path)
            continue
        kept.append(blob)
        used += len(blob) + 1
    return "\n".join(kept), excluded, skipped


def compare(repo, pr, at_baseline=True):
    print(f"\n{'=' * 78}\nreviewing {repo}#{pr}", flush=True)
    at = baseline_commit(repo, pr) if at_baseline else None
    if at:
        print(f"  reviewing at the baseline's commit {at[:12]}", flush=True)
    result = others(run_ours(repo, pr, at=at))
    _print(result)
    return result


def _summary(results):
    """Per-PR distribution, not a single number.

    A SINGLE RUN IS NOT A MEASUREMENT. The same PR produced 2, 4, 0 and 2
    findings across four rounds of this harness while the reviewer changed
    underneath it — so every difference attributed to those changes was within
    the noise, and at least one "improvement" was almost certainly variance.
    """
    ok = [r for r in results if "error" not in r]
    print(f"\n{'=' * 78}\nSUMMARY  ({len(ok)}/{len(results)} runs completed)")
    by_pr = {}
    for r in results:
        by_pr.setdefault((r["repo"], r["pr"]), []).append(r)
    print(f"{'PR':>16}  {'ours (each run)':>22} {'copilot':>8} {'hermes':>7}")
    for (repo, pr), runs in by_pr.items():
        good = [r for r in runs if "error" not in r]
        counts = ", ".join(str(len(r["findings"])) for r in good) or "-"
        errs = len(runs) - len(good)
        first = good[0] if good else {}
        cop = ("n/a" if first.get("copilot_declined")
               else str(len(first.get("copilot") or [])))
        mark = "" if all(r.get("at_baseline") for r in good) else "  (at head)"
        note = f"  +{errs} failed" if errs else ""
        print(f"{repo}#{pr:<9}  {counts:>22} {cop:>8} "
              f"{len(first.get('incumbent') or []):>7}{mark}{note}")
    rated = [r for r in ok if not r.get("copilot_declined")]
    if rated:
        print(f"\nAcross {len(rated)} run(s) where Copilot actually ran: "
              f"ours {sum(len(r['findings']) for r in rated)} finding(s) total, "
              f"Copilot {sum(len(r['copilot']) for r in rated)} counted once per "
              f"PR.")
    print("Counts are a heuristic split of prose, and one run is noise. "
          "Read the lists.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="review these PRs now")
    ap.add_argument("--prs", nargs="+", type=int)
    ap.add_argument("--rescore", help="a saved run: re-derive the baseline "
                                      "columns and reprint, without any model "
                                      "calls")
    ap.add_argument("--at-head", action="store_true",
                    help="review the PR head instead of the commit the baseline "
                         "reviewer read. Almost always wrong for a comparison: "
                         "the author fixes what the baseline found, so its "
                         "findings score against us as recall failures.")
    ap.add_argument("--n", type=int, default=1,
                    help="review each PR this many times. THE DEFAULT OF 1 IS "
                         "NOT A MEASUREMENT: the same PR produced 2, 4, 0 and 2 "
                         "findings across four single runs, so run-to-run "
                         "variance is as large as any change being tested. "
                         "Use 3 or more before believing a difference.")
    ap.add_argument("--out", default="eval/runs")
    args = ap.parse_args()

    if args.rescore:
        with open(args.rescore) as fh:
            results = json.load(fh)
        results = [others(r) if "error" not in r else r for r in results]
        for r in results:
            if "error" not in r:
                _print(r)
        with open(args.rescore, "w") as fh:
            json.dump(results, fh, indent=2)
        _summary(results)
        return

    if not (args.repo and args.prs):
        ap.error("--repo and --prs are required unless --rescore is given")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out,
                        f"{args.repo}-{time.strftime('%Y%m%d-%H%M%S')}.json")
    results = []
    for pr in args.prs:
      for run in range(1, args.n + 1):
        try:
            r = compare(args.repo, pr, at_baseline=not args.at_head)
            r["run"] = run
            results.append(r)
        except Exception as e:  # noqa: BLE001 — one bad PR must not lose the rest
            print(f"  !! {args.repo}#{pr} failed: {type(e).__name__}: {e}")
            results.append({"repo": args.repo, "pr": pr, "run": run,
                            "error": f"{type(e).__name__}: {e}"})
        # After EVERY run, not at the end: these take twenty minutes and cost
        # real money, and a crash on the last one used to lose all of it.
        with open(path, "w") as fh:
            json.dump(results, fh, indent=2)

    _summary(results)
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()

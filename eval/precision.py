"""Score a run against a labelled PR: how many findings were RIGHT, not how many
there were.

    python -m eval.precision eval/runs/<run>.json eval/labels/<repo>-<pr>.json

Every comparison before 2026-09-02 counted findings. caeli-marketing#212 is why
that stopped being acceptable: eight posted, three disproved by the author with
a reproduction, one taken. Volume said "four times the old reviewer"; the author
said "three of eight mattered". This reads the second number.

Ground truth is what the author DID — took it, disproved it, escalated it, left
it — never what a model thinks of it. Matching is by title tokens, which is a
heuristic and is printed so it can be argued with.
"""
import json
import re
import sys

STOP = {"the", "a", "an", "is", "on", "in", "of", "to", "and", "but", "at",
        "not", "for", "with", "that", "this", "when", "it", "as", "be"}


def _tokens(title):
    words = re.findall(r"[a-z0-9_]+", (title or "").lower())
    return {w for w in words if w not in STOP and len(w) > 1}


def _match(title, labels, floor=0.34):
    """The label this finding is about.

    IDENTIFIERS FIRST. A label's `keys` are the symbols that survive rewording —
    `askSubmit`, `REVIEWS`, `IntersectionObserver` — and a finding that names
    one is about that label whatever else it says. Token overlap is the
    fallback, and it is weak: the author wrote "REVIEWS array as const but
    renderer keys off verbatim" and the reviewer wrote "REVIEWS type lets name
    sit on paraphrased text", which share one token in seven.
    """
    text = title or ""
    for lab in labels:
        for key in lab.get("keys") or []:
            if key.lower() in text.lower():
                return lab, 1.0
    mine = _tokens(text)
    best, score = None, 0.0
    for lab in labels:
        theirs = _tokens(lab["title"])
        if not mine or not theirs:
            continue
        overlap = len(mine & theirs) / min(len(mine), len(theirs))
        if overlap > score:
            best, score = lab, overlap
    return (best, score) if score >= floor else (None, score)


def score_run(run, labels):
    posted = run.get("findings") or []
    dropped = [w[0] if isinstance(w, list) else w for w in (run.get("withdrawn") or [])]
    rows, seen = [], set()
    for f in posted:
        lab, s = _match(f.get("title"), labels)
        rows.append(("POSTED", f.get("title", "")[:70], lab["verdict"] if lab else "unlabelled", s))
        if lab:
            seen.add(lab["title"])
    for f in dropped:
        lab, s = _match(f.get("title"), labels)
        rows.append(("dropped", f.get("title", "")[:70], lab["verdict"] if lab else "unlabelled", s))
        if lab:
            seen.add(lab["title"])
    missed = [lab for lab in labels if lab["title"] not in seen]
    return rows, missed


def tally(rows):
    posted = [r for r in rows if r[0] == "POSTED"]
    dropped = [r for r in rows if r[0] == "dropped"]
    return {
        "posted": len(posted),
        "posted_wrong": sum(1 for r in posted if r[2] == "wrong"),
        "posted_real": sum(1 for r in posted if r[2] in ("real", "real-but-misframed")),
        "posted_unlabelled": sum(1 for r in posted if r[2] == "unlabelled"),
        "dropped_wrong": sum(1 for r in dropped if r[2] == "wrong"),
        "dropped_real": sum(1 for r in dropped if r[2] in ("real", "real-but-misframed")),
    }


def main():
    run_path, label_path = sys.argv[1], sys.argv[2]
    runs = json.load(open(run_path))
    labels = json.load(open(label_path))["findings"]
    print(f"labels: {len(labels)} — "
          + ", ".join(f"{v}={sum(1 for l in labels if l['verdict']==v)}"
                      for v in sorted({l['verdict'] for l in labels})))
    for run in runs:
        if "error" in run:
            print(f"\nrun {run.get('run')}: FAILED {run['error'][:80]}")
            continue
        rows, missed = score_run(run, labels)
        t = tally(rows)
        print(f"\nrun {run.get('run', '?')}  posted {t['posted']}: "
              f"{t['posted_real']} real, {t['posted_wrong']} WRONG, "
              f"{t['posted_unlabelled']} unlabelled  |  dropped: "
              f"{t['dropped_wrong']} wrong, {t['dropped_real']} real")
        for kind, title, verdict, s in rows:
            flag = "  <-- posted a WRONG one" if kind == "POSTED" and verdict == "wrong" else \
                   "  <-- dropped a REAL one" if kind == "dropped" and verdict in ("real", "real-but-misframed") else ""
            print(f"    {kind:8} [{verdict:18}] {s:.2f}  {title}{flag}")
        if missed:
            print("    not produced at all: "
                  + "; ".join(f"{m['title'][:40]} ({m['verdict']})" for m in missed))
    print("\nWhat good looks like: WRONG dropped, real posted, unlabelled few. "
          "Matching is token overlap; argue with a low score, not with the verdict.")


if __name__ == "__main__":
    main()

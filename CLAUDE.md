# CLAUDE.md — agentic-review

The PR reviewer, extracted on 2026-09-01 from a cron-driven script so it could
drop its one-shot agent dependency and stand on its own. Read [README.md](README.md) for
what it does. This file is the gotcha list for editing it.

**This repository is intended to go public.** Nothing Caeli-specific may be
hardcoded: org, model, ticket pattern, tracker, convention-doc names and the
alert channel are all in `config.py` with working defaults. A new coupling to
our box belongs behind an environment variable, not in a module.

## The layout, and why it is split this way

| module | what it owns |
|---|---|
| `agent.py` | the tool loop. `read_file` / `grep` / `list_files`, contained to the checkout root, turn cap, wall clock, **every turn logged** |
| `llm.py` | chat + chat_with_tools, Fireworks → OpenRouter failover, `parse_json_reply` |
| `github.py` | credentials and the raw REST call |
| `context.py` | what the reviewer is TOLD: convention docs, repo map, linked PRs |
| `tracker.py` | read-only Jira |
| `checks.py` | deterministic findings — arithmetic, not judgement |
| `review.py` | the orchestration, and every guard the PR flow has earned |
| `config.py` | everything an adopter has to set |

`review.py` is large and stays large for now. It is a sequence of guards that
each cost a real incident, and splitting it by theme before there is a second
consumer would move the risk without reducing it.

## The rules that cost something

- **Observability is the product.** A failure must name the turn it failed on.
  The whole reason hermes is gone is that `-z` logged nothing and four reviews
  died at exactly 901s with nothing to read. If you add a code path that can
  hang, it logs before it can.
- **A degraded reply must never read as a clean review.** A missing `findings`
  list raises; it does not default to `[]`. An empty answer raises. Measured:
  **7 of 12 replies were the literal `{"findings":[]}`** — a shrug that posts a
  formal APPROVE. Hence the confirmation pass, which demands `checked: [...]`.
- **Never assert on source text in a test** (`inspect.getsource(...)`). It
  passes while the behaviour changes under it and fails when the code merely
  moves. Three tests here did it and all three broke on the move; they assert
  on order of call now.
- **Prove a test fails without the fix.** Absence-assertions pass on their own.
  The containment and empty-reply guards in `test_agent.py` were mutation-tested
  against a `".." in path` check and a `if False:`.
- **Truncation is fatal at the provider layer, recoverable at the loop.** A tool
  call cut off mid-arguments cannot be salvaged. A cut-off final ANSWER can, and
  discarding ten turns of exploration to report it is the worst outcome
  available — hence exactly one retry asking for a shorter answer.
- **A merged PR is not worth reviewing and must not be alerted about.** Three
  windows: before the run starts, during the loop (checked between turns), and
  between the answer and the POST.
- **`PRClosed` exits 0, `Superseded` exits 1.** The second owed a review and did
  not produce one; the first has no PR left to be red about.
- **Deterministic checks run AFTER the agent** so the model never sees them and
  cannot be nudged into repeating or arguing with one.
- **Every truncation says so in the text the model reads.** A silent one teaches
  it that it has seen everything.

## Judging a change

`eval/compare.py` runs the reviewer against PRs that already carry a Copilot or
human review and prints what each side found that the other did not. A prompt
edit always reads better than the prompt it replaces — the question is whether
the findings moved, and in which direction.

```bash
export REVIEW_ENV_FILES="$HOME/.config/agentic-review/.env"   # keys, if not in the environment
export REVIEW_JIRA_SITE=your-site.atlassian.net                 # optional
python -m eval.compare --repo slack-app --prs 381 378 377 376
```

It reviews at each PR's **original head** and never posts — `main()` is bypassed
and nothing in `eval/` holds a write path. It also passes an EMPTY conversation:
these PRs already carry our own past reviews, and feeding them back would let
the reviewer mark its own answer sheet.

## Tests

```bash
pip install -e '.[dev]' && pytest -q
```

The suite is the specification. Nearly every test is named after a wrong answer
that was actually produced in production and the docstring says what it was —
read why a behaviour is what it is before changing it.

## Deploying it

The reusable workflow in `.github/workflows/pr-review.yml` runs on a
**self-hosted runner** that already holds the model key and the bot token, so
no reviewed repository needs a secret. Callers pin `reviewer_ref` to a commit
of this repository: bumping the reviewer is then a deliberate, reviewed change
in the caller, and a merge here does not reach anybody's runner by itself.

**A self-hosted job executes whatever a PR's workflow says, on your box.** Keep
the runner group closed to public repositories (GitHub's default), and never
attach a caller to a repository that accepts pull requests from strangers.

Run the suite before opening a PR here; `tests.yml` runs it again on every PR.

# agentic-review

[![Tests](https://github.com/tingyiy/agentic-review/actions/workflows/tests.yml/badge.svg)](https://github.com/tingyiy/agentic-review/actions/workflows/tests.yml)

A code reviewer that **runs inside a checkout of your pull request** and reads
whatever it needs, instead of being handed a diff and asked to imagine the rest.

```
review (bot)  ✓  3 findings: 1 high, 2 low
```

## Why this exists

The questions actually worth asking of a diff are all questions about code the
diff does not contain:

- does this duplicate something the repository already has?
- does it re-implement what the module beside it owns?
- does it break a pattern this codebase keeps?
- is the helper it calls shaped the way this caller assumes?
- does it do what the ticket asked for?

A single model call can only answer those from context somebody pre-selected,
which means it answers them from a guess. Measured on two real pull requests:
three one-shot runs produced nothing but hypotheticals about unseen code ("if
the parser normalizes severity differently…"), while an agent with `read_file`,
`grep` and `list_files` found four real defects — one of them in the one-shot
reviewer itself.

We evaluated [PR-Agent](https://github.com/qodo-ai/pr-agent) first and it is a
good tool. Its free tier extends diff *hunks* with surrounding lines — up to
eight, to reach the enclosing function — and does not follow definitions or
callers into untouched files; repository-wide retrieval is the paid tier. That
is the architecture we already had, so it would not have bought the one thing we
needed.

## What is different

| | pr-agent | Copilot | this |
|---|---|---|---|
| reads beyond the diff | same-file expansion only | no | **runs in the checkout** |
| refuses to approve without evidence | no | no | **a confirmation pass must return what it checked** |
| reads prior replies before re-raising a point | no | no | **four endpoints, including commit messages** |
| deterministic checks beside the model's | no | no | **yes** |
| withdraws its own stale approval | no | no | **yes** |

Two of those need explaining.

**Evidence-gated approval.** On our own prompt, **7 of 12 replies were the
literal 15-character `{"findings":[]}`** — a shrug that is indistinguishable
from a clean review, and that posts a formal APPROVE. So a first pass with no
findings does not approve: it is asked a second time to name what it actually
checked, and only a reply that lists specifics is allowed to become an approval.

**Reading the conversation first.** A reviewer that re-raises a point the author
already answered is a reviewer people stop reading. This one reads the PR's
reviews, its inline replies, its issue comments, and — the one everybody
forgets — **its commit messages**, which is where an agent working on somebody's
behalf answers a finding when it cannot comment as the repository owner.

## What it does, in order

1. Fetches the diff; drops lockfiles, images and build output.
2. Skips entirely if nothing has changed since its last review **and** nobody
   has replied — a reply is something new.
3. Checks out the PR head into a temp directory.
4. Builds the context block: the repository's own `CLAUDE.md`/`AGENTS.md`
   (root and beside every changed file), the ticket the title names, any ticket
   or pull request the description mentions, and a map of the tree.
5. Runs the agent loop, logging **every turn**: number, tool, arguments,
   latency, finish reason.
6. If it found nothing, asks it to show its work before approving.
7. Scores its own findings and drops the ones it cannot stand behind.
8. Adds the deterministic checks — ticket id in the title, a session link on
   agent-written commits, oversized convention docs.
9. Posts one review, with each finding linked to the exact blob it read.

## Install

```bash
pip install -e .
```

```bash
export FIREWORKS_API_KEY=...          # the model
export OPENROUTER_API_KEY=...         # optional, failover for the same model
export REVIEW_GITHUB_TOKEN=...        # a BOT account — see below
python -m agentic_review <repo> <pr-number>
```

`DRY=1` prints the review instead of posting it.

### The token is not a detail

A review has to be attributable to a reviewer. Posted with the workflow's own
`GITHUB_TOKEN` it appears as `github-actions`; posted with your personal token
it appears as *you reviewing your own pull request*, and afterwards it cannot be
told apart from a real review in the PR timeline. `REVIEW_GITHUB_TOKEN` should
be a bot account, and the tool refuses to post without one.

The token also needs **Commit statuses: read and write** for the
`agentic-review` status the reviewer sets on the PR head (pending while it
runs, the verdict when it posts). Without it the status is skipped with one
log line and the review still posts.

### Configuration

Everything lives in `agentic_review/config.py` and every value has a working
default:

| variable | default | what it does |
|---|---|---|
| `REVIEW_ORG` | from `GITHUB_REPOSITORY` | repository owner |
| `REVIEW_MODEL` | `deepseek-v4-flash` | any OpenAI-shaped model |
| `REVIEW_TICKET_PATTERN` | `[A-Z][A-Z0-9]+-\d+` | empty disables the title check |
| `REVIEW_JIRA_SITE` | *(unset)* | empty disables ticket context |
| `REVIEW_CONVENTION_DOCS` | `CLAUDE.md,AGENTS.md,CONTRIBUTING.md,README.md` | what counts as the rules |
| `REVIEW_AGENT_TIMEOUT` | `900` | wall clock for one pass |
| `REVIEW_ALERT_COMMAND` | *(unset)* | shell command that receives failures on stdin |
| `REVIEW_STATUS_CONTEXT` | `agentic-review` | the commit status the reviewer sets on the PR head (pending → verdict) |

### As a GitHub Action

`.github/workflows/pr-review.yml` is a reusable workflow: it fetches this
reviewer at a ref you pin and runs it on a self-hosted runner that holds the
model key and the bot token — no secrets in any reviewed repository.
`examples/pr-review-caller.yml` is the entire per-repository cost: copy it into
each repo that should get reviews and fill in the three inputs.

**GitHub-hosted instead of self-hosted.** Set `runner: '["ubuntu-latest"]'`,
hand the model key in as a repository secret, and set `post_as_actions_bot:
true` so the workflow's own token posts the review as `github-actions[bot]` —
no personal access token anywhere. GitHub withholds secrets from a fork's pull
request, so a stranger's PR runs without a key and the review step skips with
a log line. This repository reviews itself that way
(`.github/workflows/pr-review-caller.yml`).

## Reasoning is off, and that is measured

`deepseek-v4-flash` is a reasoning model whose thinking shares the `max_tokens`
budget, so the budget is a **cliff, not a cap**. On a real 30k-char diff:

```
none / 6k     6.8s     771 out  ->  6 findings
low  / 6k    41.4s    6000 out  ->  NOTHING (finish=length, billed in full)
low  / 20k   31.1s    3985 out  ->  1 finding
high / 32k  235.3s   32000 out  ->  NOTHING
max  / 32k  192.5s   32000 out  ->  NOTHING
```

Higher effort cost up to 15× and returned less, or nothing at all. In an agent
the deliberation belongs in the **loop**, where every tool result is grounded
evidence — which is worth more than unbounded thinking about a diff the model
has not looked past.

## What it produces, measured

789 findings across 117 pull requests over nine days, in twelve repositories,
scored on **what the author did next** rather than on how many were posted.

**697 of the 789 are scored below.** The other 92 say nothing either way: 82 had
no commit at all after the review, and 10 cite no file for a file-based proxy to
watch — every "PR title does not name a ticket". Counting either as ignored
would be the flattering mistake in reverse.

The proxy is *did a commit after the review touch the file the finding cites*.
It suits a workflow where a finding is answered by a push, and it is wrong in
both directions — a fix can land in another file, and a file can change for its
own reasons — so it is reported beside a control:

| | n | cited file touched afterwards |
|---|---|---|
| **model findings** (of 732 posted) | 659 | **91.0%** |
| high | 38 | 100.0% |
| medium | 243 | 93.4% |
| low | 378 | 88.6% |
| deterministic checks (of 57 posted) | 38 | 42.1% |
| **control: files the same PRs changed that no review mentioned** | 1031 | **47.2%** |

A **39.6-point gap**, about nineteen standard errors. The control is measured
from the same review's timestamp and restricted to files that existed when it
ran, so it is not "the author kept pushing".

**The deterministic checks are counted separately, and should be.** They repeat
on every review until the author fixes them, so folding them in credits or
damns the model for work it did not do — and they behave differently: 42.1%,
and 22 of the 81 findings nobody acted on are them.

**What this does not say.** Nothing about RECALL: it measures what happened to
findings that were posted, never the defects that were missed. And nothing
about correctness — "acted on" is not "accepted". Sampled findings carry author
replies of the form *"half of this is right, and I've fixed that half;
disagreeing with the other half, with evidence"*.

**How it was wrong first.** The pull request publishing these numbers was
reviewed by this tool, which found six defects in the METHOD across eight
rounds — three of them biasing the result in its author's favour. The control
read 28.3 → 35.7 → 53.5 → 39.8 → 42.7 → 39.6 points as each was fixed. One
suggested fix named a real mechanism and predicted the wrong direction, which
only running it revealed: **a suggested fix is a hypothesis.** That history is
kept with the numbers rather than tidied away, because a figure that moved six
times should be audited, not trusted.

## Evaluating a change to it

`eval/` runs the reviewer against pull requests that already have a human or
Copilot review, and reports what each side found that the other did not.
Reviewer changes are judged on that, not on how the prompt reads.

```bash
python -m eval.compare --repo slack-app --prs 378 377 376
```

## Development

```bash
pip install -e '.[dev]'
pytest -q
```

The test suite is the specification. Nearly every test is named after a wrong
answer that was actually produced in production, and the docstring says what it
was — so before changing a behaviour, read why it is that way.

MIT.

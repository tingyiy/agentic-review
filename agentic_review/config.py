"""Everything an adopter has to set, in one file.

Kept separate so the answer to "what do I have to configure" is a file listing
rather than a grep. Every value has a default that works; the ones that MUST be
set to run against your own code are `ORG` (or the `GITHUB_REPOSITORY` that
Actions provides for free) and the credentials named in `github.py` / `llm.py`.
"""
import os

#: The owner of the repositories being reviewed. GitHub Actions sets
#: `GITHUB_REPOSITORY` to `owner/name` in every job, so a workflow needs no
#: configuration at all; the environment variable is for running locally.
#: The repository owner. Actions provides GITHUB_REPOSITORY for free; anywhere
#: else, set REVIEW_ORG. Empty is refused at startup rather than guessed.
ORG = (os.environ.get("REVIEW_ORG")
       or os.environ.get("GITHUB_REPOSITORY", "").split("/")[0])

#: Bounded so a 500-file PR cannot turn one review into an afternoon. The agent
#: can still READ every file it wants — the cap is on what is pasted into the
#: prompt, not on what is reviewable.
MAX_DIFF = int(os.environ.get("REVIEW_MAX_DIFF", 60_000))

#: How many review calls one pull request may cost. The first pass takes
#: `MAX_DIFF` chars of diff and the rest go into further passes, so this is the
#: ceiling on a big PR rather than a budget for a normal one — most PRs fit in
#: one and pay nothing for this.
MAX_PASSES = int(os.environ.get("REVIEW_MAX_PASSES", 3))

#: A single file larger than this is not reviewable text, whatever its
#: extension, and gets a skeleton instead of a pass.
#:
#: `pr_diff` keeps the first file of a pass WHATEVER its size, so a cap smaller
#: than one file cannot produce an empty review — and multi-pass turned that
#: rule into a guarantee that an over-budget file gets its own pass. infra#180
#: added a 2 MB JSONL corpus: it became "pass 2 of 2: 2,084,684 chars", the
#: transcript hit its budget on turn one, the model answered in 32 characters
#: having made no tool calls, and the evidence guard correctly refused the
#: whole review. Nobody was ever going to read a truncated slab of that file.
MAX_FILE_DIFF = int(os.environ.get("REVIEW_MAX_FILE_DIFF", 2 * MAX_DIFF))

#: Seconds after which no FURTHER pass is started. The job is
#: `timeout-minutes: 25` and a pass runs 4-8 minutes, so three passes plus their
#: revisions can outlive it — and a killed job posts NOTHING, which is worse
#: than an honest partial review. The pass already running is not interrupted;
#: `AGENT_TIMEOUT` bounds that.
PASS_DEADLINE = int(os.environ.get("REVIEW_PASS_DEADLINE", 780))

#: More than this and a review stops being read. A reviewer that files twelve
#: findings has said something; one that files forty has said nothing.
MAX_FINDINGS = int(os.environ.get("REVIEW_MAX_FINDINGS", 12))

#: Wall clock for one agent pass.
AGENT_TIMEOUT = int(os.environ.get("REVIEW_AGENT_TIMEOUT", 900))

#: Wall clock for the whole job, against which each pass is sized.
JOB_BUDGET = int(os.environ.get("REVIEW_JOB_BUDGET", 25 * 60))

#: Require every PR title to name a tracker issue, e.g. `SCRUM-1234`. Empty
#: disables the check. This is deliberately a REGEX and not a boolean: the
#: shape of a ticket id is the one thing no two teams agree on.
TICKET_PATTERN = os.environ.get("REVIEW_TICKET_PATTERN", r"[A-Z][A-Z0-9]+-\d+")

#: Tracker base URL, used to fetch the description of the ticket in the title
#: and of any ticket the PR body mentions. Empty disables tracker context.
JIRA_SITE = os.environ.get("REVIEW_JIRA_SITE", "")

#: Files whose contents are handed to the reviewer verbatim, in this order,
#: when they exist at the repository root or beside a changed file.
#: The commit-status context the reviewer sets on the PR head (pending while it
#: runs, the verdict when it posts). Rename it if another tool already owns it.
STATUS_CONTEXT = os.environ.get("REVIEW_STATUS_CONTEXT", "agentic-review")

CONVENTION_DOCS = [d for d in os.environ.get(
    "REVIEW_CONVENTION_DOCS", "CLAUDE.md,AGENTS.md,CONTRIBUTING.md,README.md"
).split(",") if d.strip()]

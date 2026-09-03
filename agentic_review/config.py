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
CONVENTION_DOCS = [d for d in os.environ.get(
    "REVIEW_CONVENTION_DOCS", "CLAUDE.md,AGENTS.md,CONTRIBUTING.md,README.md"
).split(",") if d.strip()]

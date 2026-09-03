"""The failure taxonomy.

Each of these exists because the review responded differently to it, and
collapsing any two of them produced a wrong report at least once.
"""


class ReviewError(RuntimeError):
    """The review could not complete. Loud: the caller alerts and exits non-zero.

    A review that fails silently is worse than one that fails: the PR shows no
    finding and no red check, which is indistinguishable from a clean review.
    """


class AgentFailed(ReviewError):
    """The model did not produce an answer — provider error, or the clock ran out.

    Distinct from an unusable ANSWER, and the distinction is load-bearing: a
    provider outage retried as though it were a shrug spends the remaining
    budget on a run that cannot succeed, then reports "the reply was unusable",
    misattributing infrastructure as content.
    """


class PRClosed(Exception):
    """The PR merged or closed; there is nothing left to review.

    Not a failure. Exits 0 deliberately — a review posted onto a merged PR is
    noise nobody reads, and a red check on one is a complaint about nothing.
    """


class Superseded(Exception):
    """This run was cancelled in favour of a newer one; not a failure to report.

    Fires on the most ordinary path this tool has: asking for a re-review while
    one is running should supersede it, and opening a PR then requesting a
    reviewer does exactly that within seconds. The superseding run posts the
    review, so there is nothing a human is needed for.
    """

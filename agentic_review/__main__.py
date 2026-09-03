"""Entry point: `python -m agentic_review <repo> <pr-number>`.

    DRY=1 python -m agentic_review <repo> <pr>   prints the review instead

Everything this does is convert an outcome into an EXIT CODE and, when nobody
may be watching, an alert. The exit codes are the contract:

    0   a review was posted, or there was nothing left to review
    1   the review did not happen and somebody should know
"""
import sys

from . import notify
from .errors import ReviewError
from .review import _main_unless_superseded


def main():
    try:
        _main_unless_superseded()
    except SystemExit:
        raise
    except ReviewError as e:
        notify.alert(f"🚨 pr-review BROKEN: {e}")
        raise SystemExit(1)
    except Exception as e:  # noqa: BLE001 — last-resort guard
        # A crash and a handled failure are reported differently on purpose: the
        # wording is the first thing a person reads, and "crashed" sends them to
        # a stack trace while "BROKEN" sends them to the cause.
        notify.alert(f"🚨 pr-review crashed: {type(e).__name__}: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

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
from . import status
from .review import _CURRENT, _main_unless_superseded


def main():
    try:
        _main_unless_superseded()
    except SystemExit:
        raise
    except ReviewError as e:
        _failed(str(e))
        notify.alert(f"🚨 pr-review BROKEN: {e}")
        raise SystemExit(1)
    except Exception as e:  # noqa: BLE001 — last-resort guard
        # A crash and a handled failure are reported differently on purpose: the
        # wording is the first thing a person reads, and "crashed" sends them to
        # a stack trace while "BROKEN" sends them to the cause.
        _failed(f"{type(e).__name__}: {e}")
        notify.alert(f"🚨 pr-review crashed: {type(e).__name__}: {e}")
        raise SystemExit(1)


def _failed(reason):
    """The PR page must not keep saying "review in progress" after the run
    died. Best-effort, like the status itself."""
    try:
        status.failed(_CURRENT.get("repo"), _CURRENT.get("head"), reason)
    except Exception:  # noqa: BLE001 — a courtesy must never mask the cause
        pass


if __name__ == "__main__":
    main()

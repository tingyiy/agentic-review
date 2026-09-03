"""Test support.

The package imports normally — unlike the cron suite it came from, where the
scripts were hyphen-named files loaded by path. `pr_review` is kept as a fixture
name so the tests that moved over did not have to be rewritten to prove they
still pass.
"""
import os
import sys
from pathlib import Path

# The suite's repository owner. Set BEFORE the package is imported — config
# reads it at import time — and only if the environment did not choose one.
os.environ.setdefault("REVIEW_ORG", "example-org")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_script(_name="pr-review"):
    """The reviewer module. The argument is ignored — it survives from the cron
    suite's by-path loader, and keeping the signature meant the ported tests
    could be run BEFORE being edited, which is the only way to know the port
    did not change behaviour."""
    from agentic_review import review
    return review


@pytest.fixture
def pr_review():
    from agentic_review import review
    return review


@pytest.fixture(autouse=True)
def _agent_looked_by_default(monkeypatch):
    """Seed the tool-call count that the approval guard reads.

    Almost every test here stubs `run_agent`, which is the thing that records
    how much looking the loop did — so without this the guard would see zero
    calls on every stubbed run and refuse every approval, testing the guard
    instead of the behaviour under test.

    Seeded to a REAL run's shape (it looked), so tests exercise the normal path.
    A test about the guard itself sets it to zero explicitly, and
    `test_the_stats_actually_come_from_the_loop` pins the wiring so this fixture
    cannot quietly become the only thing populating it.
    """
    from agentic_review import review
    monkeypatch.setitem(review._CURRENT, "stats", {"turns": 4, "tool_calls": 6})

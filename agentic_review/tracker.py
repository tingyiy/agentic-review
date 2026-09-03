"""The issue tracker, read-only.

A PR is half of a change; the other half is the ticket that says what it was
supposed to do. A reviewer with the diff alone can check that code is correct
and cannot check that it is the RIGHT code — the most valuable finding available
to a reviewer is "this does not do what the ticket asked", and it is the one no
diff-only tool can ever make.

Optional by construction: with `REVIEW_JIRA_SITE` unset, everything here returns
nothing and the review proceeds without tracker context. A tracker outage must
never fail a review — an unreachable ticket is one less piece of evidence, not a
reason to leave a PR unreviewed.
"""
import base64
import json
import os
import re
import urllib.error
import urllib.request

from . import env
from .config import JIRA_SITE, TICKET_PATTERN

#: Ticket ids named anywhere in prose. Anchored to a word boundary so a git SHA
#: or a version string cannot masquerade as one.
TICKET_RE = re.compile(rf"\b({TICKET_PATTERN})\b") if TICKET_PATTERN else None

#: How much of one ticket the reviewer sees. A ticket is context, not the
#: subject: an epic with forty comments must not crowd out the diff.
MAX_DESCRIPTION = 3_000
MAX_COMMENT = 800
MAX_COMMENTS = 6

#: Tickets fetched per review, across the title and every mention. A PR body
#: that lists twenty tickets is a release note, and fetching all of them would
#: cost more than it explains.
MAX_TICKETS = 4


def ticket_ids(*texts):
    """Every distinct ticket id mentioned, in order of first appearance."""
    if TICKET_RE is None:
        return []
    seen, out = set(), []
    for text in texts:
        for match in TICKET_RE.findall(text or ""):
            key = match.upper()
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def _creds():
    email = env.get("REVIEW_JIRA_EMAIL") or env.get("ATLASSIAN_USERNAME")
    token = env.get("REVIEW_JIRA_TOKEN") or env.get("ATLASSIAN_API_KEY")
    if not email or not token:
        return None
    return base64.b64encode(f"{email}:{token}".encode()).decode()


def available():
    return bool(JIRA_SITE and _creds())


def _get(path):
    auth = _creds()
    req = urllib.request.Request(f"https://{JIRA_SITE}/rest/api/3{path}")
    req.add_header("Authorization", f"Basic {auth}")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _adf_text(node, out=None):
    """Atlassian Document Format to plain text.

    ADF is a nested node tree, and the description is the part of a ticket that
    actually says what to build — dropping it because the format is awkward
    would throw away the reason for reading the ticket at all. Anything
    unrecognised contributes its `text` and its children, which degrades to
    "slightly wrong whitespace" rather than to nothing.
    """
    out = [] if out is None else out
    if isinstance(node, str):
        out.append(node)
        return out
    if isinstance(node, list):
        for item in node:
            _adf_text(item, out)
        return out
    if not isinstance(node, dict):
        return out
    if node.get("type") == "hardBreak":
        out.append("\n")
    if node.get("text"):
        out.append(node["text"])
    _adf_text(node.get("content") or [], out)
    if node.get("type") in ("paragraph", "heading", "listItem", "codeBlock"):
        out.append("\n")
    return out


def _text(field):
    if isinstance(field, str):
        return field
    return "".join(_adf_text(field)).strip()


def fetch(key):
    """One ticket as plain text, or None. Never raises.

    Never raises is the whole contract: a 404 on a ticket somebody typo'd, an
    expired token, a tracker outage — none of those are reasons to abandon a
    review, and each of them would if this propagated.
    """
    if not available():
        return None
    try:
        issue = _get(f"/issue/{key}?fields=summary,description,status,issuetype,comment")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError,
            ValueError) as e:
        print(f"[review] could not read {key}: {type(e).__name__}: {e}")
        return None
    fields = issue.get("fields") or {}
    comments = ((fields.get("comment") or {}).get("comments") or [])[-MAX_COMMENTS:]
    return {
        "key": key,
        "summary": fields.get("summary") or "",
        "status": ((fields.get("status") or {}).get("name") or ""),
        "type": ((fields.get("issuetype") or {}).get("name") or ""),
        "description": _text(fields.get("description"))[:MAX_DESCRIPTION],
        "comments": [
            {"who": ((c.get("author") or {}).get("displayName") or "?"),
             "body": _text(c.get("body"))[:MAX_COMMENT]}
            for c in comments],
    }


def render(tickets):
    """The tracker section of the prompt, or "" when there is nothing to say."""
    blocks = []
    for t in tickets:
        if not t:
            continue
        lines = [f"### {t['key']} — {t['summary']}  [{t['type']}, {t['status']}]"]
        if t["description"]:
            lines.append(t["description"])
        for c in t["comments"]:
            if c["body"].strip():
                lines.append(f"[comment — {c['who']}]\n{c['body']}")
        blocks.append("\n\n".join(lines))
    if not blocks:
        return ""
    return ("\nWHAT THE TICKET ASKED FOR. This is the change's intent, and it is\n"
            "the one thing the diff cannot tell you. A change that is correct and\n"
            "does not do what was asked is a finding — say which requirement is\n"
            "unmet. Do NOT invent scope from it: a ticket describing more than\n"
            "this PR claims to deliver is normal, and a partial PR is not a\n"
            "defect unless it says otherwise.\n\n" + "\n\n---\n\n".join(blocks) + "\n")

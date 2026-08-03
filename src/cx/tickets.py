"""Support tickets, written locally as JSONL.

Once a Match ID is captured for a match-level issue, there is enough to hand a
human agent: what broke, which match, and the player's own words. That record is
appended to `tickets/tickets.jsonl`.

This deliberately does not talk to Zendesk. Filing into a real helpdesk sends
player data off the machine and needs credentials, which is a decision for the
operator, not a default. `to_zendesk_payload` shapes a record the way the
Zendesk Tickets API expects, so wiring it up later is one HTTP call.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT

TICKETS_DIR = Path(os.environ.get("CX_TICKETS_DIR", ROOT / "tickets"))
TICKETS_FILE = TICKETS_DIR / "tickets.jsonl"

# Ordered most specific first: a message mentioning both a crash and a missing
# prize is really a payout problem, and routing should reflect that.
_CATEGORIES: list[tuple[str, str]] = [
    ("suspected_cheating", "cheat rigged"),
    ("replay_unavailable", "replay"),
    ("missing_payout", "prize payout winnings refund reimburse chargeback"),
    ("pending_or_canceled", "pending canceled cancelled"),
    ("aborted_or_crash", "crash abort froze frozen disconnect stuck lag kicked booted glitch error"),
]

CATEGORY_LABELS = {
    "unclear_request": "Unclear request — needs a human",
    "suspected_cheating": "Suspected cheating",
    "replay_unavailable": "Replay unavailable",
    "missing_payout": "Missing prize or payout",
    "pending_or_canceled": "Pending or canceled match",
    "aborted_or_crash": "Aborted / crashed match",
    "match_issue": "Match issue",
}


@dataclass
class Ticket:
    ticket_id: str
    created_at: str
    category: str
    match_id: str | None
    summary: str
    session_id: str | None = None
    transcript: list[dict] = field(default_factory=list)
    articles: list[dict] = field(default_factory=list)
    status: str = "open"

    @property
    def label(self) -> str:
        return CATEGORY_LABELS.get(self.category, CATEGORY_LABELS["match_issue"])

    def to_zendesk_payload(self) -> dict:
        """Shape this ticket the way the Zendesk Tickets API expects.

        Unused until someone supplies credentials and decides to file for real.
        """
        body = [f"Match ID: {self.match_id or 'not provided'}", "", "Transcript:"]
        body += [f"{m['role']}: {m['content']}" for m in self.transcript]
        if self.articles:
            body += ["", "Help-center articles consulted:"]
            body += [f"- {a['title']} (article {a['id']})" for a in self.articles]
        return {
            "ticket": {
                "subject": f"[{self.label}] Match {self.match_id or 'unknown'}",
                "comment": {"body": "\n".join(body)},
                "tags": ["skillz-assistant", self.category],
                "external_id": self.ticket_id,
            }
        }


def new_ticket_id() -> str:
    """A short reference a player can read back over the phone."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"SKZ-{stamp}-{secrets.token_hex(2).upper()}"


def categorise(text: str) -> str:
    lowered = (text or "").lower()
    for category, terms in _CATEGORIES:
        if any(term in lowered for term in terms.split()):
            return category
    return "match_issue"


def write(ticket: Ticket) -> Ticket:
    TICKETS_DIR.mkdir(parents=True, exist_ok=True)
    with TICKETS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(ticket)) + "\n")
    return ticket


def load_all() -> list[Ticket]:
    if not TICKETS_FILE.exists():
        return []
    tickets = []
    for line in TICKETS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            tickets.append(Ticket(**json.loads(line)))
    return tickets

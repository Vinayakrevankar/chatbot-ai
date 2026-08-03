"""Recognising match-level problems and collecting the Match ID.

Support cannot investigate a specific match without its Match ID, so when a
player reports a crash, an abort, a missing prize or a suspect opponent, the
assistant has to come away with that number. Detection is deliberately
deterministic rather than left to the model: whether to ask is a policy
decision, and policy should not vary with sampling.
"""

from __future__ import annotations

import re

# Words that mean "something went wrong with one specific match". Informational
# questions ("how do trophies work") must not trip these.
_ISSUE_TERMS = re.compile(
    r"""\b(
        crash(ed|es|ing)? | abort(ed|ing)? | froze | frozen | freezing
      | disconnect(ed|ion|ing)? | glitch(ed|y)? | stuck | lag(ged|ging)?
      | pending | cancell?ed | cancell?ing
      | replay | cheat(ed|er|ing)? | rigged
      | refund(ed)? | reimburse(d)? | chargeback
      | (didn't|did\snot|couldn't|could\snot)\s(finish|complete|submit|load)
      | (wrong|missing|incorrect|lost)\s(score|prize|payout|winnings|money|entry)
      | (never|didn't|did\snot|haven't|have\snot)\s(got|get|gotten|receive|received)
        \s(my\s)?(prize|payout|winnings|money|refund)
      | kicked\sout | booted | error
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# An explicit label wins over a bare number appearing anywhere in the message.
_LABELLED_ID = re.compile(
    r"\b(?:match|game)\s*(?:id|#|number)?\s*[:#]?\s*(\d{5,20})\b", re.IGNORECASE
)
_BARE_ID = re.compile(r"\b(\d{6,20})\b")

# Phrases meaning "I can't give you that", so the assistant explains where to
# find it instead of asking a second time.
_CANNOT_PROVIDE = re.compile(
    r"""(
        \b(don't|dont|do\snot|doesn't|can't|cant|cannot|couldn't)\b.{0,24}
        \b(have|find|know|see|locate|remember|get)\b
      | \bno\sidea\b | \bnot\ssure\b | \bwhere\b.{0,20}\b(find|is|get)\b
      | \bhow\b.{0,20}\b(find|get|locate)\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def is_match_issue(text: str) -> bool:
    """True when the text describes a problem with a specific match."""
    return bool(_ISSUE_TERMS.search(text or ""))


def extract_match_id(text: str) -> str | None:
    """Pull a Match ID out of a player's message, if there is one."""
    if not text:
        return None
    labelled = _LABELLED_ID.search(text)
    if labelled:
        return labelled.group(1)
    bare = _BARE_ID.search(text)
    return bare.group(1) if bare else None


def cannot_provide(text: str) -> bool:
    """True when the player is saying they don't have or can't find the ID."""
    return bool(_CANNOT_PROVIDE.search(text or ""))

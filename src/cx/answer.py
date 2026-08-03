"""Turn retrieved chunks into a grounded support answer.

Two things make this more than a single prompt call. Multi-turn: "how long does
that take?" is meaningless to a retriever, so the latest message is rewritten
into a standalone query first. And match-level issues: support cannot
investigate a crash or an abort without the Match ID, so the assistant has to
collect it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

import requests

from .config import CHAT_MODEL, OLLAMA_HOST, REFUSAL, TOP_K
from .embed import OllamaError
from .matchid import cannot_provide, extract_match_id, is_match_issue
from .store import Hit, Index, tokenize
from .tickets import Ticket, categorise, new_ticket_id
from .tickets import write as write_ticket

SYSTEM_PROMPT = f"""You are a customer support assistant for Skillz, answering \
players from the official help center.

Answer using ONLY the numbered sources provided. They are the entire truth \
available to you.

Rules:
- Use whatever the sources do say. If a source answers the question in passing, \
or as one item in a list, that is a real answer — give it. Only when the sources \
contain nothing relevant at all, reply exactly: "{REFUSAL}" Never guess and never \
fall back on general knowledge about gaming apps, but do not refuse work the \
sources can actually do.
- Cite the sources you used inline, like [1] or [2]. Every factual claim needs a \
citation.
- Never invent policy specifics. Dollar amounts, percentages, timeframes, fees, \
eligibility rules, email addresses and URLs must appear verbatim in a source or \
not at all.
- Be concise: two to four sentences. Use a short numbered list only when the \
answer is genuinely a sequence of steps.
- Write plainly and warmly to a player, not about them. No preamble, no \
"according to the sources".
- Earlier turns are context for what the player means. Facts still come only \
from the sources in the current turn."""

# Deliberately lists no example product nouns: naming them here caused the
# model to inject them into unrelated rewrites ("Dogecoin?" came back as a
# question about Ticketz and Z Coins).
CONDENSE_PROMPT = """Rewrite the player's latest message into a standalone \
search query for a help-center search engine.

Rules:
- Resolve pronouns and references using the conversation.
- Keep every noun and product name the player actually wrote, spelled the same \
way.
- Never introduce a topic, product or term the player did not mention.
- If the message already stands alone, return it unchanged.
- Output the query only. No quotes, no explanation, no preamble."""

# Appended to the system prompt for the turn. Asking is a policy decision, so
# it is decided in code and handed to the model as an instruction.
ASK_FOR_MATCH_ID = """

This player is reporting a problem with one specific match. Support cannot \
investigate without the Match ID. After answering, ask them for it in one short \
sentence. Ask once, warmly, and do not lecture."""

EXPLAIN_WHERE_TO_FIND = """

This player has a match problem and has just indicated they do not have, or \
cannot find, their Match ID. Do NOT ask for it again. Instead use the sources \
to tell them exactly where to find it in the app, as short numbered steps."""

ASK_TO_CLARIFY = """

The player's message is too vague to answer from the help center, and the \
retrieved sources are probably not what they meant. Do NOT answer from the \
sources and do NOT cite anything. Instead, in at most two sentences: say you \
want to make sure you point them at the right thing, and ask one specific \
question that would narrow it down — what they were trying to do, or which part \
of the app it happened in. Stay warm and brief."""

# Fixed text, not generated. Handing off is a policy outcome, and the model
# kept echoing its own previous "could you tell me more?" turn instead of
# actually closing the loop. A canned message is also instant.
ESCALATION_TEXT = (
    "I'm not able to pin this down from the help center, so let's get a human "
    "support agent on it. I've drafted ticket {ticket_id} below — have a read, "
    "edit the description if you'd like, and send it when you're happy."
)

CONFIRM_MATCH_ID = """

The player has provided Match ID {match_id}, and a DRAFT ticket {ticket_id} is \
now shown below your reply for them to review, edit and send. In one short \
sentence: confirm the Match ID, and tell them to check the draft below and send \
it when ready. Do NOT say the ticket has been created, opened, raised or \
submitted — it has not been, and it will not be until they send it. Then answer \
anything else they asked."""

# How many prior messages to carry. Support threads are short; more history
# mostly adds latency and lets stale topics pull retrieval off course.
MAX_HISTORY_MESSAGES = 8

# Function words: too common to protect in a rewrite, and never enough on
# their own to search on. Contractions matter more than they look — "I don't
# have" was treated as a searchable question until "don't" and "have" landed
# here, and it retrieved an article about forgetting a password.
_STOPWORDS = frozenset(
    """a about all also am an and another any anything are aren't as at back be
    been being both but by can can't cannot could couldn't did didn't do does
    doesn't doing don't done each even ever every few for from further get gets
    getting got had hadn't has hasn't have haven't having he her here hers him
    his how i i'd i'll i'm i've if in into is isn't it it's its just know let
    like may maybe me might mine more most much must my need needs no nor not
    now of off ok okay on once one only or other our ours out over own please
    really same say see shall she should shouldn't so some something still such
    sure than thanks that that's the their theirs them then there there's these
    they this those thought through to too try under until up us use used using
    very want was wasn't we well were what what's when where whether which
    while who whom why will with won't would wouldn't yeah yes yet you you're
    your yours""".split()
)

# Corpus IDF above which a term counts as distinctive. At 141 chunks this is
# roughly "appears in six chunks or fewer".
_DISTINCTIVE_IDF = 3.0

# Measured on this corpus: real questions bottom out around 0.74 top cosine
# ("my game crashed" 0.742), while vague ones top out around 0.738 ("game").
# The floor sits just under the clear band, so borderline queries get answered
# rather than interrogated — being asked to rephrase a fair question is worse
# than a slightly loose answer.
_VAGUE_COSINE_FLOOR = 0.72
# A lone content word needs a stronger match to count as a real question:
# "What are Ticketz?" scores 0.766, "game" only 0.738.
_LONE_TERM_COSINE = 0.75


@dataclass
class Conversation:
    """Everything carried between turns of one support thread."""

    history: list[dict] = field(default_factory=list)
    match_id: str | None = None
    match_id_asks: int = 0
    clarify_asks: int = 0
    ticket_id: str | None = None
    # Drafted and shown to the player for review. Becomes a real ticket only
    # when they submit it.
    pending_ticket: Ticket | None = None

    @property
    def ticket_in_flight(self) -> str | None:
        """Reference of the ticket already raised or awaiting the player."""
        if self.ticket_id:
            return self.ticket_id
        return self.pending_ticket.ticket_id if self.pending_ticket else None


@dataclass
class Prepared:
    """The result of retrieval and policy, ready to hand to the model."""

    hits: list[Hit]
    search_query: str
    messages: list[dict]
    match_id: str | None = None
    asked_for_match_id: bool = False
    asked_to_clarify: bool = False
    escalated: bool = False
    # When set, this is the reply verbatim and the model is not called at all.
    canned_reply: str | None = None
    # Reserved but not yet written: if generation fails the ticket never lands,
    # while the model can still quote its reference in the reply.
    ticket: Ticket | None = None


@dataclass
class Answer:
    text: str
    hits: list[Hit] = field(default_factory=list)
    search_query: str = ""
    match_id: str | None = None
    ticket: Ticket | None = None


def build_context(hits: list[Hit]) -> str:
    return "\n\n---\n\n".join(
        f"[{i}] (article {h.chunk.article_id})\n{h.chunk.text}"
        for i, h in enumerate(hits, 1)
    )


def build_messages(
    question: str,
    hits: list[Hit],
    history: list[dict] | None = None,
    directive: str = "",
) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT + directive}]
    if history:
        messages.extend(history[-MAX_HISTORY_MESSAGES:])
    messages.append(
        {
            "role": "user",
            "content": (
                f"Sources:\n\n{build_context(hits)}\n\n---\n\n"
                f"Player question: {question}"
            ),
        }
    )
    return messages


def _post_chat(payload: dict, *, stream: bool, timeout: int):
    try:
        return requests.post(
            f"{OLLAMA_HOST}/api/chat", json=payload, stream=stream, timeout=timeout
        )
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. Start it with `ollama serve`."
        ) from e


def _chat_stream(messages: list[dict], temperature: float) -> Iterator[str]:
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": True,
        # Low temperature: for support answers, faithfulness to the KB matters
        # far more than variety of phrasing.
        "options": {"temperature": temperature},
    }
    r = _post_chat(payload, stream=True, timeout=300)
    if r.status_code != 200:
        raise OllamaError(f"/api/chat returned {r.status_code}: {r.text[:400]}")

    for line in r.iter_lines():
        if not line:
            continue
        event = json.loads(line)
        piece = event.get("message", {}).get("content", "")
        if piece:
            yield piece
        if event.get("done"):
            break


def _distinctive_terms(question: str, index: Index | None) -> set[str]:
    """Terms in `question` rare enough that losing them changes what was asked.

    Rarity comes from the corpus itself, via the BM25 index. A term the corpus
    has never seen ("Dogecoin") is maximally distinctive; common words like
    "long" or "take" are not, so a rewrite is still free to drop them.
    """
    if index is None:
        return set()
    terms = set()
    for token in tokenize(question):
        if len(token) < 4 or token in _STOPWORDS:
            continue
        idf = index.bm25.idf.get(token)
        if idf is None or idf >= _DISTINCTIVE_IDF:
            terms.add(token)
    return terms


def _content_terms(text: str) -> set[str]:
    """Terms substantial enough to search on at all."""
    return {t for t in tokenize(text) if len(t) >= 4 and t not in _STOPWORDS}


def _last_substantive_message(history: list[dict]) -> str:
    for message in reversed(history):
        if message.get("role") == "user" and _content_terms(message.get("content", "")):
            return message["content"]
    return ""


def condense_query(
    question: str, history: list[dict] | None, index: Index | None = None
) -> str:
    """Rewrite a follow-up into a standalone query, with fallbacks that keep it
    anchored to the conversation."""
    if not history:
        return question

    # "I don't have", "yes", "it still didn't work" contain nothing searchable.
    # Left alone they retrieve on stopword coincidence — "I don't have" once
    # matched an article about forgetting a password, purely on "don't". Anchor
    # them to the last real thing the player asked.
    anchor = "" if _content_terms(question) else _last_substantive_message(history)
    base = f"{anchor} {question}".strip() if anchor else question

    transcript = "\n".join(
        f"{m['role']}: {m['content']}" for m in history[-MAX_HISTORY_MESSAGES:]
    )
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": CONDENSE_PROMPT},
            {
                "role": "user",
                "content": f"Conversation:\n{transcript}\n\nLatest message: {question}",
            },
        ],
        "stream": False,
        # Deterministic, and capped short: this is a rewrite, not an answer.
        "options": {"temperature": 0.0, "num_predict": 64},
    }
    try:
        r = _post_chat(payload, stream=False, timeout=60)
        if r.status_code != 200:
            return base
        rewritten = r.json().get("message", {}).get("content", "").strip()
    except (OllamaError, ValueError, requests.exceptions.RequestException):
        return base

    rewritten = rewritten.splitlines()[0].strip().strip('"').strip()
    # A rewrite that came back empty or rambling means the model ignored the
    # instruction; the anchored question is a safer query than a bad paraphrase.
    if not rewritten or len(rewritten) > 300:
        return base

    # The model often decides a bare fragment is "already standalone" and hands
    # it straight back. That is exactly the case anchoring exists for.
    if not _content_terms(rewritten):
        return base

    # A rewrite must not drop a distinctive term the player used...
    if _distinctive_terms(question, index) - set(tokenize(rewritten)):
        return base

    # ...nor invent one the conversation never mentioned, which is how a
    # question about Dogecoin turned into one about Ticketz.
    spoken = " ".join(
        [m.get("content", "") for m in history[-MAX_HISTORY_MESSAGES:]] + [question]
    )
    if _distinctive_terms(rewritten, index) - set(tokenize(spoken)):
        return base

    return rewritten


def _issue_in_thread(question: str, history: list[dict]) -> bool:
    """Whether this thread is about a specific broken match.

    Checked across the thread, not just the latest message: a player who opened
    with "Aborted" and then said "I don't know where to find match ID" is still
    reporting the same problem.
    """
    if is_match_issue(question):
        return True
    return any(
        m.get("role") == "user" and is_match_issue(m.get("content", ""))
        for m in history[-MAX_HISTORY_MESSAGES:]
    )


def _match_id_directive(
    question: str, conversation: Conversation
) -> tuple[str, str | None, bool]:
    """Decide what to do about the Match ID this turn.

    Returns the system-prompt directive, the Match ID now known, and whether
    this turn asks for it.
    """
    if not _issue_in_thread(question, conversation.history):
        return "", conversation.match_id, False

    supplied = extract_match_id(question)
    if supplied and supplied != conversation.match_id:
        # The confirmation wording is built in `prepare`, which is where the
        # ticket reference it has to quote gets reserved.
        return "", supplied, False

    known = conversation.match_id
    if known:
        return "", known, False

    # Asking twice for something the player has already said they cannot find
    # is how support bots earn their reputation.
    if cannot_provide(question) or conversation.match_id_asks >= 1:
        return EXPLAIN_WHERE_TO_FIND, None, False

    return ASK_FOR_MATCH_ID, None, True


def is_vague(query: str, hits: list[Hit]) -> bool:
    """Whether we plausibly understood the question at all.

    Judged on the condensed search query rather than the raw message, so a
    context-dependent reply like "I don't have" — which anchoring has already
    expanded into something specific — is not mistaken for a vague one.
    """
    content = _content_terms(query)
    if not content:
        return True
    if not hits:
        return True

    top = max(h.dense_score for h in hits)
    if top < _VAGUE_COSINE_FLOOR:
        return True
    return len(content) == 1 and top < _LONE_TERM_COSINE


def _reserve_ticket(
    question: str,
    conversation: Conversation,
    hits: list[Hit],
    *,
    match_id: str | None = None,
    category: str | None = None,
) -> Ticket:
    """Build a ticket record for this thread.

    The summary is the player's own words, not a paraphrase — that is what an
    agent picking this up actually wants to read.
    """
    issue_text = next(
        (
            m["content"]
            for m in conversation.history
            if m.get("role") == "user" and is_match_issue(m.get("content", ""))
        ),
        question,
    )
    if category is None:
        category = categorise(issue_text)
    else:
        # An escalation is about the whole thread, not one matched phrase.
        issue_text = next(
            (m["content"] for m in conversation.history if m.get("role") == "user"),
            question,
        )

    return Ticket(
        ticket_id=new_ticket_id(),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        category=category,
        match_id=match_id,
        summary=issue_text.strip(),
        transcript=[*conversation.history, {"role": "user", "content": question}],
        articles=[
            {"id": h.chunk.article_id, "title": h.chunk.title} for h in hits[:3]
        ],
    )


def prepare(
    index: Index,
    question: str,
    conversation: Conversation | None = None,
    top_k: int = TOP_K,
) -> Prepared:
    """Retrieve and apply policy, without calling the model for an answer yet."""
    conversation = conversation or Conversation()
    search_query = condense_query(question, conversation.history, index)
    hits = index.search(search_query, top_k=top_k)
    directive, match_id, asked = _match_id_directive(question, conversation)

    # A Match ID on a match-level thread is everything an agent needs, so the
    # ticket opens itself. One per thread.
    ticket = None
    clarify = False
    escalate = False
    canned = None

    if match_id and not conversation.ticket_in_flight:
        ticket = _reserve_ticket(question, conversation, hits, match_id=match_id)
        directive = CONFIRM_MATCH_ID.format(
            match_id=match_id, ticket_id=ticket.ticket_id
        )
    elif not directive and is_vague(search_query, hits):
        # Collecting a Match ID takes precedence: while that is in flight, a
        # short reply is an answer to our question, not a vague request.
        if conversation.clarify_asks >= 1:
            escalate = True
            reference = conversation.ticket_in_flight
            if not reference:
                ticket = _reserve_ticket(
                    question, conversation, hits, category="unclear_request"
                )
                reference = ticket.ticket_id
            canned = ESCALATION_TEXT.format(ticket_id=reference)
        else:
            clarify = True
            directive = ASK_TO_CLARIFY

    return Prepared(
        hits=hits,
        search_query=search_query,
        messages=build_messages(question, hits, conversation.history, directive),
        match_id=match_id,
        asked_for_match_id=asked,
        asked_to_clarify=clarify,
        escalated=escalate,
        canned_reply=canned,
        ticket=ticket,
    )


def commit(
    conversation: Conversation, question: str, reply: str, prepared: Prepared
) -> None:
    """Fold one completed turn back into the conversation."""
    conversation.history.append({"role": "user", "content": question})
    conversation.history.append({"role": "assistant", "content": reply})
    if len(conversation.history) > MAX_HISTORY_MESSAGES:
        del conversation.history[:-MAX_HISTORY_MESSAGES]

    if prepared.match_id:
        conversation.match_id = prepared.match_id
    if prepared.asked_for_match_id:
        conversation.match_id_asks += 1
    if prepared.asked_to_clarify:
        conversation.clarify_asks += 1
    # Held as a draft, not written. The player reviews it first; `submit_ticket`
    # is what actually raises it.
    if prepared.ticket and not conversation.ticket_in_flight:
        conversation.pending_ticket = prepared.ticket
        # The transcript has to include the turn that just finished, or the
        # player reviews a history missing its own last exchange.
        prepared.ticket.transcript = list(conversation.history)


def submit_ticket(
    conversation: Conversation, summary: str | None = None
) -> Ticket | None:
    """Raise the drafted ticket. This is the only place a ticket is written.

    `summary` is the player's edited description. Only that half is taken from
    them — the transcript below the divider is composed from the conversation,
    so it cannot be rewritten on the way out.
    """
    ticket = conversation.pending_ticket
    if ticket is None:
        return None

    if summary is not None and summary.strip():
        ticket.summary = summary.strip()

    write_ticket(ticket)
    conversation.ticket_id = ticket.ticket_id
    conversation.pending_ticket = None
    return ticket


def discard_ticket(conversation: Conversation) -> bool:
    """Throw away the draft without raising anything."""
    if conversation.pending_ticket is None:
        return False
    conversation.pending_ticket = None
    return True


def answer(
    index: Index,
    question: str,
    *,
    conversation: Conversation | None = None,
    top_k: int = TOP_K,
    temperature: float = 0.1,
    stream_to: object | None = None,
) -> Answer:
    """Retrieve, then generate. `stream_to` is any object with a `.write`."""
    conversation = conversation or Conversation()
    prepared = prepare(index, question, conversation, top_k)

    fixed = prepared.canned_reply or (None if prepared.hits else REFUSAL)
    if fixed is not None:
        if stream_to is not None:
            stream_to.write(fixed)
            stream_to.flush()
        commit(conversation, question, fixed, prepared)
        return Answer(
            fixed,
            prepared.hits,
            prepared.search_query,
            prepared.match_id,
            prepared.ticket,
        )

    pieces: list[str] = []
    for piece in _chat_stream(prepared.messages, temperature):
        pieces.append(piece)
        if stream_to is not None:
            stream_to.write(piece)
            stream_to.flush()

    reply = "".join(pieces).strip()
    commit(conversation, question, reply, prepared)
    return Answer(
        reply, prepared.hits, prepared.search_query, prepared.match_id, prepared.ticket
    )

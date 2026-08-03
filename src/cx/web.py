"""FastAPI server backing the chat UI.

Streams NDJSON rather than SSE: it is trivially parseable from `fetch` and
carries structured events, so the UI can render retrieved sources the moment
retrieval finishes, while the answer is still being generated.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import config
from .answer import Conversation, commit, prepare
from .answer import _chat_stream  # streaming primitive, shared with the CLI
from .config import REFUSAL
from .embed import OllamaError
from .store import Index

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Skillz support assistant")

# Loaded once at startup: the index is ~141 vectors, but re-reading it per
# request would still add pointless latency.
_index: Index | None = None
# Server-side transcripts, keyed by session. In-memory on purpose — this is a
# single-user local tool, and support transcripts are not worth persisting
# without a retention decision behind them.
_sessions: dict[str, Conversation] = {}


def get_index() -> Index:
    global _index
    if _index is None:
        _index = Index.load(config.INDEX_DIR)
    return _index


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    top_k: int = config.TOP_K
    temperature: float = 0.1


def _event(kind: str, **payload) -> str:
    return json.dumps({"type": kind, **payload}) + "\n"


def _stream_reply(req: ChatRequest, session_id: str) -> Iterator[str]:
    conversation = _sessions.setdefault(session_id, Conversation())
    try:
        index = get_index()
        prepared = prepare(index, req.message, conversation, req.top_k)
        query = prepared.search_query

        yield _event(
            "sources",
            session_id=session_id,
            search_query=query,
            rewritten=query.strip().lower() != req.message.strip().lower(),
            match_id=prepared.match_id,
            asking_for_match_id=prepared.asked_for_match_id,
            sources=[
                {
                    "n": i,
                    "title": h.chunk.title,
                    "article_ids": [h.chunk.article_id, *h.chunk.aliases],
                    "score": round(h.score, 4),
                    "text": h.chunk.text,
                }
                for i, h in enumerate(prepared.hits, 1)
            ],
        )

        fixed = prepared.canned_reply or (None if prepared.hits else REFUSAL)
        if fixed is not None:
            yield _event("token", text=fixed)
            reply = fixed
        else:
            pieces: list[str] = []
            for piece in _chat_stream(prepared.messages, req.temperature):
                pieces.append(piece)
                yield _event("token", text=piece)
            reply = "".join(pieces).strip()

        if prepared.ticket:
            prepared.ticket.session_id = session_id
        commit(conversation, req.message, reply, prepared)

        ticket = prepared.ticket
        yield _event(
            "done",
            match_id=conversation.match_id,
            ticket=(
                {
                    "id": ticket.ticket_id,
                    "label": ticket.label,
                    "match_id": ticket.match_id,
                    "summary": ticket.summary,
                }
                if ticket
                else None
            ),
        )

    except OllamaError as e:
        yield _event("error", message=str(e))
    except FileNotFoundError as e:
        yield _event("error", message=f"{e}")


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    session_id = req.session_id or uuid.uuid4().hex
    return StreamingResponse(
        _stream_reply(req, session_id),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/api/reset")
def reset(payload: dict) -> dict:
    session_id = payload.get("session_id")
    if session_id:
        _sessions.pop(session_id, None)
    return {"ok": True, "session_id": uuid.uuid4().hex}


@app.get("/api/health")
def health() -> dict:
    try:
        index = get_index()
    except FileNotFoundError as e:
        return {"ok": False, "error": str(e)}
    return {
        "ok": True,
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
        "chunks": len(index.chunks),
        "articles": index.meta.get("articles"),
        "source_articles": index.meta.get("source_articles"),
        "merged": len(index.meta.get("merged_duplicates", [])),
    }


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

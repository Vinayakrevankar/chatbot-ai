"""Tunables. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("CX_DATA_DIR", ROOT / "data" / "articles"))
INDEX_DIR = Path(os.environ.get("CX_INDEX_DIR", ROOT / "index"))

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
# bge-m3 covers 100+ languages. The knowledge base is English, but players are
# not: nomic-embed-text scored a Spanish question at 0.50 against its own
# English answer, under the vagueness floor, so non-English players were asked
# to rephrase and then escalated. bge-m3 matches across languages directly.
EMBED_MODEL = os.environ.get("CX_EMBED_MODEL", "bge-m3")
# qwen2.5:14b (~9GB at 4-bit) is the largest model that sits comfortably in
# 16GB alongside the embedder. It follows the grounding rules more reliably
# than llama3.1:8b and resolves vague follow-ups better, at roughly twice the
# latency. Set CX_CHAT_MODEL=llama3.1:8b to trade accuracy back for speed.
CHAT_MODEL = os.environ.get("CX_CHAT_MODEL", "qwen2.5:14b-instruct")

# Task prefixes are per-model, not universal. nomic-embed-text is trained with
# them and loses quality without them; bge-m3 is trained without them and loses
# quality with them, since the prefix is just unexpected tokens. Getting this
# wrong degrades retrieval silently, so it is keyed off the model name.
_EMBED_PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
}
EMBED_DOC_PREFIX, EMBED_QUERY_PREFIX = _EMBED_PREFIXES.get(
    EMBED_MODEL.split(":")[0], ("", "")
)

# Articles are small (median ~800 chars), so most become a single chunk and
# keep their full context. Only the handful of long ones get split.
MAX_CHUNK_CHARS = 1800
CHUNK_OVERLAP_BLOCKS = 1

# The KB ships parallel Skillz / Skillz Arena copies of ~35 articles that are
# near-identical. Above this word-level Jaccard overlap they are merged.
DEDUPE_THRESHOLD = 0.90

TOP_K = 5
# Reciprocal-rank-fusion damping. 60 is the value from the original RRF paper.
RRF_K = 60

# HTTP Basic credentials. Empty means no auth, which is fine on localhost and
# not fine the moment the server is reachable from anywhere else — `cx tunnel`
# refuses to start without these.
AUTH_USER = os.environ.get("CX_AUTH_USER", "")
AUTH_PASS = os.environ.get("CX_AUTH_PASS", "")

REFUSAL = (
    "I don't have that in the help center — let me hand you to a support agent."
)

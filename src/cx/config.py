"""Tunables. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("CX_DATA_DIR", ROOT / "data" / "articles"))
INDEX_DIR = Path(os.environ.get("CX_INDEX_DIR", ROOT / "index"))

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
EMBED_MODEL = os.environ.get("CX_EMBED_MODEL", "nomic-embed-text")
# qwen2.5:14b (~9GB at 4-bit) is the largest model that sits comfortably in
# 16GB alongside the embedder. It follows the grounding rules more reliably
# than llama3.1:8b and resolves vague follow-ups better, at roughly twice the
# latency. Set CX_CHAT_MODEL=llama3.1:8b to trade accuracy back for speed.
CHAT_MODEL = os.environ.get("CX_CHAT_MODEL", "qwen2.5:14b-instruct")

# nomic-embed-text is trained with task prefixes; using them lifts retrieval
# quality noticeably over embedding raw text on both sides.
EMBED_DOC_PREFIX = "search_document: "
EMBED_QUERY_PREFIX = "search_query: "

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

REFUSAL = (
    "I don't have that in the help center — let me hand you to a support agent."
)

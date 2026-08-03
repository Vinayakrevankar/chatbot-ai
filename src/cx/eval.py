"""Retrieval sanity checks.

Article titles are real support questions, which makes them a free labelled
set: ask the title, expect that article back. It is an easy benchmark, so treat
it as a smoke test that catches broken chunking or a mismatched embedding
prefix — not as evidence of production quality. For that, point `--file` at
real questions from your ticket log.
"""

from __future__ import annotations

import json
from pathlib import Path

from .store import Index


def title_cases(index: Index) -> list[dict]:
    seen: set[str] = set()
    cases = []
    for chunk in index.chunks:
        if chunk.article_id in seen:
            continue
        seen.add(chunk.article_id)
        cases.append({"question": chunk.title, "article_id": chunk.article_id})
    return cases


def load_cases(path: Path) -> list[dict]:
    """Read a JSON list of {"question": ..., "article_id": ...} objects."""
    cases = json.loads(path.read_text(encoding="utf-8"))
    for case in cases:
        if "question" not in case or "article_id" not in case:
            raise ValueError(
                f"Every eval case needs 'question' and 'article_id'; got {case!r}"
            )
    return cases


def evaluate(index: Index, cases: list[dict], top_k: int = 5) -> dict:
    hits_at_1 = 0
    hits_at_k = 0
    reciprocal = 0.0
    misses = []

    for case in cases:
        results = index.search(case["question"], top_k=top_k)
        expected = str(case["article_id"])

        rank = None
        for i, hit in enumerate(results):
            # A merged duplicate is a correct answer under either of its ids.
            if expected == hit.chunk.article_id or expected in hit.chunk.aliases:
                rank = i
                break

        if rank == 0:
            hits_at_1 += 1
        if rank is not None:
            hits_at_k += 1
            reciprocal += 1.0 / (rank + 1)
        else:
            misses.append(
                {
                    "question": case["question"],
                    "expected": expected,
                    "got": [h.chunk.title for h in results[:3]],
                }
            )

    n = max(len(cases), 1)
    return {
        "n": len(cases),
        "top_k": top_k,
        "recall@1": hits_at_1 / n,
        f"recall@{top_k}": hits_at_k / n,
        "mrr": reciprocal / n,
        "misses": misses,
    }

"""Index build/load plus hybrid retrieval.

The corpus is ~119 short articles, so a brute-force dot product over a dense
matrix is instant and a vector database would be pure overhead. Retrieval fuses
that dense score with BM25: support questions lean on exact product nouns
("Ticketz", "Z Coins", "Gemz") that embeddings happily blur together, and
lexical scoring is what keeps those distinct.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import INDEX_DIR, RRF_K, TOP_K
from .corpus import Chunk
from .embed import embed, embed_query

# Unicode-aware: `[a-z0-9']+` matched nothing at all in Hindi or Japanese, so
# any such message looked like it contained no words. `[^\W_]` is "letter or
# digit in any script", and the apostrophe group keeps contractions whole so
# "don't" stays one token rather than becoming "don" and "t".
#
# Combining marks have to be spelled out. Python's `\w` excludes them, so the
# vowel signs in "निकासी" read as separators and the word shattered into 1-2
# character fragments. The range covers the diacritics and the Indic and Thai
# blocks, which all sit below U+2000.
_COMBINING = "".join(
    chr(c)
    for c in range(0x0300, 0x2000)
    if unicodedata.category(chr(c)) in {"Mn", "Mc"}
)
_WORDCHAR = rf"(?:[^\W_]|[{re.escape(_COMBINING)}])"
_TOKEN = re.compile(rf"{_WORDCHAR}+(?:['’]{_WORDCHAR}+)*", re.UNICODE)

BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    # The help center is full of typographic apostrophes ("opponent's") while
    # players type straight ones. Normalising makes those match lexically.
    return _TOKEN.findall(text.lower().replace("’", "'"))


class BM25:
    def __init__(self, docs: list[list[str]]) -> None:
        self.n = len(docs)
        self.doc_len = np.asarray([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.tf: list[Counter] = [Counter(d) for d in docs]

        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return out
        norm = BM25_K1 * (1 - BM25_B + BM25_B * self.doc_len / max(self.avgdl, 1e-9))
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            freqs = np.asarray([tf.get(term, 0) for tf in self.tf], dtype=np.float32)
            out += idf * (freqs * (BM25_K1 + 1)) / (freqs + norm)
        return out


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None
    lexical_rank: int | None
    # Raw cosine against the query. Unlike the fused score this is calibrated
    # and comparable across queries, which is what makes it usable as a
    # confidence signal for "did we understand the question at all".
    dense_score: float = 0.0
    # Best cosine anywhere in the corpus for this query, repeated on each hit.
    # It answers "did anything match?", which is not the same as the best score
    # among the returned hits — RRF and per-article dedup can leave the closest
    # chunk out of the top-k, and reading confidence off the survivors then
    # understates it.
    query_top_dense: float = 0.0


class Index:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, meta: dict) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.meta = meta
        self.bm25 = BM25([tokenize(c.text) for c in chunks])

    # -- persistence -------------------------------------------------------

    def save(self, index_dir: Path = INDEX_DIR) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self.embeddings)
        (index_dir / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], indent=2), encoding="utf-8"
        )
        (index_dir / "meta.json").write_text(
            json.dumps(self.meta, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> "Index":
        chunks_path = index_dir / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"No index at {index_dir}. Build one with `cx ingest`."
            )
        chunks = [Chunk.from_dict(d) for d in json.loads(chunks_path.read_text())]
        embeddings = np.load(index_dir / "embeddings.npy")
        meta_path = index_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(chunks, embeddings, meta)

    @classmethod
    def build(cls, chunks: list[Chunk], meta: dict | None = None) -> "Index":
        embeddings = embed([c.text for c in chunks])
        return cls(chunks, embeddings, meta or {})

    # -- retrieval ---------------------------------------------------------

    def search(self, query: str, top_k: int = TOP_K, pool: int = 25) -> list[Hit]:
        """Hybrid dense + BM25 retrieval fused with reciprocal rank fusion.

        RRF combines the two rankings without needing the scores to be on a
        comparable scale, which cosine similarity and BM25 emphatically are not.
        """
        if not self.chunks:
            return []

        dense = self.embeddings @ embed_query(query)
        lexical = self.bm25.scores(query)
        top_dense = float(dense.max())

        pool = min(pool, len(self.chunks))
        dense_order = np.argsort(-dense)[:pool]
        # A zero BM25 score means no query term occurs in the chunk, which is
        # not weak evidence of relevance — it is none. Ranking those anyway
        # handed real RRF weight to whatever `argsort` happened to put first,
        # and for a query in a language the corpus does not use, that is every
        # chunk: Spanish recall@1 sat at 0.11 purely on that noise.
        lexical_order = [i for i in np.argsort(-lexical)[:pool] if lexical[i] > 0]

        dense_rank = {int(idx): r for r, idx in enumerate(dense_order)}
        lexical_rank = {int(idx): r for r, idx in enumerate(lexical_order)}

        fused: dict[int, float] = {}
        for ranks in (dense_rank, lexical_rank):
            for idx, rank in ranks.items():
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        # Only one chunk per article reaches the model; a long article's
        # neighbouring chunks would otherwise fill the whole context.
        best: list[Hit] = []
        seen_articles: set[str] = set()
        for idx in sorted(fused, key=lambda i: -fused[i]):
            chunk = self.chunks[idx]
            if chunk.article_id in seen_articles:
                continue
            seen_articles.add(chunk.article_id)
            best.append(
                Hit(
                    chunk=chunk,
                    score=fused[idx],
                    dense_rank=dense_rank.get(idx),
                    lexical_rank=lexical_rank.get(idx),
                    dense_score=float(dense[idx]),
                    query_top_dense=top_dense,
                )
            )
            if len(best) >= top_k:
                break
        return best

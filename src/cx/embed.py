"""Embedding calls against the local Ollama server."""

from __future__ import annotations

import numpy as np
import requests

from .config import EMBED_DOC_PREFIX, EMBED_MODEL, EMBED_QUERY_PREFIX, OLLAMA_HOST


class OllamaError(RuntimeError):
    pass


def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    try:
        r = requests.post(f"{OLLAMA_HOST}{path}", json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. Start it with `ollama serve`."
        ) from e
    if r.status_code != 200:
        raise OllamaError(f"{path} returned {r.status_code}: {r.text[:400]}")
    return r.json()


def embed(texts: list[str], *, is_query: bool = False, batch_size: int = 32) -> np.ndarray:
    """Embed texts and return an L2-normalized float32 matrix.

    Normalizing here means cosine similarity later is a plain dot product.
    """
    prefix = EMBED_QUERY_PREFIX if is_query else EMBED_DOC_PREFIX
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[start : start + batch_size]]
        data = _post("/api/embed", {"model": EMBED_MODEL, "input": batch})
        got = data.get("embeddings")
        if not got:
            raise OllamaError(f"No embeddings returned for batch at offset {start}.")
        vectors.extend(got)

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Guard against a zero vector making the row NaN.
    np.maximum(norms, 1e-12, out=norms)
    return matrix / norms


def embed_query(text: str) -> np.ndarray:
    return embed([text], is_query=True)[0]

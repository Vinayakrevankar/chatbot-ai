"""Load help-center markdown, repair the export, drop duplicates, chunk.

The Zendesk export lost inline formatting: bold runs were flattened into their
own lines, so a sentence can arrive as five fragments. `reflow` puts those back
together before anything downstream sees them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import CHUNK_OVERLAP_BLOCKS, DEDUPE_THRESHOLD, MAX_CHUNK_CHARS

_ARTICLE_ID = re.compile(r"^(\d+)-")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_HEADING = re.compile(r"^\s*#{1,6}\s+")
_WORD = re.compile(r"[a-z0-9']+")
_PAYOUT_ROW = re.compile(r"^\$[\d,]+(\.\d{2})?$")

# Zendesk exports occasionally include a league results table: hundreds of rows
# of player display name, score and prize. They answer no support question and
# are full of personal data, so they are kept out of the index — which matters
# most when the assistant is exposed beyond localhost. One article in this
# corpus trips this; the next one closest has zero payout rows.
LEADERBOARD_ROW_THRESHOLD = 25
# A fragment starting with one of these is a continuation, not a new word.
_TRAILING_PUNCT = re.compile(r"^[.,;:!?)%\]]")


@dataclass
class Article:
    article_id: str
    title: str
    body: str
    path: Path
    # Article ids of near-identical copies folded into this one.
    aliases: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.body}"


@dataclass
class Chunk:
    chunk_id: str
    article_id: str
    title: str
    text: str
    ordinal: int
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "article_id": self.article_id,
            "title": self.title,
            "text": self.text,
            "ordinal": self.ordinal,
            "aliases": self.aliases,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def reflow(body: str) -> str:
    """Rejoin lines the export split mid-sentence, preserving lists and headings."""
    out_blocks: list[str] = []
    for block in re.split(r"\n\s*\n", body):
        lines = [ln.strip() for ln in block.splitlines()]
        lines = [ln for ln in lines if ln]
        if not lines:
            continue

        merged: list[str] = []
        for line in lines:
            starts_structure = bool(_LIST_ITEM.match(line) or _HEADING.match(line))
            if not merged or starts_structure or _LIST_ITEM.match(merged[-1]):
                # Keep list items and headings on their own lines. A line
                # following a list item also stays put, so wrapped bullets
                # don't get glued onto the bullet above them.
                merged.append(line)
            elif _TRAILING_PUNCT.match(line):
                merged[-1] = merged[-1].rstrip() + line
            else:
                merged[-1] = f"{merged[-1]} {line}"
        out_blocks.append("\n".join(merged))

    return "\n\n".join(out_blocks).strip()


def load_articles(data_dir: Path) -> list[Article]:
    """Read every .md file in `data_dir` into an Article."""
    articles: list[Article] = []
    for path in sorted(data_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            continue

        lines = raw.splitlines()
        if _HEADING.match(lines[0]):
            title = _HEADING.sub("", lines[0]).strip()
            body = "\n".join(lines[1:])
        else:
            title = path.stem
            body = raw

        m = _ARTICLE_ID.match(path.name)
        article_id = m.group(1) if m else path.stem

        body = reflow(body)
        if not body:
            continue
        articles.append(Article(article_id, title, body, path))

    return articles


def looks_like_leaderboard(body: str) -> bool:
    """True for articles that are mostly a table of players and their prizes."""
    rows = sum(1 for line in body.splitlines() if _PAYOUT_ROW.match(line.strip()))
    return rows >= LEADERBOARD_ROW_THRESHOLD


def _word_set(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedupe(articles: list[Article]) -> tuple[list[Article], list[tuple[str, str]]]:
    """Fold near-identical articles together.

    Skillz and Skillz Arena publish parallel copies of much of the KB. Left
    alone they crowd each other out of the top-k and spend context restating
    one answer. The longer copy wins; the loser's id is kept as an alias so
    citations can still point at either help center.

    Returns the surviving articles and the (kept_id, dropped_id) pairs.
    """
    # Longest first, so the more complete copy becomes canonical.
    ordered = sorted(articles, key=lambda a: len(a.text), reverse=True)
    words = {a.article_id: _word_set(a.text) for a in ordered}

    kept: list[Article] = []
    merges: list[tuple[str, str]] = []
    for article in ordered:
        match = None
        for candidate in kept:
            same_title = candidate.title.strip().lower() == article.title.strip().lower()
            overlap = _jaccard(words[candidate.article_id], words[article.article_id])
            # An exact title match is strong evidence on its own, but still
            # require substantial body overlap so genuinely different answers
            # under a shared title survive as separate articles.
            if overlap >= DEDUPE_THRESHOLD or (same_title and overlap >= 0.75):
                match = candidate
                break

        if match is None:
            kept.append(article)
        else:
            match.aliases.append(article.article_id)
            merges.append((match.article_id, article.article_id))

    kept.sort(key=lambda a: a.article_id)
    return kept, merges


def _blocks(body: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]


def chunk_article(article: Article) -> list[Chunk]:
    """Split one article into retrievable chunks.

    Every chunk carries the title, so a fragment retrieved on its own still
    says what question it answers.
    """
    header = f"# {article.title}\n\n"
    budget = MAX_CHUNK_CHARS - len(header)

    if len(article.body) <= budget:
        return [
            Chunk(
                chunk_id=f"{article.article_id}#0",
                article_id=article.article_id,
                title=article.title,
                text=header + article.body,
                ordinal=0,
                aliases=list(article.aliases),
            )
        ]

    blocks = _blocks(article.body)
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0

    for block in blocks:
        # A single oversized block still has to land somewhere; give it its own
        # chunk rather than dropping it.
        if current and size + len(block) + 2 > budget:
            groups.append(current)
            carry = current[-CHUNK_OVERLAP_BLOCKS:] if CHUNK_OVERLAP_BLOCKS else []
            current = list(carry)
            size = sum(len(b) + 2 for b in current)
        current.append(block)
        size += len(block) + 2

    if current:
        groups.append(current)

    return [
        Chunk(
            chunk_id=f"{article.article_id}#{i}",
            article_id=article.article_id,
            title=article.title,
            text=header + "\n\n".join(group),
            ordinal=i,
            aliases=list(article.aliases),
        )
        for i, group in enumerate(groups)
    ]


def build_chunks(
    data_dir: Path,
) -> tuple[list[Chunk], list[Article], list[tuple[str, str]], list[Article]]:
    articles = load_articles(data_dir)

    withheld = [a for a in articles if looks_like_leaderboard(a.body)]
    if withheld:
        excluded = {a.article_id for a in withheld}
        articles = [a for a in articles if a.article_id not in excluded]

    articles, merges = dedupe(articles)
    chunks = [c for a in articles for c in chunk_article(a)]
    return chunks, articles, merges, withheld

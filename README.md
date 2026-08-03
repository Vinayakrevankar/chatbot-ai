# Skillz support assistant (local RAG)

A customer-support assistant that runs entirely on this machine. It answers
player questions from the Skillz help-center articles in
`s3://skillz-support-kb-637423462092`, and cites the article it used.

Nothing leaves the machine: the model, the embeddings and the knowledge base
are all local.

## Why retrieval rather than fine-tuning

Fine-tuning teaches a model *style*; it does not reliably teach it *facts*. A
LoRA trained on 119 FAQ articles will fluently paraphrase your withdrawal policy
and get the timeframe wrong, with no way to tell that it did. Retrieval puts the
actual article text in front of the model at answer time, so answers are
quotable and traceable, and editing an article changes the answer immediately
instead of after a retraining run.

If you later want a specific brand voice, that is the part worth a LoRA — on top
of retrieval, not instead of it.

## Setup

Requires [Ollama](https://ollama.com) and the two models:

```bash
brew install ollama && ollama serve
```

```bash
ollama pull qwen2.5:14b-instruct && ollama pull nomic-embed-text
```

`qwen2.5:14b-instruct` (~9 GB at 4-bit) is the largest model that fits
comfortably in 16 GB alongside the embedder. It follows the grounding rules more
reliably and resolves vague follow-ups better than an 8B, at roughly twice the
latency (~20 s vs ~10 s per answer here). For a faster, less accurate
assistant:

```bash
ollama pull llama3.1:8b
```

then run with `CX_CHAT_MODEL=llama3.1:8b`.

Then install the package, pull the knowledge base and build the index:

```bash
uv venv --python 3.11 && uv pip install -e . && .venv/bin/cx sync && .venv/bin/cx ingest
```

`cx sync` needs AWS credentials with read access to the source bucket. The
articles themselves are not in this repository — one of them is a league
leaderboard listing around a thousand real player display names and payout
amounts, which does not belong in version control.

## Use

Start the chatbot and open <http://127.0.0.1:8000>:

```bash
.venv/bin/cx serve
```

The UI streams the answer as it is generated, shows which help-center articles
were retrieved, and renders each `[1]` as a chip that opens the source it came
from. When a question falls outside the KB the reply is styled as a handoff
rather than an answer, so it reads as an escalation instead of a failure.

Ask one question from the terminal instead:

```bash
.venv/bin/cx ask "How long does a withdrawal take?"
```

Or a terminal conversation (`reset` clears it):

```bash
.venv/bin/cx chat
```

Inspect retrieval without generating an answer — useful when an answer looks
wrong and you need to know whether retrieval or the model is at fault:

```bash
.venv/bin/cx search "can I deposit with a gift card" -k 5
```

Refresh the knowledge base from S3, then re-index:

```bash
.venv/bin/cx sync && .venv/bin/cx ingest
```

## Exposing it publicly

```bash
.venv/bin/cx tunnel
```

Starts the server, opens an ngrok tunnel, and prints the public URL with
generated HTTP Basic credentials. Requires `ngrok` on PATH with an authtoken
configured.

Two things are enforced rather than left to the operator, because the failure
mode is silent:

**Auth is not optional through the tunnel.** The server runs open on localhost,
which is fine, but `cx tunnel` always sets credentials — an unauthenticated
public endpoint is an invitation to run inference on someone else's GPU, and the
bill for that is your laptop's battery and everyone's latency.

**Leaderboard articles never reach the index.** The export includes a league
results table listing roughly a thousand player display names against their
payout amounts. It answers no support question, and before this it was
retrievable at rank 2 for "all star league winners prizes" — which would have
served real players' names and winnings to anyone with the URL. `cx ingest` now
detects that shape (25+ payout rows) and withholds it, naming what it withheld.

## How it works

| Stage | What happens |
| --- | --- |
| `corpus.py` | Loads markdown, repairs the export, merges near-duplicates, chunks |
| `embed.py` | Embeds via `nomic-embed-text` with its document/query task prefixes |
| `store.py` | Dense cosine + BM25, fused with reciprocal rank fusion |
| `answer.py` | Condenses follow-ups, applies policy, grounded prompt → model |
| `matchid.py` | Detects match-level issues, extracts the Match ID |
| `tickets.py` | Opens and stores support tickets |
| `web.py` | FastAPI, NDJSON streaming, in-memory sessions |
| `eval.py` | Retrieval recall / MRR |

### Follow-up questions

"How long does that take?" is meaningless to a retriever, so before searching,
the latest message is rewritten into a standalone query using the conversation
so far — that one becomes `withdrawal processing time`. The UI shows the
rewritten query whenever it differs from what you typed.

Rewriting is itself a place to hallucinate, and three guards sit around it.

*The rewrite must not drop what the player said.* An early condense prompt
listed product nouns as examples and the model began injecting them: "Can I pay
with Dogecoin?" came back as a question about Ticketz and Z Coins. The prompt
now names no example products, and `_distinctive_terms` checks the rewrite
against the corpus's own BM25 statistics — if the player used a term the corpus
considers rare, or has never seen at all, and the rewrite dropped it, the raw
question is used instead. Common words may disappear freely, so genuine pronoun
resolution still works.

*The rewrite must not invent what the player didn't say.* A distinctive term in
the rewrite that appears nowhere in the conversation means the model changed the
subject, and the rewrite is discarded.

*Some messages cannot be searched at all.* "I don't have", "yes", "it still
didn't work" contain nothing but function words. Passed through untouched they
retrieve on stopword coincidence — "I don't have" once matched an article about
forgetting a password, on the strength of "don't". These are now anchored to the
last substantive thing the player asked, so "I don't have" after a question
about Match IDs searches for the Match ID topic. This is why the stopword list
includes contractions, and why `tokenize` normalises typographic apostrophes:
`don’t` and `don't` have to reach the same place.

### Vague questions

"help", "problem", "not working" cannot be answered from a knowledge base, and
answering them anyway produces confident nonsense. So the assistant asks once
for detail, and if the next message is still vague it stops guessing and hands
off to a human, opening a ticket.

Vagueness is measured, not guessed. The signal is the top dense cosine of the
retrieved set, which unlike the fused RRF score is calibrated across queries. On
this corpus real questions bottom out at **0.742** ("my game crashed") and vague
ones top out at **0.738** ("game"), so the floor sits at 0.72 — deliberately
just under the clear band, because being asked to rephrase a fair question is
more annoying than a slightly loose answer. A query with a single content word
needs 0.75 to count, separating "What are Ticketz?" (0.766) from "game" (0.738).

Two details matter:

- Vagueness is judged on the **condensed** query, not the raw message. "I don't
  have" has already been anchored to something specific by then, so it is not
  mistaken for a vague request.
- Collecting a Match ID takes precedence. While that is in flight a short reply
  is an answer to the assistant's own question, not a vague request, and must
  not trigger clarification.

The handoff message is fixed text rather than generated. Escalating is a policy
outcome, and when the model wrote it, it kept echoing its own previous "could
you tell me more?" turn instead of closing the loop.

### Match issues and tickets

Support cannot investigate a crash, an abort, a suspect opponent or a missing
prize without the Match ID, so the assistant collects it. Whether to ask is
decided in code (`matchid.py`), not by the model — policy should not vary with
sampling. The rules:

1. The thread mentions a match-level problem → ask for the Match ID, once.
2. The player says they don't have it → never ask again; explain where to find
   it in the app instead.
3. The player gives it → a ticket opens automatically, and the reply quotes both
   the Match ID and the ticket reference.

Tickets land in `tickets/tickets.jsonl` with the category, the Match ID, the
player's own words, the transcript and the articles consulted:

```bash
.venv/bin/cx tickets -v
```

The ticket reference is reserved before generation and written after, so a
failed generation leaves no orphan ticket, and the model can still quote the
reference in its reply.

**These tickets are local files, not Zendesk.** Filing into a real helpdesk
sends player data off the machine and needs credentials, which is an operator
decision rather than a default. `Ticket.to_zendesk_payload()` already shapes the
record the way the Zendesk Tickets API expects, so wiring it up is one
authenticated HTTP call.

Three more decisions are worth knowing about, because they were driven by what
is actually in this KB:

**The export is damaged.** Zendesk flattened inline bold into standalone lines,
so a sentence arrives as fragments (`the winnings` / `on top` / `100% real cash`
/ `.`). `reflow()` rejoins them while leaving lists and headings alone. Without
it the model reads shredded text.

**A quarter of the KB is duplicated.** Skillz and Skillz Arena publish parallel
help centers. 26 article pairs are ~97% identical; left alone they crowd each
other out of the top-5 and spend context restating one answer. Ingest merges
them, keeping the longer copy and recording the other id as an alias so
citations still resolve in either help center. `cx ingest --show-merges` lists
every merge.

All 119 articles stay searchable — merging is a retrieval optimisation, not a
smaller knowledge base. A merged twin is reachable through its alias, and
citations print both ids. The "93 unique" figure is only the number of distinct
things retrieval can return.

**Retrieval is hybrid, not pure vector.** Support questions turn on exact
product nouns — Ticketz, Z Coins, Gemz, Bonus Cash — which embeddings blur
together because they occupy near-identical semantic space. BM25 keeps them
distinct; RRF combines the two rankings without needing their scores to be
comparable.

## Evaluating changes

```bash
.venv/bin/cx eval
```

This asks each article's own title and checks the article comes back. It is an
easy benchmark and should stay near-perfect — treat a drop as a signal that
chunking or the embedding prefix broke, not as evidence of production quality.

Current: **recall@1 0.839, recall@5 1.000, MRR 0.919** over 93 articles.

For real signal, write an eval set from actual ticket logs and pass it with
`-f`. The format is a JSON list:

```json
[
  {"question": "how long till my check arrives", "article_id": "204372895"},
  {"question": "can i use a visa gift card", "article_id": "115001140043"}
]
```

## Known limitation: the two products are conflated

Eleven article pairs share a title but give *different* answers, because Skillz
and Skillz Arena are different products with different currencies. "Are Skillz
games free?" answers "Z" for classic and "Gemz, cash games from $0.60" for
Arena. The assistant has no way to tell which product the player is on, so it
retrieves both and may blend them — asking about withdrawal timing today returns
both the 1–2 week and the 4–6 week figure.

The fix is a product facet: tag each article at ingest, then filter retrieval by
the player's product. That needs a mapping from article id to product, which has
to come from Zendesk (the article text alone is not a reliable signal). Until
then, treat cross-product answers as needing human review.

## Configuration

Every value in `src/cx/config.py` reads from an environment variable. To try a
different model:

```bash
CX_CHAT_MODEL=qwen2.5:7b .venv/bin/cx ask "how do trophies work"
```

Changing `CX_EMBED_MODEL` requires re-running `cx ingest`, since the stored
vectors come from the old model.

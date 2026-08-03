"""Command line entry point: sync, ingest, search, ask, chat, eval."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import config
from .answer import Conversation, answer
from .corpus import build_chunks
from .embed import OllamaError
from .eval import evaluate, load_cases, title_cases
from .store import Index

S3_URI = "s3://skillz-support-kb-637423462092/articles/"


def cmd_sync(args: argparse.Namespace) -> int:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "sync", args.uri, str(config.DATA_DIR)]
    if args.delete:
        # Opt-in: this removes local files that are gone from the bucket.
        cmd.append("--delete")
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return result.returncode
    print("\nRe-run `cx ingest` to pick up the changes.")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    print(f"Reading {config.DATA_DIR} ...")
    chunks, articles, merges, withheld = build_chunks(config.DATA_DIR)
    if not chunks:
        print(f"No markdown found in {config.DATA_DIR}.", file=sys.stderr)
        return 1

    for article in withheld:
        print(
            f"  withheld {article.article_id}: {article.title[:52]}"
            f"\n    (league results table — player names and payouts, not support content)"
        )
    print(f"  {len(articles) + len(merges) + len(withheld)} articles read")
    print(f"  {len(merges)} near-duplicates merged -> {len(articles)} unique")
    print(f"  {len(chunks)} chunks")
    print(f"Embedding with {config.EMBED_MODEL} ...")

    meta = {
        "embed_model": config.EMBED_MODEL,
        "chat_model": config.CHAT_MODEL,
        # Every source article stays searchable: a merged twin is reachable
        # through its alias, so `source_articles` is what the KB actually covers.
        "source_articles": len(articles) + len(merges),
        "withheld_articles": [
            {"id": a.article_id, "title": a.title} for a in withheld
        ],
        "articles": len(articles),
        "chunks": len(chunks),
        "merged_duplicates": [{"kept": k, "dropped": d} for k, d in merges],
    }
    index = Index.build(chunks, meta)
    index.save(config.INDEX_DIR)

    dims = index.embeddings.shape[1]
    print(f"Wrote {config.INDEX_DIR} ({len(chunks)} x {dims} embeddings)")
    if merges and args.show_merges:
        print("\nMerged duplicates (kept <- dropped):")
        for kept, dropped in merges:
            print(f"  {kept} <- {dropped}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    index = Index.load(config.INDEX_DIR)
    hits = index.search(args.question, top_k=args.top_k)
    for i, hit in enumerate(hits, 1):
        d = "-" if hit.dense_rank is None else str(hit.dense_rank + 1)
        l = "-" if hit.lexical_rank is None else str(hit.lexical_rank + 1)
        print(f"[{i}] {hit.chunk.title}")
        print(f"    article {hit.chunk.article_id}  rrf={hit.score:.4f}  dense=#{d} bm25=#{l}")
        if hit.chunk.aliases:
            print(f"    also published as: {', '.join(hit.chunk.aliases)}")
    return 0


def _print_sources(hits) -> None:
    if not hits:
        return
    print("\n\nSources:")
    for i, hit in enumerate(hits, 1):
        ids = ", ".join([hit.chunk.article_id, *hit.chunk.aliases])
        print(f"  [{i}] {hit.chunk.title}  (article {ids})")


def cmd_ask(args: argparse.Namespace) -> int:
    index = Index.load(config.INDEX_DIR)
    result = answer(
        index,
        args.question,
        top_k=args.top_k,
        temperature=args.temperature,
        stream_to=sys.stdout,
    )
    if args.show_sources:
        _print_sources(result.hits)
    print()
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    index = Index.load(config.INDEX_DIR)
    conversation = Conversation()
    print(f"Skillz support assistant ({config.CHAT_MODEL}). Ctrl-C or 'exit' to quit.")
    print("'reset' starts a new conversation.\n")
    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            return 0
        if question.lower() == "reset":
            conversation = Conversation()
            print("\n(new conversation)\n")
            continue

        print("\nbot > ", end="")
        result = answer(
            index,
            question,
            conversation=conversation,
            top_k=args.top_k,
            temperature=args.temperature,
            stream_to=sys.stdout,
        )
        if args.show_sources:
            _print_sources(result.hits)
        if result.ticket:
            t = result.ticket
            print(f"\n  ── ticket {t.ticket_id} created ──")
            print(f"     issue    : {t.label}")
            print(f"     match ID : {t.match_id}")
            print(f"     reported : {t.summary}")
        elif result.match_id:
            print(f"\n  [Match ID on file: {result.match_id}]")
        print("\n")


def cmd_tunnel(args: argparse.Namespace) -> int:
    """Serve the UI and expose it through an ngrok tunnel."""
    import secrets
    import shutil
    import time
    import urllib.request

    if not shutil.which("ngrok"):
        print("ngrok is not installed. `brew install ngrok`", file=sys.stderr)
        return 1

    Index.load(config.INDEX_DIR)

    # Never expose this without credentials. Generating them is friendlier than
    # refusing, and a generated password beats whatever would be typed here.
    user = args.user or os.environ.get("CX_AUTH_USER") or "skillz"
    password = os.environ.get("CX_AUTH_PASS") or secrets.token_urlsafe(12)

    env = {**os.environ, "CX_AUTH_USER": user, "CX_AUTH_PASS": password}
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "cx.web:app",
         "--host", "127.0.0.1", "--port", str(args.port)],
        env=env,
    )
    tunnel = subprocess.Popen(
        ["ngrok", "http", str(args.port), "--log", "stdout"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    public_url = None
    try:
        # ngrok publishes the assigned URL on its local API once connected.
        for _ in range(40):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(
                    "http://127.0.0.1:4040/api/tunnels", timeout=2
                ) as r:
                    tunnels = json.load(r).get("tunnels", [])
                public_url = next(
                    (t["public_url"] for t in tunnels if t["public_url"].startswith("https")),
                    None,
                )
                if public_url:
                    break
            except Exception:
                continue

        if not public_url:
            print("Could not read the tunnel URL from ngrok.", file=sys.stderr)
            return 1

        # flush=True: stdout is block-buffered when redirected, which otherwise
        # hides the credentials until the process exits — exactly the moment
        # they stop being useful.
        print(f"\n  public URL : {public_url}", flush=True)
        print(f"  username   : {user}", flush=True)
        print(f"  password   : {password}", flush=True)
        print(
            "\n  Anyone with the URL and these credentials can use your local"
            "\n  model. Ctrl-C closes the tunnel.\n",
            flush=True,
        )
        server.wait()
    except KeyboardInterrupt:
        print("\nclosing tunnel")
    finally:
        for proc in (tunnel, server):
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


def cmd_tickets(args: argparse.Namespace) -> int:
    from . import tickets as tickets_mod

    all_tickets = tickets_mod.load_all()
    if not all_tickets:
        print(f"No tickets yet ({tickets_mod.TICKETS_FILE}).")
        return 0

    for t in reversed(all_tickets[-args.limit :]):
        print(f"{t.ticket_id}  {t.created_at}  [{t.status}]")
        print(f"  issue    : {t.label}")
        print(f"  match ID : {t.match_id or '-'}")
        print(f"  reported : {t.summary}")
        if args.verbose:
            for m in t.transcript:
                print(f"      {m['role']:9} {m['content']}")
        print()
    print(f"{len(all_tickets)} ticket(s) in {tickets_mod.TICKETS_FILE}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    # Fail fast with a useful message rather than 500ing on the first question.
    Index.load(config.INDEX_DIR)
    print(f"Chat UI on http://{args.host}:{args.port}")
    uvicorn.run("cx.web:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    index = Index.load(config.INDEX_DIR)
    cases = load_cases(Path(args.file)) if args.file else title_cases(index)
    label = args.file or "article titles (smoke test)"
    print(f"Evaluating retrieval on {len(cases)} cases from {label} ...\n")

    report = evaluate(index, cases, top_k=args.top_k)
    k = args.top_k
    print(f"  recall@1   {report['recall@1']:.3f}")
    print(f"  recall@{k}   {report[f'recall@{k}']:.3f}")
    print(f"  MRR        {report['mrr']:.3f}")

    if report["misses"]:
        print(f"\n  {len(report['misses'])} misses:")
        for miss in report["misses"][: args.show_misses]:
            print(f"    - {miss['question']}")
            print(f"        expected {miss['expected']}, got {miss['got']}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cx", description="Local RAG assistant over the Skillz support KB."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sync", help="pull the latest articles from S3")
    p.add_argument("--uri", default=S3_URI)
    p.add_argument(
        "--delete",
        action="store_true",
        help="also remove local articles that no longer exist in the bucket",
    )
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("ingest", help="chunk, embed and index the articles")
    p.add_argument("--show-merges", action="store_true", help="list merged duplicates")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="show what retrieval returns, no generation")
    p.add_argument("question")
    p.add_argument("-k", "--top-k", type=int, default=config.TOP_K)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="answer a single question")
    p.add_argument("question")
    p.add_argument("-k", "--top-k", type=int, default=config.TOP_K)
    p.add_argument("-t", "--temperature", type=float, default=0.1)
    p.add_argument("--no-sources", dest="show_sources", action="store_false")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive session")
    p.add_argument("-k", "--top-k", type=int, default=config.TOP_K)
    p.add_argument("-t", "--temperature", type=float, default=0.1)
    p.add_argument("--no-sources", dest="show_sources", action="store_false")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("tunnel", help="expose the chat UI publicly via ngrok")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--user", help="basic-auth username (default: skillz)")
    p.set_defaults(func=cmd_tunnel)

    p = sub.add_parser("tickets", help="list tickets opened by the assistant")
    p.add_argument("-n", "--limit", type=int, default=20)
    p.add_argument("-v", "--verbose", action="store_true", help="include transcripts")
    p.set_defaults(func=cmd_tickets)

    p = sub.add_parser("serve", help="run the chat UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="auto-reload on edits")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="measure retrieval quality")
    p.add_argument("-f", "--file", help="JSON eval set; defaults to title smoke test")
    p.add_argument("-k", "--top-k", type=int, default=5)
    p.add_argument("--json", help="write the full report here")
    p.add_argument("--show-misses", type=int, default=15)
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OllamaError, FileNotFoundError) as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

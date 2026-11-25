"""
Compare chunking strategies end to end, one strategy at a time, saving as we go.

    chunk -> embed (FAISS) -> retrieve top-k -> answer -> RAGAS

Each strategy is a self-contained unit. Its results are written to disk the
moment they exist, so a network error in one strategy never costs you the others
and never leaves silent empty rows in a CSV.

Layout written under --out (default: results/):

  strategies/<name>/chunks.json         the chunks, for inspection
  strategies/<name>/answers.json        retrieved contexts + answers, saved after EVERY question
  strategies/<name>/ragas_scores.json   RAGAS scores per question, saved after every RAGAS attempt
  strategies/<name>/per_question.csv    final per-question table for this strategy
  strategies/<name>/summary.json        final summary + status (complete | incomplete | failed)
  summary.csv, per_question.csv         combined views, rebuilt from the folders after each strategy

Rerun behaviour (just run the same command again):
  * status == complete and same inputs  -> skipped entirely
  * failed or incomplete                -> resumes: reuses saved answers, only recomputes
                                           questions that are missing or have NaN RAGAS scores
  * --force                             -> ignores every cache and skip

Usage:
  python compare_chunking_v2.py --retrieval-only            # cheapest: embeddings only
  python compare_chunking_v2.py --skip-ragas --limit 10     # + generation
  python compare_chunking_v2.py                             # everything
  python compare_chunking_v2.py --strategies header_only    # just one strategy
  python compare_chunking_v2.py --force --strategies fixed_500
"""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
EMBED_MODEL = "text-embedding-3-small"
MODE_RANK = {"retrieval_only": 0, "generation": 1, "full": 2}
TEXT_COLS = {"user_input", "retrieved_contexts", "response", "reference"}


# --------------------------------------------------------------------------
# Chunking strategies. Each takes raw markdown and returns list[Document].
# --------------------------------------------------------------------------
def _doc(text: str, section: str = "root") -> Document:
    return Document(page_content=text, metadata={"section_path": section})


def _sections(text: str) -> list[Document]:
    return MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS).split_text(text)


def _path(section: Document) -> str:
    return " > ".join(section.metadata.values()) or "root"


def fixed_500(text: str) -> list[Document]:
    """Naive baseline: cut every 500 characters, ignoring structure."""
    splitter = CharacterTextSplitter(separator="", chunk_size=500, chunk_overlap=0)
    return [_doc(c) for c in splitter.split_text(text)]


def recursive_800(text: str) -> list[Document]:
    """Recursive splitter on the whole document, no header awareness."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120, separators=SEPARATORS
    )
    return [_doc(c) for c in splitter.split_text(text)]


def recursive_300(text: str) -> list[Document]:
    """Small chunks: precise matches, but facts may lose their context."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50, separators=SEPARATORS
    )
    return [_doc(c) for c in splitter.split_text(text)]


def header_only(text: str) -> list[Document]:
    """One chunk per markdown section, however long or short."""
    return [_doc(s.page_content, _path(s)) for s in _sections(text)]


def header_recursive(text: str, size: int = 800, overlap: int = 120) -> list[Document]:
    """Same strategy as your chain.py: split by header, then sub-split long sections."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=SEPARATORS
    )
    out = []
    for s in _sections(text):
        for piece in splitter.split_text(s.page_content):
            out.append(_doc(piece, _path(s)))
    return out


def header_recursive_prefixed(text: str) -> list[Document]:
    """Contextual chunks: the section path is prepended so it is embedded too."""
    docs = header_recursive(text, size=600, overlap=100)
    for d in docs:
        d.page_content = f"[{d.metadata['section_path']}]\n{d.page_content}"
    return docs


def semantic(text: str) -> list[Document]:
    """Split where embedding similarity between sentences drops (needs langchain_experimental)."""
    from langchain_experimental.text_splitter import SemanticChunker

    chunker = SemanticChunker(
        OpenAIEmbeddings(model=EMBED_MODEL, max_retries=6),
        breakpoint_threshold_type="percentile",
    )
    return [_doc(c) for c in chunker.split_text(text)]


STRATEGIES = {
    "fixed_500": fixed_500,
    "recursive_800": recursive_800,
    "recursive_300": recursive_300,
    "header_only": header_only,
    "header_recursive_800": header_recursive,
    "header_recursive_prefixed_600": header_recursive_prefixed,
}

# --------------------------------------------------------------------------
# Answer chain and cheap deterministic metrics
# --------------------------------------------------------------------------
PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the context below. "
    "If the context doesn't contain the answer, say you don't have enough information.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def norm(s: str) -> str:
    return " ".join(s.split())


def evidence_recall(contexts: list[str], evidence: list[str]) -> tuple[float, bool]:
    """Fraction of required evidence strings present in the retrieved text."""
    blob = norm(" ".join(contexts))
    found = [norm(e) in blob for e in evidence]
    return sum(found) / len(found), any(found)


def abstained(answer: str) -> bool:
    a = answer.lower()
    return "don't have enough information" in a or "do not have enough information" in a


# --------------------------------------------------------------------------
# Small file helpers (atomic, so a crash never leaves a half-written file)
# --------------------------------------------------------------------------
def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None  # treat a corrupt file as missing


def strategy_dir(out_dir: Path, name: str) -> Path:
    d = out_dir / "strategies" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def requested_mode(args) -> str:
    if args.retrieval_only:
        return "retrieval_only"
    return "generation" if args.skip_ragas else "full"


def fingerprint(text: str, items: list[dict], args) -> str:
    """Changes whenever anything that affects results changes, so stale caches are never reused."""
    payload = {
        "doc": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "items": hashlib.sha256(json.dumps(items, sort_keys=True).encode("utf-8")).hexdigest(),
        "k": args.k,
        "embed": EMBED_MODEL,
        "gen_model": args.gen_model,
        "judge_model": args.judge_model,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def mean_or_nan(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(s.mean()), 3) if len(s) else float("nan")


# --------------------------------------------------------------------------
# Stage 1: retrieve + answer. Saved after every question, resumes on rerun.
# --------------------------------------------------------------------------
def stage_retrieve_and_answer(chunks, items, args, embeddings, llm, d: Path, fp: str, mode: str):
    cache_path = d / "answers.json"
    cache = {}
    if not args.force:
        saved = load_json(cache_path)
        if saved and saved.get("fingerprint") == fp:
            cache = {r["id"]: r for r in saved["rows"]}
            if cache:
                print(f"  resuming: {len(cache)} questions already saved")

    store = FAISS.from_documents(chunks, embeddings)
    retriever = store.as_retriever(search_kwargs={"k": args.k})

    rows = []
    for item in items:
        row = cache.get(item["id"])
        if row is None:
            t0 = time.time()
            contexts = [doc.page_content for doc in retriever.invoke(item["question"])]
            row = {
                "id": item["id"], "type": item["type"], "answerable": item["answerable"],
                "question": item["question"], "ground_truth": item["ground_truth"],
                "contexts": contexts,
                "retrieval_ms": round((time.time() - t0) * 1000, 1),
            }
            if item["evidence"]:
                row["evidence_recall"], row["hit"] = evidence_recall(contexts, item["evidence"])

        if mode != "retrieval_only" and "answer" not in row:
            row["answer"] = (PROMPT | llm).invoke(
                {"context": "\n\n".join(row["contexts"]), "question": row["question"]}
            ).content
            row["abstained"] = abstained(row["answer"])

        rows.append(row)
        atomic_write(cache_path, json.dumps({"fingerprint": fp, "rows": rows}, indent=2))
    return rows


# --------------------------------------------------------------------------
# Stage 2: RAGAS. Per-question scores saved after every attempt; only NaN rows are retried.
# --------------------------------------------------------------------------
def run_ragas(pending_rows: list[dict], args) -> pd.DataFrame:
    """Score the given rows once. Returns metric columns only, same order as the input."""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(model=args.judge_model, temperature=0, max_retries=6, timeout=60)
    )
    judge_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBED_MODEL, max_retries=6))

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
                reference=r["ground_truth"],
            )
            for r in pending_rows
        ]
    )
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=judge_llm),
            ResponseRelevancy(llm=judge_llm, embeddings=judge_emb),
            LLMContextPrecisionWithReference(llm=judge_llm),
            LLMContextRecall(llm=judge_llm),
        ],
        # fewer parallel workers and longer timeouts: the usual cause of NaN rows is rate limits
        run_config=RunConfig(timeout=args.ragas_timeout, max_retries=10, max_workers=args.ragas_workers),
        raise_exceptions=False,
    )
    df = result.to_pandas()
    return df[[c for c in df.columns if c not in TEXT_COLS]]


def _is_complete(scores: dict | None) -> bool:
    return bool(scores) and all(v is not None for v in scores.values())


def stage_ragas(rows: list[dict], args, d: Path, fp: str):
    """Returns (scores_by_id, number_of_rows_still_incomplete)."""
    path = d / "ragas_scores.json"
    saved = {}
    if not args.force:
        blob = load_json(path)
        if blob and blob.get("fingerprint") == fp:
            saved = blob["scores"]

    todo = [r for r in rows if r["answerable"] and "answer" in r]
    pending = [r for r in todo if not _is_complete(saved.get(r["id"]))]
    if len(todo) - len(pending):
        print(f"  RAGAS: {len(todo) - len(pending)}/{len(todo)} questions already scored")

    for attempt in range(1, args.ragas_retries + 2):  # first try + N retries
        if not pending:
            break
        print(f"  RAGAS attempt {attempt}: scoring {len(pending)} questions")
        try:
            df = run_ragas(pending, args)
        except Exception as e:  # network, rate limit, etc.: keep what we have and try again
            print(f"  RAGAS attempt {attempt} raised {type(e).__name__}: {e}")
            time.sleep(min(5 * attempt, 30))
            continue

        for r, (_, srow) in zip(pending, df.iterrows()):
            new = {k: (None if pd.isna(v) else float(v)) for k, v in srow.items()}
            old = saved.get(r["id"], {})
            # a retry may only fill gaps, never overwrite a good value with a missing one
            saved[r["id"]] = {k: new[k] if new[k] is not None else old.get(k) for k in new}
        atomic_write(path, json.dumps({"fingerprint": fp, "scores": saved}, indent=2))
        pending = [r for r in todo if not _is_complete(saved.get(r["id"]))]
        if pending:
            print(f"  RAGAS: {len(pending)} questions still have NaN metrics")
            time.sleep(min(5 * attempt, 30))

    return saved, len(pending)


# --------------------------------------------------------------------------
# One strategy, end to end
# --------------------------------------------------------------------------
def should_skip(prev: dict | None, fp: str, mode: str, force: bool) -> bool:
    if force or not prev:
        return False
    return (
        prev.get("fingerprint") == fp
        and prev.get("status") == "complete"
        and MODE_RANK.get(prev.get("mode"), -1) >= MODE_RANK[mode]
    )


def run_strategy(name, chunker, text, items, args, embeddings, llm, out_dir: Path, fp: str):
    d = strategy_dir(out_dir, name)
    mode = requested_mode(args)

    prev = load_json(d / "summary.json")
    if should_skip(prev, fp, mode, args.force):
        print(f"  already complete, skipping (use --force to redo)")
        return prev

    chunks = chunker(text)
    lengths = [len(c.page_content) for c in chunks]
    atomic_write(
        d / "chunks.json",
        json.dumps(
            [{"i": i, "section": c.metadata.get("section_path"),
              "chars": len(c.page_content), "text": c.page_content}
             for i, c in enumerate(chunks)],
            indent=2,
        ),
    )
    if args.show_chunks:
        print(f"\n===== {name}: first {args.show_chunks} chunks =====")
        for i, c in enumerate(chunks[: args.show_chunks]):
            print(f"--- chunk {i} ({len(c.page_content)} chars, {c.metadata['section_path']}) ---")
            print(c.page_content)

    rows = stage_retrieve_and_answer(chunks, items, args, embeddings, llm, d, fp, mode)

    scores, n_incomplete = {}, 0
    if mode == "full":
        scores, n_incomplete = stage_ragas(rows, args, d, fp)

    # ---- assemble the per-question table (no contexts: those live in answers.json)
    records = []
    for r in rows:
        rec = {"strategy": name}
        rec.update({k: v for k, v in r.items()
                    if k not in {"contexts", "question", "ground_truth"}})
        rec.update(scores.get(r["id"], {}))
        records.append(rec)
    per_q = pd.DataFrame(records)
    write_csv(d / "per_question.csv", per_q)

    # ---- summary
    ans = per_q[per_q["answerable"]]
    summary = {
        "strategy": name,
        "status": "complete" if n_incomplete == 0 else "incomplete",
        "mode": mode,
        "fingerprint": fp,
        "n_chunks": len(chunks),
        "avg_chars": round(sum(lengths) / len(lengths)),
        "max_chars": max(lengths),
        "retrieval_ms": mean_or_nan(per_q["retrieval_ms"]),
        "evidence_recall": mean_or_nan(ans["evidence_recall"]),
        "hit_rate": mean_or_nan(ans["hit"].dropna().astype(float)),
    }
    if mode != "retrieval_only":
        unans = per_q[~per_q["answerable"]]
        summary["abstain_rate_unanswerable"] = mean_or_nan(unans["abstained"].astype(float))
        summary["false_refusal_rate_answerable"] = mean_or_nan(ans["abstained"].astype(float))
    if mode == "full":
        metric_cols = sorted({k for s in scores.values() for k in s})
        for col in metric_cols:
            summary[col] = mean_or_nan(per_q[col])
        n_expected = int(per_q["answerable"].sum())
        summary["ragas_rows_ok"] = n_expected - n_incomplete
        summary["ragas_rows_expected"] = n_expected

    atomic_write(d / "summary.json", json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------
# Combined views, rebuilt from whatever is on disk
# --------------------------------------------------------------------------
def rebuild_combined(out_dir: Path, order: list[str]) -> pd.DataFrame:
    summaries, details = [], []
    root = out_dir / "strategies"
    names = [n for n in order if (root / n).exists()]
    names += sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in order) if root.exists() else []
    for name in names:
        s = load_json(root / name / "summary.json")
        if s:
            summaries.append(s)
        pq = root / name / "per_question.csv"
        if pq.exists():
            details.append(pd.read_csv(pq))
    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        write_csv(out_dir / "summary.csv", summary_df)
    if details:
        write_csv(out_dir / "per_question.csv", pd.concat(details, ignore_index=True))
    return summary_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--doc", default="quillstream_handbook.md")
    p.add_argument("--eval", default="eval_set.json")
    p.add_argument("--out", default="results")
    p.add_argument("--k", type=int, default=4, help="chunks retrieved per question")
    p.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    p.add_argument("--semantic", action="store_true", help="also run SemanticChunker")
    p.add_argument("--limit", type=int, default=0, help="only use the first N eval questions")
    p.add_argument("--retrieval-only", action="store_true", help="no generation, no RAGAS")
    p.add_argument("--skip-ragas", action="store_true", help="generate answers but skip RAGAS")
    p.add_argument("--force", action="store_true", help="ignore skip logic and all caches")
    p.add_argument("--show-chunks", type=int, default=0, help="print the first N chunks per strategy")
    p.add_argument("--gen-model", default="gpt-4o-mini")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--ragas-retries", type=int, default=3, help="extra attempts for NaN or failed RAGAS rows")
    p.add_argument("--ragas-workers", type=int, default=4, help="parallel RAGAS calls (lower = fewer rate limits)")
    p.add_argument("--ragas-timeout", type=int, default=180, help="seconds per RAGAS call")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    text = Path(args.doc).read_text(encoding="utf-8")
    items = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]
    fp = fingerprint(text, items, args)

    strategies = {n: STRATEGIES[n] for n in args.strategies}
    if args.semantic:
        strategies["semantic"] = semantic

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, max_retries=6)
    llm = None if args.retrieval_only else ChatOpenAI(
        model=args.gen_model, temperature=0, max_retries=6, timeout=60
    )

    order = list(STRATEGIES) + (["semantic"] if args.semantic else [])
    problems = []
    for name, chunker in strategies.items():
        print(f"\n>>> strategy: {name}")
        try:
            summary = run_strategy(name, chunker, text, items, args, embeddings, llm, out_dir, fp)
        except Exception as e:
            traceback.print_exc()
            summary = {"strategy": name, "status": "failed", "mode": requested_mode(args),
                       "fingerprint": fp, "error": f"{type(e).__name__}: {e}"}
            atomic_write(strategy_dir(out_dir, name) / "summary.json", json.dumps(summary, indent=2))
        if summary.get("status") != "complete":
            problems.append(name)
        rebuild_combined(out_dir, order)  # CSVs on disk are valid after every strategy

    summary_df = rebuild_combined(out_dir, order)
    print("\n================ SUMMARY ================")
    if summary_df.empty:
        print("(no results)")
    else:
        drop = [c for c in ("fingerprint", "error") if c in summary_df.columns]
        print(summary_df.drop(columns=drop).to_string(index=False))
    print(f"\nCombined CSVs: {out_dir / 'summary.csv'}, {out_dir / 'per_question.csv'}")
    print(f"Per-strategy data: {out_dir / 'strategies' / '<name>'}")

    if problems:
        print(f"\nNOT COMPLETE: {', '.join(problems)}")
        print("Rerun only these (saved answers and good RAGAS scores are reused):")
        print(f"  python {Path(sys.argv[0]).name} --strategies {' '.join(problems)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
Retrieval experiments (steps 3, 4, 5): vary k, add hybrid search, add a reranker.

The chunker is FIXED (default: header_recursive_800). Only the retrieval setup changes,
so any score difference comes from retrieval alone. Embeddings, prompt, generator and
judge are the same for every variant.

Variants (edit VARIANTS below to add your own):
  vec_k4              vector search, top 4              <- baseline, same as before
  vec_k8              vector search, top 8              <- step 3: does a bigger k fix multi-chunk questions?
  hybrid_k4           BM25 + vector, fused, top 4       <- step 4: do exact tokens (ERR_4221) improve?
  hybrid_k8           BM25 + vector, fused, top 8
  vec_rerank_top4     vector top POOL -> reranker -> 4  <- step 5: isolates the reranker
  hybrid_rerank_top4  hybrid top POOL -> reranker -> 4  <- step 5: the full chain.py design

Extra metrics this script adds:
  pool_evidence_recall   evidence recall of the candidate list BEFORE the final cut/rerank.
                         final evidence_recall << pool_evidence_recall means the cut or the
                         reranker threw good chunks away.
  top_score_*            highest reranker score per question (rerank variants only), split by
                         answerable vs unanswerable: tells you whether a confidence floor works.

Same save-as-you-go and resume behaviour as compare_chunking_v2.py:
  results_retrieval/variants/<name>/{answers.json, ragas_scores.json, per_question.csv, summary.json}
  results_retrieval/{summary.csv, per_question.csv, by_type.csv}

Extra installs:  pip install rank_bm25 sentence-transformers

Usage:
  python compare_retrieval.py --retrieval-only          # cheap: no generation, no RAGAS
  python compare_retrieval.py --skip-ragas              # + answers and refusal rates
  python compare_retrieval.py                           # everything
  python compare_retrieval.py --variants vec_k4 hybrid_k4
"""
import argparse
import hashlib
import json
import re
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from compare_chunkings import (
    EMBED_MODEL,
    PROMPT,
    STRATEGIES,
    abstained,
    atomic_write,
    evidence_recall,
    load_json,
    mean_or_nan,
    requested_mode,
    should_skip,
    stage_ragas,
    write_csv,
)

VARIANTS = {
    "vec_k4": {"search": "vector", "k": 4},
    "vec_k8": {"search": "vector", "k": 8},
    "hybrid_k4": {"search": "hybrid", "k": 4},
    "hybrid_k8": {"search": "hybrid", "k": 8},
    "vec_rerank_top4": {"search": "vector", "rerank": True, "k": 4},
    "hybrid_rerank_top4": {"search": "hybrid", "rerank": True, "k": 4},
}


# --------------------------------------------------------------------------
# Retrieval building blocks
# --------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. \\w keeps underscores, so ERR_4221 and qs_test_ stay single tokens."""
    return re.findall(r"\w+", text.lower())


def rrf(ranked_lists: list[list[int]], weights: list[float], c: int = 60) -> list[int]:
    """Weighted Reciprocal Rank Fusion: score(d) = sum_i weight_i / (c + rank_i(d)).
    This is what LangChain's EnsembleRetriever does, written out so you can see it."""
    scores: dict[int, float] = {}
    for ranking, w in zip(ranked_lists, weights):
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + w / (c + rank)
    return sorted(scores, key=lambda i: -scores[i])


class CrossEncoderReranker:
    """Uses sentence-transformers directly so we get the scores back (LangChain's
    CrossEncoderReranker discards them, which is why chain.py's confidence floor never fired)."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise SystemExit("pip install sentence-transformers  (needed for rerank variants)") from e
        self.model = CrossEncoder(model_name)

    def rank(self, query: str, docs: list) -> list[tuple[int, float]]:
        """Returns [(position_in_docs, score)] best first."""
        scores = self.model.predict([(query, d.page_content) for d in docs])
        order = sorted(range(len(docs)), key=lambda i: -float(scores[i]))
        return [(i, float(scores[i])) for i in order]


class Retrieval:
    """Index built once and shared by every variant."""

    def __init__(self, chunks, embeddings, args):
        for i, c in enumerate(chunks):
            c.metadata["chunk_idx"] = i
        self.chunks = chunks
        self.args = args
        self.store = FAISS.from_documents(chunks, embeddings)
        self._bm25 = None
        self._reranker = None

    def vector(self, q: str, depth: int) -> list[int]:
        return [d.metadata["chunk_idx"] for d in self.store.similarity_search(q, k=depth)]

    def bm25(self, q: str, depth: int) -> list[int]:
        if self._bm25 is None:
            try:
                from rank_bm25 import BM25Okapi
            except ImportError as e:
                raise SystemExit("pip install rank_bm25  (needed for hybrid variants)") from e
            self._bm25 = BM25Okapi([tokenize(c.page_content) for c in self.chunks])
        scores = self._bm25.get_scores(tokenize(q))
        ranked = sorted(range(len(self.chunks)), key=lambda i: -float(scores[i]))
        return [i for i in ranked[:depth] if scores[i] > 0]  # zero-score chunks are noise, not matches

    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.args.rerank_model)
        return self._reranker

    def retrieve(self, cfg: dict, q: str) -> dict:
        a = self.args
        rerank = cfg.get("rerank", False)
        if cfg["search"] == "vector":
            pool = self.vector(q, a.pool if rerank else cfg["k"])
        else:
            depth = a.pool if rerank else max(cfg["k"], a.pool)
            pool = rrf([self.vector(q, depth), self.bm25(q, depth)], a.weights, a.rrf_c)

        if not rerank:
            return {"final": pool[: cfg["k"]], "pool": pool, "top_score": None}

        ranked = self.reranker().rank(q, [self.chunks[i] for i in pool])
        final = [pool[pos] for pos, _ in ranked[: cfg["k"]]]
        return {"final": final, "pool": pool, "top_score": ranked[0][1]}


# --------------------------------------------------------------------------
# Per-variant pipeline (save after every question, resume on rerun)
# --------------------------------------------------------------------------
def variant_fingerprint(text, items, name, cfg, args) -> str:
    payload = {
        "doc": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "items": hashlib.sha256(json.dumps(items, sort_keys=True).encode("utf-8")).hexdigest(),
        "chunker": args.chunker, "variant": name, "cfg": cfg,
        "pool": args.pool, "weights": args.weights, "rrf_c": args.rrf_c,
        "rerank_model": args.rerank_model if cfg.get("rerank") else None,
        "embed": EMBED_MODEL, "gen_model": args.gen_model, "judge_model": args.judge_model,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def stage_retrieve_and_answer(cfg, get_retrieval, items, args, llm, d: Path, fp: str, mode: str):
    cache_path = d / "answers.json"
    cache = {}
    if not args.force:
        saved = load_json(cache_path)
        if saved and saved.get("fingerprint") == fp:
            cache = {r["id"]: r for r in saved["rows"]}
            if cache:
                print(f"  resuming: {len(cache)} questions already saved")

    rows = []
    for item in items:
        row = cache.get(item["id"])
        if row is None:
            ret = get_retrieval()  # builds the index only if something is actually left to do
            t0 = time.time()
            out = ret.retrieve(cfg, item["question"])
            ms = round((time.time() - t0) * 1000, 1)
            contexts = [ret.chunks[i].page_content for i in out["final"]]
            pool_ctx = [ret.chunks[i].page_content for i in out["pool"]]
            row = {
                "id": item["id"], "type": item["type"], "answerable": item["answerable"],
                "question": item["question"], "ground_truth": item["ground_truth"],
                "contexts": contexts, "retrieval_ms": ms, "pool_size": len(out["pool"]),
            }
            if out["top_score"] is not None:
                row["top_score"] = round(out["top_score"], 4)
            if item["evidence"]:
                row["evidence_recall"], row["hit"] = evidence_recall(contexts, item["evidence"])
                row["pool_evidence_recall"], _ = evidence_recall(pool_ctx, item["evidence"])

        if mode != "retrieval_only" and "answer" not in row:
            row["answer"] = (PROMPT | llm).invoke(
                {"context": "\n\n".join(row["contexts"]), "question": row["question"]}
            ).content
            row["abstained"] = abstained(row["answer"])

        rows.append(row)
        atomic_write(cache_path, json.dumps({"fingerprint": fp, "rows": rows}, indent=2))
    return rows


def run_variant(name, cfg, text, items, args, get_retrieval, llm, out_dir: Path):
    d = out_dir / "variants" / name
    d.mkdir(parents=True, exist_ok=True)
    mode = requested_mode(args)
    fp = variant_fingerprint(text, items, name, cfg, args)

    prev = load_json(d / "summary.json")
    if should_skip(prev, fp, mode, args.force):
        print("  already complete, skipping (use --force to redo)")
        return prev

    rows = stage_retrieve_and_answer(cfg, get_retrieval, items, args, llm, d, fp, mode)

    scores, n_incomplete = {}, 0
    if mode == "full":
        scores, n_incomplete = stage_ragas(rows, args, d, fp)

    records = []
    for r in rows:
        rec = {"variant": name}
        rec.update({k: v for k, v in r.items() if k not in {"contexts", "question", "ground_truth"}})
        rec.update(scores.get(r["id"], {}))
        records.append(rec)
    per_q = pd.DataFrame(records)
    write_csv(d / "per_question.csv", per_q)

    ans = per_q[per_q["answerable"]]
    unans = per_q[~per_q["answerable"]]
    summary = {
        "variant": name,
        "status": "complete" if n_incomplete == 0 else "incomplete",
        "mode": mode,
        "fingerprint": fp,
        "search": cfg["search"],
        "k_final": cfg["k"],
        "rerank": bool(cfg.get("rerank", False)),
        "retrieval_ms": mean_or_nan(per_q["retrieval_ms"]),
        "avg_pool_size": mean_or_nan(per_q["pool_size"]),
        "pool_evidence_recall": mean_or_nan(ans["pool_evidence_recall"]),
        "evidence_recall": mean_or_nan(ans["evidence_recall"]),
        "hit_rate": mean_or_nan(ans["hit"].dropna().astype(float)),
    }
    if "top_score" in per_q.columns:
        summary["top_score_answerable"] = mean_or_nan(ans["top_score"])
        summary["top_score_unanswerable"] = mean_or_nan(unans["top_score"])
    if mode != "retrieval_only":
        summary["abstain_rate_unanswerable"] = mean_or_nan(unans["abstained"].astype(float))
        summary["false_refusal_rate_answerable"] = mean_or_nan(ans["abstained"].astype(float))
    if mode == "full":
        for col in sorted({k for s in scores.values() for k in s}):
            summary[col] = mean_or_nan(per_q[col])
        n_expected = int(per_q["answerable"].sum())
        summary["ragas_rows_ok"] = n_expected - n_incomplete
        summary["ragas_rows_expected"] = n_expected

    atomic_write(d / "summary.json", json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------
# Combined views
# --------------------------------------------------------------------------
def rebuild_combined(out_dir: Path, order: list[str]):
    root = out_dir / "variants"
    summaries, details = [], []
    names = [n for n in order if (root / n).exists()]
    for name in names:
        s = load_json(root / name / "summary.json")
        if s:
            summaries.append(s)
        pq = root / name / "per_question.csv"
        if pq.exists():
            details.append(pd.read_csv(pq))
    summary_df = pd.DataFrame(summaries)
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    if not summary_df.empty:
        write_csv(out_dir / "summary.csv", summary_df)
    if not detail_df.empty:
        write_csv(out_dir / "per_question.csv", detail_df)
    return summary_df, detail_df


def by_type_table(detail_df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Evidence recall per question type per variant: shows WHICH kinds of question each change helps."""
    ans = detail_df[detail_df["answerable"] & detail_df["evidence_recall"].notna()]
    table = ans.pivot_table(index="type", columns="variant", values="evidence_recall", aggfunc="mean")
    return table[[v for v in order if v in table.columns]].round(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--doc", default="quillstream_handbook.md")
    p.add_argument("--eval", default="eval_set.json")
    p.add_argument("--out", default="results_retrieval")
    p.add_argument("--chunker", default="header_recursive_800", choices=list(STRATEGIES))
    p.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--pool", type=int, default=10,
                   help="candidates fetched before reranking / per retriever in hybrid (chain.py used 15)")
    p.add_argument("--weights", type=float, nargs=2, default=[0.6, 0.4], metavar=("VECTOR", "BM25"))
    p.add_argument("--rrf-c", type=int, default=60)
    p.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--skip-ragas", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--gen-model", default="gpt-4o-mini")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--ragas-retries", type=int, default=3)
    p.add_argument("--ragas-workers", type=int, default=4)
    p.add_argument("--ragas-timeout", type=int, default=180)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    text = Path(args.doc).read_text(encoding="utf-8")
    items = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, max_retries=6)
    llm = None if args.retrieval_only else ChatOpenAI(
        model=args.gen_model, temperature=0, max_retries=6, timeout=60
    )

    holder = {}

    def get_retrieval() -> Retrieval:
        if "r" not in holder:
            chunks = STRATEGIES[args.chunker](text)
            print(f"  [index] {args.chunker}: {len(chunks)} chunks (built once, shared by all variants)")
            holder["r"] = Retrieval(chunks, embeddings, args)
        return holder["r"]

    order = list(VARIANTS)
    problems = []
    for name in args.variants:
        print(f"\n>>> variant: {name}  {VARIANTS[name]}")
        try:
            summary = run_variant(name, VARIANTS[name], text, items, args, get_retrieval, llm, out_dir)
        except (Exception, SystemExit) as e:
            if isinstance(e, SystemExit):  # a missing package: say so and stop, don't mark every variant failed
                raise
            traceback.print_exc()
            summary = {"variant": name, "status": "failed", "mode": requested_mode(args),
                       "error": f"{type(e).__name__}: {e}"}
            d = out_dir / "variants" / name
            d.mkdir(parents=True, exist_ok=True)
            atomic_write(d / "summary.json", json.dumps(summary, indent=2))
        if summary.get("status") != "complete":
            problems.append(name)
        rebuild_combined(out_dir, order)

    summary_df, detail_df = rebuild_combined(out_dir, order)
    print("\n================ SUMMARY ================")
    if summary_df.empty:
        print("(no results)")
    else:
        drop = [c for c in ("fingerprint", "error") if c in summary_df.columns]
        print(summary_df.drop(columns=drop).to_string(index=False))
    if not detail_df.empty and "evidence_recall" in detail_df.columns:
        table = by_type_table(detail_df, order)
        write_csv(out_dir / "by_type.csv", table.reset_index())
        print("\n=========== EVIDENCE RECALL BY QUESTION TYPE ===========")
        print(table.to_string())
    print(f"\nCombined CSVs in {out_dir}/ : summary.csv, per_question.csv, by_type.csv")

    if problems:
        print(f"\nNOT COMPLETE: {', '.join(problems)}")
        print("Rerun only these (saved answers and good RAGAS scores are reused):")
        print(f"  python {Path(sys.argv[0]).name} --variants {' '.join(problems)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
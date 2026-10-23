import argparse
import json
import time
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

HEADERS = [('#', 'h1'), ('##', 'h2'), ('###', 'h3')]
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

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
            chunk_size=800, chunk_overlap=120, separators=SEPARATORS
        )
    return [_doc(c) for c in splitter.split_text(text)]

def header_only(text: str) -> list[Document]:
    """One chunk per markdown section, however long or short."""
    return [_doc(s.page_content, _path(s)) for s in _sections(text)]

def header_recursive(text: str, size: int = 800, overlap: int = 120):
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

def semantic_chunking(text: str) -> list[Document]:
    """Split where embedding similarity between sentences drops (needs langchain_experimental)."""
    from langchain_experimental.text_splitter import SemanticChunker

    chunker = SemanticChunker(
        OpenAIEmbeddings(model="text-embedding-3-small"),
        breakpoint_threshold_type="percentile"
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
    return sum(found)/len(found), any(found)

def abstained(answer: str) -> bool:
    a = answer.lower()
    return "don't have enough information" in a or "do not have enough information" in a

def run_ragas(samples: list[dict], judge_model: str) -> pd.DataFrame:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    judge_llm = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0))
    judge_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s["contexts"],
                reference=s["ground_truth"],
            )
            for s in samples
        ]
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=judge_llm),
            ResponseRelevancy(llm=judge_llm, embeddings=judge_emb),
            LLMContextPrecisionWithReference(llm=judge_llm),
            LLMContextRecall(llm=judge_llm),
        ]
    )
    df = result.to_pandas()
    text_cols = {"user_input", "retrieved_contexts", "response", "reference"}
    return df[[c for c in df.columns if c not in text_cols]]

def evaluate_strategy(name, chunker, text, items, args, embeddings, llm, out_dir):
    chunks = chunker(text)
    lengths = [len(c.page_content) for c in chunks]

    (out_dir/f"chunks_{name}.json").write_text(
        json.dumps(
            [
                {"i": i , "section": c.metadata.get("section_path"),
                 "chars": len(c.page_content), "text": c.page_content}
                for i,c in enumerate(chunks)
            ],
            indent=2,
        )
    )

    if args.show_chunks:
        print(f"\n===== {name}: first {args.show_chunks} chunks =====")
        for i,c in enumerate(chunks[: args.show_chunks]):
            print(f"--- chunk {i} ({len(c.page_content)} chars, {c.metadata['section_path']}) ---")
            print(c.page_content)

    store = FAISS.from_documents(chunks, embeddings)
    retriever = store.as_retriever(search_kwargs={"k": args.k})

    rows, ragas_inputs = [], []
    retrieval_ms = []

    for item in items:
        t0 = time.time()
        contexts = [d.page_content for d in retriever.invoke(item["question"])]
        retrieval_ms.append((time.time() - t0) * 1000)
        row = {"id": item["id"], "type": item["type"], "answerable": item["answerable"]}
        if item["evidence"]:
            row["evidence_recall"], row["hit"] = evidence_recall(contexts, item["evidence"])

        if not args.retrieval_only:
            answer = (PROMPT | llm).invoke(
                {"context": "\n\n".join(contexts), "question": item["question"]}
            ).content
            row["answer"] = answer
            if not item["answerable"]:
                row["abstained"] = abstained(answer)
            elif not args.skip_ragas:
                ragas_inputs.append(
                    {"question": item["question"], "answer": answer,
                    "contexts": contexts, "ground_truth": item["ground_truth"]}
                )
        rows.append(row)

    per_q = pd.DataFrame(rows)
    if ragas_inputs:
        scores = run_ragas(ragas_inputs, args.judge_model)
        answerable_idx = per_q.index[per_q["answerable"]]
        scores.index = answerable_idx
        per_q = per_q.join(scores)

    ans = per_q[per_q["answerable"]]
    summary = {
        "strategy": name,
        "n_chunks": len(chunks),
        "avg_chunks": round(sum(lengths) / len(lengths)),
        "max_chars": max(lengths),
        "retrieval_ms": round(sum(retrieval_ms) / len(retrieval_ms), 1),
        "evidence_recall": round(ans["evidence_recall"].mean(), 3),
        "hit_rate": round(ans["hit"].mean(), 3),
    }
    if "abstained" in per_q:
        summary["abstain_rate_unanswerable"] = per_q["abstained"].dropna().mean()

    if ragas_inputs:
        for col in scores.columns:
            summary[col] = round(scores[col].mean(), 3)

    per_q.insert(0, "strategy", name)
    return summary, per_q



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
    p.add_argument("--show-chunks", type=int, default=0, help="print the first N chunks per strategy")
    p.add_argument("--gen-model", default="gpt-4o-mini")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    text = Path(args.doc).read_text(encoding="utf-8")
    items = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]
 
    strategies = {n: STRATEGIES[n] for n in args.strategies}
    if args.semantic:
        strategies["semantic"] = semantic_chunking
 
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = None if args.retrieval_only else ChatOpenAI(model=args.gen_model, temperature=0)
 
    summaries, details = [], []
    for name, chunker in strategies.items():
        print(f"\n>>> running strategy: {name}")
        summary, per_q = evaluate_strategy(name, chunker, text, items, args, embeddings, llm, out_dir)
        summaries.append(summary)
        details.append(per_q)
 
    summary_df = pd.DataFrame(summaries)
    detail_df = pd.concat(details, ignore_index=True)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    detail_df.to_csv(out_dir / "per_question.csv", index=False)
 
    print("\n================ SUMMARY ================")
    print(summary_df.to_string(index=False))
    print(f"\nPer-question rows: {out_dir / 'per_question.csv'}")
    print(f"Chunk dumps:       {out_dir / 'chunks_<strategy>.json'}")

if __name__ == "__main__":
    main()
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("rag_pipeline")

def chunk_document(
        raw_markdown: str,
        source_id: str,
        allowed_roles: list[str]
) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[('#', 'h1'), ('##', 'h2'), ('###', 'h3')]
    )
    sections = header_splitter.split_text(raw_markdown)

    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        seperators=['\n\n', '\n', '.', ' ']
    )

    chunks: list[Document] = []

    for section in sections:
        pieces = sub_splitter.split_text(section.page_content) if len(section.page_content) > 800 else [section.page_content]
        for i, piece in enumerate(pieces):
            content_hash = hashlib.sha256(piece.encode()).hexdigest()[:16]
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source_id": source_id,
                        "chunk_id": f"{source_id}:{content_hash}",
                        "section_path": " > ".join(section.metadata.values()) or "root",
                        "allowed_roles": allowed_roles,
                        "ingested_at": time.time(),
                    }
                )
            )

    return chunks



class IngestionManager:
    def __init__(self, vectorestore: FAISS):
        self.vectorstore = vectorestore
        self._known_chunk_ids: set[str] = set()

    def sync(self, new_chunks: list[Document], source_id: str):
        new_ids = [c.metadata["chunk_id"] for c in new_chunks]
        stale_ids = {cid for cid in self._known_chunk_ids if cid.startswith(f"{source_id}:") and cid not in new_ids}
        to_add = [c for c in new_chunks if c.metadata["chunk_id"] not in self._known_chunk_ids]

        if stale_ids:
            logger.info(f"Removing {len(stale_ids)} stale chunks for {source_id}")
            self.vectorstore.delete(list(stale_ids))
            self._known_chunk_ids -= stale_ids
 
        if to_add:
            logger.info(f"Adding {len(to_add)} new/changed chunks for {source_id}")
            self.vectorstore.add_documents(to_add, ids=[c.metadata["chunk_id"] for c in to_add])
            self._known_chunk_ids |= {c.metadata["chunk_id"] for c in to_add}
        else:
            logger.info(f"No changes for {source_id}, skipping embedding calls")


@dataclass
class UserContext:
    user_id: str
    roles: list[str]

def build_retriever(vectorstore: FAISS, all_docs: list[Document], user: UserContext, k: int = 15):
    def rbac_filter(doc: Document) -> bool:
        return bool(set(doc.metadata.get("allowed_roles",[])) & set(user.roles))

    visible_docs = [d for d in all_docs if rbac_filter(d)]
    if not visible_docs:
        logger.warning(f"user={user.user_id} has zero visible documents")

    bm25_retriever = BM25Retriever.from_documents(visible_docs)
    bm25_retriever.k = k

    vector_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k,
            "filter": rbac_filter,
        }
    )

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )

def build_reranked_retriever(base_retriever, top_n: int = 4):
    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)

MIN_RELEVANT_CHUNKS = 1
CONFIDENCE_FLOOR = 0.35

@dataclass
class RetrievalResult:
    chunks: list[Document]
    scores: list[float]
    is_confident: bool
    reason: Optional[str] = None

def retrieve_wtih_confidence(reranked_retriever, query: str) -> RetrievalResult:
    results = reranked_retriever.invoke(query)
    if len(results) < MIN_RELEVANT_CHUNKS:
        return RetrievalResult(chunks=[], scores=[], is_confident=False, reason="no_results_after_rbac_and_filter")

    scores = [d.metadata.get("relevant_score", 0.0) for d in results]
    if max(scores, default=0.0) < CONFIDENCE_FLOOR:
        return RetrievalResult(chunks=results, scores=scores, is_confident=False,
                                reason="below_confidence_floor")

    return RetrievalResult(chunks=results, scores=scores, is_confident=True)


@dataclass
class QueryTrace:
    query: str
    user_id: str
    retrieval_ms: float
    generation_ms: float
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    was_confident: bool = True
    answer: str = ""

    def log(self):
        logger.info(
            f"query_trace user={self.user_id} "
            f"retrieval_ms={self.retrieval_ms:.0f} generation_ms={self.generation_ms:.0f} "
            f"chunks={self.retrieved_chunk_ids} scores={[round(s,3) for s in self.scores]} "
            f"confident={self.was_confident}"
        )

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "Asnwer the question using ONLY the context below",
    "If the context doesnt't contain the answer, say you don't have enough information.\n\n",
    "Context: \n{context}\n\nQuestion: {question}"
)

FALLBACK_ANSWER = "I don't have enough confidently relevant information to answer that."

def answer_query(reranked_retriever, llm: ChatOpenAI, query: str, user: UserContext) -> tuple[str, QueryTrace]:
    t0 = time.time()
    retrieval = retrieve_wtih_confidence(reranked_retriever, query)
    retrieval_ms = (time.time() - t0) * 1000

    trace = QueryTrace(
        qeury = query, user_id=user.id, retrieval_ms=retrieval_ms, generation_ms=0.0,
        retrieved_chunk_ids=[d.metadata.get("chunk_id", "?") for d in retrieval.chunks],
        scores=retrieval.scores, was_confident=retrieval.is_confident
    )

    if not retrieval.is_confident:
        trace.answer = FALLBACK_ANSWER
        trace.log()
        return FALLBACK_ANSWER, trace

    t1 = time.time()
    context = "\n\n".join(f"[{d.metadata['section_path']}] {d.page_content}" for d in retrieval.chunks)
    chain = ANSWER_PROMPT | llm
    answer = chain.invoke({"context": context, "question": query}).content
    trace.generation_ms = (time.time() - t1) * 1000
    trace.answer = answer
    trace.log()
    return answer, trace


# EVAL 

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall

def run_eval_suite(reranked_retriever, llm, eval_set: list[dict], user: UserContext, judge_llm=None, judge_embeddings=None):
    """
    eval_set: [{"question": ..., "ground_truth": ...}, ...]
    Run this in CI whenever chunking, retrieval weights, reranker, or prompt changes.
    Fail the build if scores regress below a baseline.
 
    judge_llm / judge_embeddings: optionally pass a specific judge model for scoring
    (often a cheaper, fixed model kept constant across runs so scores stay comparable
    over time). Defaults to wrapping the same `llm` passed in, with a fresh
    OpenAIEmbeddings instance for the embedding-based metrics.
    """
     
    evaluater_llm = judge_llm or LangchainLLMWrapper
    evaluater_embeddings = judge_embeddings or LangchainEmbeddingsWrapper

    samples = []
    for item in eval_set:
        ans, trace = answer_query(reranked_retriever, llm, item["question"], user)
        retrieved_contexts = [d.page_content for d in retrieve_wtih_confidence(reranked_retriever, item["question"].chunks)]

        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=ans,
            retrieved_contexts=retrieved_contexts,
            reference=item["ground_truth"]
        ))

    evaluation_dataset = EvaluationDataset(samples=samples)

    metrics = [
        Faithfulness(llm=evaluater_llm),
        ResponseRelevancy(llm=evaluater_llm, embeddings=evaluater_embeddings),
        LLMContextPrecisionWithReference(llm=evaluater_llm),
        LLMContextRecall(llm=evaluater_llm)
    ]

    results = evaluate(dataset=evaluation_dataset, metrics=metrics)
    df = results.to_pandas()
    logger.info(f"\n{df}")
    return df

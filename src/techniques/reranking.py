"""Technique 2: Cross-Encoder Reranking.

Retrieve a large candidate set (top-k*4), then rerank with a cross-encoder
that jointly encodes query+document for precise relevance scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore


@dataclass
class RerankResult:
    document: Document
    initial_rank: int
    final_rank: int
    rerank_score: float


class CrossEncoderReranker:
    """HuggingFace cross-encoder reranker (BAAI/bge-reranker-v2-m3 by default)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 5,
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("pip install sentence-transformers") from exc

        self._model = CrossEncoder(model_name, device=device)
        self._top_n = top_n

    def rerank(self, query: str, docs: list[Document]) -> list[RerankResult]:
        pairs = [(query, doc.page_content) for doc in docs]
        scores: list[float] = self._model.predict(pairs).tolist()

        scored = sorted(
            zip(docs, scores, range(len(docs))),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            RerankResult(
                document=doc,
                initial_rank=init_rank,
                final_rank=final_rank,
                rerank_score=score,
            )
            for final_rank, (doc, score, init_rank) in enumerate(scored[: self._top_n])
        ]


class RerankerRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        reranker: CrossEncoderReranker,
        candidate_k: int = 20,
    ) -> None:
        self._vector_store = vector_store
        self._reranker = reranker
        self._candidate_k = candidate_k

    def retrieve(self, query: str) -> list[RerankResult]:
        candidates = self._vector_store.similarity_search(query, k=self._candidate_k)
        return self._reranker.rerank(query, candidates)


def run_reranking_rag(
    query: str,
    retriever: RerankerRetriever,
    llm: BaseChatModel,
) -> dict[str, Any]:
    results = retriever.retrieve(query)
    context = "\n\n".join(r.document.page_content for r in results)
    prompt = f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {query}"
    response = llm.invoke(prompt)
    return {
        "answer": response.content,
        "sources": [r.document.metadata for r in results],
        "rerank_scores": [r.rerank_score for r in results],
        "rank_improvements": [r.initial_rank - r.final_rank for r in results],
    }

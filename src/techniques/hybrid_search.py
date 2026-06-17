"""Technique 1: Hybrid Search with Reciprocal Rank Fusion (RRF).

Dense vector search + BM25 sparse search fused via RRF scoring.
RRF(d) = sum(1 / (k + rank_i(d))) where k=60 is the RRF constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore


@dataclass
class HybridSearchResult:
    document: Document
    dense_rank: int | None
    sparse_rank: int | None
    rrf_score: float


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[tuple[Document, float]]:
    """Merge N ranked lists via RRF. Higher score = better."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            doc_id = doc.page_content
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_map[doc_id] = doc

    return sorted(
        [(doc_map[doc_id], score) for doc_id, score in scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )


class HybridRetriever:
    """Combines dense FAISS retrieval with BM25 sparse retrieval via RRF."""

    def __init__(
        self,
        vector_store: VectorStore,
        docs: list[Document],
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError("pip install rank-bm25") from exc

        self._vector_store = vector_store
        self._docs = docs
        self._top_k = top_k
        self._rrf_k = rrf_k

        tokenized = [doc.page_content.lower().split() for doc in docs]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str) -> list[HybridSearchResult]:
        dense_docs = self._vector_store.similarity_search(query, k=self._top_k * 2)

        tokens = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokens)
        sparse_ranked = [
            self._docs[i] for i in sorted(range(len(bm25_scores)), key=lambda x: -bm25_scores[x])
        ][: self._top_k * 2]

        fused = reciprocal_rank_fusion([dense_docs, sparse_ranked], k=self._rrf_k)

        dense_index = {doc.page_content: rank for rank, doc in enumerate(dense_docs)}
        sparse_index = {doc.page_content: rank for rank, doc in enumerate(sparse_ranked)}

        return [
            HybridSearchResult(
                document=doc,
                dense_rank=dense_index.get(doc.page_content),
                sparse_rank=sparse_index.get(doc.page_content),
                rrf_score=score,
            )
            for doc, score in fused[: self._top_k]
        ]


def run_hybrid_rag(
    query: str,
    retriever: HybridRetriever,
    llm: BaseChatModel,
) -> dict[str, Any]:
    results = retriever.retrieve(query)
    context = "\n\n".join(r.document.page_content for r in results)
    prompt = f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {query}"
    response = llm.invoke(prompt)
    return {
        "answer": response.content,
        "sources": [r.document.metadata for r in results],
        "rrf_scores": [r.rrf_score for r in results],
    }

"""Technique 4: Context Compression.

After retrieval, compress each document to only the spans relevant
to the query — reduces noise, fits more sources in the context window.

Two flavours:
- LLMExtractor: ask the LLM to extract relevant sentences
- EmbeddingsFilter: keep only semantically-close sentences (no LLM call)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

_EXTRACT_PROMPT = (
    "Extract only the sentences from the passage below that are directly relevant "
    "to the question. Return extracted text only, no commentary.\n\n"
    "Question: {query}\n\nPassage:\n{passage}"
)


@dataclass
class CompressionResult:
    original_doc: Document
    compressed_doc: Document
    compression_ratio: float


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


class LLMExtractor:
    """Compress each doc by extracting only query-relevant sentences via LLM."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def compress(self, query: str, doc: Document) -> Document:
        extracted = self._llm.invoke(
            _EXTRACT_PROMPT.format(query=query, passage=doc.page_content)
        ).content.strip()
        return Document(page_content=extracted or doc.page_content, metadata=doc.metadata)


class EmbeddingsFilter:
    """Keep only sentences whose embedding similarity to the query exceeds threshold."""

    def __init__(self, embeddings: Embeddings, similarity_threshold: float = 0.76) -> None:
        self._embeddings = embeddings
        self._threshold = similarity_threshold

    def compress(self, query: str, doc: Document) -> Document:
        sentences = [s.strip() for s in doc.page_content.split(".") if s.strip()]
        if not sentences:
            return doc

        query_vec = self._embeddings.embed_query(query)
        sent_vecs = self._embeddings.embed_documents(sentences)

        relevant = [
            s
            for s, v in zip(sentences, sent_vecs)
            if _cosine_similarity(query_vec, v) >= self._threshold
        ]
        text = ". ".join(relevant) + "." if relevant else doc.page_content
        return Document(page_content=text, metadata=doc.metadata)


class CompressionRetriever:
    """Retrieve candidates, then compress each to query-relevant content."""

    def __init__(
        self,
        vector_store: VectorStore,
        compressor: LLMExtractor | EmbeddingsFilter,
        candidate_k: int = 10,
        top_k: int = 5,
    ) -> None:
        self._vector_store = vector_store
        self._compressor = compressor
        self._candidate_k = candidate_k
        self._top_k = top_k

    def invoke(self, query: str) -> list[Document]:
        candidates = self._vector_store.similarity_search(query, k=self._candidate_k)
        compressed = [self._compressor.compress(query, doc) for doc in candidates]
        non_empty = [d for d in compressed if d.page_content.strip()]
        return non_empty[: self._top_k]


def build_llm_compression_retriever(
    vector_store: VectorStore,
    llm: BaseChatModel,
    top_k: int = 5,
) -> CompressionRetriever:
    return CompressionRetriever(vector_store, LLMExtractor(llm), top_k=top_k)


def build_embeddings_filter_retriever(
    vector_store: VectorStore,
    embeddings: Embeddings,
    similarity_threshold: float = 0.76,
    top_k: int = 5,
) -> CompressionRetriever:
    return CompressionRetriever(
        vector_store,
        EmbeddingsFilter(embeddings, similarity_threshold=similarity_threshold),
        top_k=top_k,
    )


def measure_compression(original: Document, compressed: Document) -> CompressionResult:
    orig_len = len(original.page_content)
    comp_len = len(compressed.page_content)
    ratio = comp_len / orig_len if orig_len > 0 else 1.0
    return CompressionResult(
        original_doc=original,
        compressed_doc=compressed,
        compression_ratio=ratio,
    )


def run_compression_rag(
    query: str,
    retriever: CompressionRetriever,
    llm: BaseChatModel,
) -> dict[str, Any]:
    compressed_docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in compressed_docs)
    prompt = f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {query}"
    answer = llm.invoke(prompt).content
    return {
        "answer": answer,
        "sources": [doc.metadata for doc in compressed_docs],
        "compressed_doc_count": len(compressed_docs),
        "total_context_chars": len(context),
    }

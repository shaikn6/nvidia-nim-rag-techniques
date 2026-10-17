"""Concrete BaseRAGPipeline implementations for the benchmark harness.

Includes:
- NaiveRAGPipeline: plain retrieve-then-generate baseline, no optimization.
- Thin adapters wrapping each technique's `run_*` function so it conforms
  to BaseRAGPipeline and can be dropped into `run_benchmark`.
- DemoLLM / DemoEmbeddings / DemoReranker: deterministic, offline stand-ins
  used by the benchmark's __main__ block when no NVIDIA NIM API key is
  configured, so `python -m src.pipeline.benchmark` produces real output
  without requiring API credentials.
"""

from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from src.pipeline.base import BaseRAGPipeline
from src.techniques.compression import CompressionRetriever, run_compression_rag
from src.techniques.corrective_rag import _GENERATE_PROMPT, run_corrective_rag
from src.techniques.hybrid_search import HybridRetriever, run_hybrid_rag
from src.techniques.query_rewriting import run_query_rewriting_rag
from src.techniques.reranking import RerankerRetriever, RerankResult, run_reranking_rag


class NaiveRAGPipeline(BaseRAGPipeline):
    """Naive RAG baseline: embed, retrieve top-k, stuff into the prompt. No optimization."""

    def __init__(self, vector_store: VectorStore, llm: BaseChatModel, top_k: int = 5) -> None:
        self._vector_store = vector_store
        self._llm = llm
        self._top_k = top_k

    @property
    def name(self) -> str:
        return "naive_baseline"

    def run(self, query: str) -> dict[str, Any]:
        docs = self._vector_store.similarity_search(query, k=self._top_k)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = _GENERATE_PROMPT.format(context=context, query=query)
        response = self._llm.invoke(prompt)
        return {
            "answer": response.content,
            "sources": [doc.metadata for doc in docs],
        }


class _DelegatingPipeline(BaseRAGPipeline):
    """Adapter wrapping a technique's `run_*(query, retriever, llm)` function."""

    def __init__(self, name: str, run_fn: Any, retriever: Any, llm: BaseChatModel) -> None:
        self._name = name
        self._run_fn = run_fn
        self._retriever = retriever
        self._llm = llm

    @property
    def name(self) -> str:
        return self._name

    def run(self, query: str) -> dict[str, Any]:
        return self._run_fn(query, self._retriever, self._llm)


class HybridSearchPipeline(_DelegatingPipeline):
    """Adapter wrapping run_hybrid_rag for the benchmark harness."""

    def __init__(self, retriever: HybridRetriever, llm: BaseChatModel) -> None:
        super().__init__("hybrid_search", run_hybrid_rag, retriever, llm)


class RerankingPipeline(_DelegatingPipeline):
    """Adapter wrapping run_reranking_rag for the benchmark harness."""

    def __init__(self, retriever: RerankerRetriever, llm: BaseChatModel) -> None:
        super().__init__("reranking", run_reranking_rag, retriever, llm)


class QueryRewritingPipeline(BaseRAGPipeline):
    """Adapter wrapping run_query_rewriting_rag for the benchmark harness."""

    def __init__(
        self,
        strategy: str,
        llm: BaseChatModel,
        vector_store: VectorStore,
        embeddings: Any | None = None,
    ) -> None:
        self._strategy = strategy
        self._llm = llm
        self._vector_store = vector_store
        self._embeddings = embeddings

    @property
    def name(self) -> str:
        return f"query_rewriting_{self._strategy}"

    def run(self, query: str) -> dict[str, Any]:
        return run_query_rewriting_rag(
            query, self._strategy, self._llm, self._vector_store, self._embeddings
        )


class CompressionPipeline(_DelegatingPipeline):
    """Adapter wrapping run_compression_rag for the benchmark harness."""

    def __init__(self, retriever: CompressionRetriever, llm: BaseChatModel) -> None:
        super().__init__("compression", run_compression_rag, retriever, llm)


class CorrectiveRAGPipeline(BaseRAGPipeline):
    """Adapter wrapping run_corrective_rag for the benchmark harness."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm: BaseChatModel,
        top_k: int = 5,
        relevance_threshold: float = 0.5,
    ) -> None:
        self._vector_store = vector_store
        self._llm = llm
        self._top_k = top_k
        self._relevance_threshold = relevance_threshold

    @property
    def name(self) -> str:
        return "corrective_rag"

    def run(self, query: str) -> dict[str, Any]:
        return run_corrective_rag(
            query, self._vector_store, self._llm, self._top_k, self._relevance_threshold
        )


class DemoLLM:
    """Deterministic, offline stand-in for a NIM-backed chat model.

    Only used by the benchmark's __main__ block when NVIDIA_NIM_API_KEY isn't
    configured, so the documented quickstart command works without requiring
    real API credentials.
    """

    def invoke(self, prompt: str) -> SimpleNamespace:
        if prompt.strip().endswith("Answer with exactly one word: yes or no."):
            content = self._grade(prompt)
        elif "Extract only the sentences" in prompt:
            content = self._extract(prompt)
        elif "authoritative document" in prompt:
            content = self._hyde(prompt)
        elif "different phrasings" in prompt:
            content = self._multi_query(prompt)
        elif "broader, more general question" in prompt:
            content = self._step_back(prompt)
        elif "Context:" in prompt and "Question:" in prompt:
            content = self._generate(prompt)
        else:
            content = "[demo-llm] mock response"
        return SimpleNamespace(content=content)

    @staticmethod
    def _grade(prompt: str) -> str:
        doc_part = prompt.split("Document:\n", 1)[-1].split("\n\nAnswer")[0]
        query_part = prompt.split("Question: ", 1)[-1].split("\n\n")[0]
        doc_words = {w.strip(".,?!:;") for w in doc_part.lower().split()}
        query_words = {w.strip(".,?!:;") for w in query_part.lower().split()}
        return "yes" if doc_words & query_words else "no"

    @staticmethod
    def _generate(prompt: str) -> str:
        context = prompt.split("Context:\n", 1)[-1].split("\n\nQuestion:")[0]
        question = prompt.rsplit("Question:", 1)[-1].strip()
        snippet = " ".join(context.split())[:180]
        return f"[demo-llm] Regarding '{question}': {snippet or 'no relevant context found.'}"

    @staticmethod
    def _extract(prompt: str) -> str:
        passage = prompt.split("Passage:\n", 1)[-1].strip()
        first_sentence = passage.split(".")[0].strip()
        return f"{first_sentence}." if first_sentence else ""

    @staticmethod
    def _hyde(prompt: str) -> str:
        question = prompt.split("directly answer: ", 1)[-1].split("\n")[0]
        return f"A relevant passage would explain {question.rstrip('?')} in technical detail."

    @staticmethod
    def _multi_query(prompt: str) -> str:
        question = prompt.rsplit("Question: ", 1)[-1].strip()
        base = question.rstrip("?").lower()
        return "\n".join([question, f"What is {base}?", f"Can you explain {base}?"])

    @staticmethod
    def _step_back(prompt: str) -> str:
        question = prompt.split("Question: ", 1)[-1].split("\n\n")[0]
        return f"What are the general principles behind {question.rstrip('?').lower()}?"


class DemoEmbeddings(Embeddings):
    """Deterministic, offline stand-in for NIM embeddings (hashed bag-of-words vectors)."""

    _DIM = 64

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self._DIM
        for word in text.lower().split():
            # sha256, not hash(): str hashing is salted per process (PYTHONHASHSEED).
            digest = hashlib.sha256(word.encode()).digest()
            vec[int.from_bytes(digest[:8], "big") % self._DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class DemoReranker:
    """Deterministic, offline stand-in for CrossEncoderReranker (word-overlap scoring)."""

    def __init__(self, top_n: int = 5) -> None:
        self._top_n = top_n

    def rerank(self, query: str, docs: list[Document]) -> list[RerankResult]:
        query_words = set(query.lower().split())
        scored = sorted(
            (
                (doc, init_rank, len(query_words & set(doc.page_content.lower().split())))
                for init_rank, doc in enumerate(docs)
            ),
            key=lambda triple: triple[2],
            reverse=True,
        )
        return [
            RerankResult(
                document=doc,
                initial_rank=init_rank,
                final_rank=final_rank,
                rerank_score=float(score),
            )
            for final_rank, (doc, init_rank, score) in enumerate(scored[: self._top_n])
        ]

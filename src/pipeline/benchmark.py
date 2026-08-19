"""Benchmark harness — compare all 5 RAG techniques on a query set."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.pipeline.base import BaseRAGPipeline


@dataclass
class PipelineResult:
    pipeline_name: str
    query: str
    answer: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class BenchmarkReport:
    results: list[PipelineResult] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        by_pipeline: dict[str, list[PipelineResult]] = {}
        for r in self.results:
            by_pipeline.setdefault(r.pipeline_name, []).append(r)

        return {
            name: {
                "avg_latency_ms": sum(r.latency_ms for r in rows) / len(rows),
                "error_rate": sum(1 for r in rows if r.error) / len(rows),
                "query_count": len(rows),
            }
            for name, rows in by_pipeline.items()
        }


def run_benchmark(
    pipelines: list[BaseRAGPipeline],
    queries: list[str],
) -> BenchmarkReport:
    report = BenchmarkReport()

    for query in queries:
        for pipeline in pipelines:
            t0 = time.perf_counter()
            try:
                result = pipeline.run(query)
                answer = result.get("answer", "")
                error = None
            except Exception as exc:
                answer = ""
                error = str(exc)
                result = {}

            latency_ms = (time.perf_counter() - t0) * 1000

            report.results.append(
                PipelineResult(
                    pipeline_name=pipeline.name,
                    query=query,
                    answer=answer,
                    latency_ms=latency_ms,
                    metadata={k: v for k, v in result.items() if k != "answer"},
                    error=error,
                )
            )

    return report


def _build_llm_and_embeddings() -> tuple[Any, Any]:
    """Real NIM clients when NVIDIA_NIM_API_KEY is configured, offline demo stand-ins otherwise."""
    import os

    from src.pipeline.pipelines import DemoEmbeddings, DemoLLM

    api_key = os.environ.get("NVIDIA_NIM_API_KEY")
    if api_key and api_key != "nvapi-xxxxxxxxxxxxxxxxxxxx":
        from src.core.embeddings import build_nim_embeddings
        from src.core.llm import build_nim_llm

        return build_nim_llm(), build_nim_embeddings()

    return DemoLLM(), DemoEmbeddings()


def _run_demo() -> None:
    """Benchmark all 5 techniques against the naive baseline on a small bundled corpus."""
    from src.core.vector_store import build_faiss_store
    from src.pipeline.demo_data import DEMO_CORPUS, DEMO_QUERIES
    from src.pipeline.pipelines import (
        CompressionPipeline,
        CorrectiveRAGPipeline,
        DemoReranker,
        HybridSearchPipeline,
        NaiveRAGPipeline,
        QueryRewritingPipeline,
        RerankingPipeline,
    )
    from src.techniques.compression import build_embeddings_filter_retriever
    from src.techniques.hybrid_search import HybridRetriever
    from src.techniques.reranking import RerankerRetriever

    llm, embeddings = _build_llm_and_embeddings()
    vector_store = build_faiss_store(DEMO_CORPUS, embeddings)

    hybrid_retriever = HybridRetriever(vector_store, DEMO_CORPUS)
    reranker = DemoReranker()
    rerank_retriever = RerankerRetriever(vector_store, reranker)
    compression_retriever = build_embeddings_filter_retriever(vector_store, embeddings)

    pipelines: list[BaseRAGPipeline] = [
        NaiveRAGPipeline(vector_store, llm),
        HybridSearchPipeline(hybrid_retriever, llm),
        RerankingPipeline(rerank_retriever, llm),
        QueryRewritingPipeline("hyde", llm, vector_store, embeddings),
        CompressionPipeline(compression_retriever, llm),
        CorrectiveRAGPipeline(vector_store, llm),
    ]

    report = run_benchmark(pipelines, DEMO_QUERIES)

    print(f"Benchmark: {len(pipelines)} pipelines x {len(DEMO_QUERIES)} queries\n")
    summary = report.summary()
    for name, stats in summary.items():
        print(
            f"{name:>24}  avg_latency_ms={stats['avg_latency_ms']:.2f}  "
            f"error_rate={stats['error_rate']:.2f}  query_count={stats['query_count']}"
        )


if __name__ == "__main__":
    _run_demo()

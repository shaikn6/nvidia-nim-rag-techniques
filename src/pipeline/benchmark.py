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

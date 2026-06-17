"""Tests for benchmark harness."""

from __future__ import annotations


import pytest

from src.pipeline.base import BaseRAGPipeline
from src.pipeline.benchmark import run_benchmark


class MockPipeline(BaseRAGPipeline):
    def __init__(self, name: str, answer: str = "mock answer") -> None:
        self._name = name
        self._answer = answer

    @property
    def name(self) -> str:
        return self._name

    def run(self, query: str) -> dict:
        return {"answer": self._answer, "sources": []}


class ErrorPipeline(BaseRAGPipeline):
    @property
    def name(self) -> str:
        return "error_pipeline"

    def run(self, query: str) -> dict:
        raise RuntimeError("pipeline crashed")


class TestRunBenchmark:
    def test_result_count_matches_queries_times_pipelines(self):
        pipelines = [MockPipeline("p1"), MockPipeline("p2")]
        queries = ["q1", "q2", "q3"]
        report = run_benchmark(pipelines, queries)
        assert len(report.results) == 6

    def test_error_pipeline_captured_not_raised(self):
        report = run_benchmark([ErrorPipeline()], ["query"])
        assert report.results[0].error == "pipeline crashed"

    def test_latency_is_positive(self):
        report = run_benchmark([MockPipeline("p1")], ["query"])
        assert report.results[0].latency_ms >= 0

    def test_answer_captured_correctly(self):
        report = run_benchmark(
            [MockPipeline("p1", answer="specific answer")], ["query"]
        )
        assert report.results[0].answer == "specific answer"

    def test_empty_queries_returns_empty_report(self):
        report = run_benchmark([MockPipeline("p1")], [])
        assert len(report.results) == 0

    def test_empty_pipelines_returns_empty_report(self):
        report = run_benchmark([], ["query"])
        assert len(report.results) == 0


class TestBenchmarkReport:
    def test_summary_has_entry_per_pipeline(self):
        report = run_benchmark(
            [MockPipeline("hybrid"), MockPipeline("rerank")], ["q1", "q2"]
        )
        summary = report.summary()
        assert "hybrid" in summary
        assert "rerank" in summary

    def test_summary_avg_latency_is_float(self):
        report = run_benchmark([MockPipeline("p1")], ["q1"])
        assert isinstance(report.summary()["p1"]["avg_latency_ms"], float)

    def test_summary_error_rate_for_error_pipeline(self):
        report = run_benchmark([ErrorPipeline()], ["q1", "q2"])
        assert report.summary()["error_pipeline"]["error_rate"] == 1.0

    def test_summary_error_rate_zero_for_clean_pipeline(self):
        report = run_benchmark([MockPipeline("clean")], ["q1", "q2"])
        assert report.summary()["clean"]["error_rate"] == 0.0


class TestBaseRAGPipeline:
    def test_abstract_class_not_instantiatable(self):
        with pytest.raises(TypeError):
            BaseRAGPipeline()

    def test_concrete_subclass_works(self):
        p = MockPipeline("test")
        assert p.name == "test"
        assert "answer" in p.run("query")

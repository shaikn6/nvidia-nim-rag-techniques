"""Tests for Technique 2: Cross-Encoder Reranking."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.techniques.reranking import (
    RerankResult,
    RerankerRetriever,
    run_reranking_rag,
)


@pytest.fixture()
def mock_cross_encoder_cls():
    """Patch CrossEncoder at the sentence_transformers module level (lazy import)."""
    with patch("sentence_transformers.CrossEncoder") as MockCE:
        instance = MagicMock()
        instance.predict.return_value = np.array([0.9, 0.3, 0.7, 0.5, 0.1])
        MockCE.return_value = instance
        yield MockCE, instance


@pytest.fixture()
def reranker(mock_cross_encoder_cls):
    from src.techniques.reranking import CrossEncoderReranker

    return CrossEncoderReranker(top_n=3)


class TestCrossEncoderReranker:
    def test_rerank_returns_sorted_by_score(self, reranker, mock_cross_encoder_cls, sample_docs):
        results = reranker.rerank("NIM query", sample_docs[:5])
        scores = [r.rerank_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_respects_top_n(self, mock_cross_encoder_cls, sample_docs):
        from src.techniques.reranking import CrossEncoderReranker

        r = CrossEncoderReranker(top_n=2)
        results = r.rerank("query", sample_docs[:5])
        assert len(results) == 2

    def test_rerank_result_has_initial_and_final_rank(
        self, reranker, mock_cross_encoder_cls, sample_docs
    ):
        results = reranker.rerank("query", sample_docs[:5])
        for r in results:
            assert isinstance(r.initial_rank, int)
            assert isinstance(r.final_rank, int)

    def test_missing_sentence_transformers_raises(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            from src.techniques.reranking import CrossEncoderReranker

            with pytest.raises(ImportError, match="sentence-transformers"):
                CrossEncoderReranker()


class TestRerankerRetriever:
    def test_retrieve_calls_reranker(self, mock_cross_encoder_cls, mock_vector_store, sample_docs):
        _, mock_ce = mock_cross_encoder_cls
        mock_ce.predict.return_value = np.array([0.9, 0.3, 0.7, 0.5, 0.1] * 4)
        from src.techniques.reranking import CrossEncoderReranker

        reranker = CrossEncoderReranker(top_n=3)
        retriever = RerankerRetriever(mock_vector_store, reranker, candidate_k=5)
        results = retriever.retrieve("What is RAG?")
        assert len(results) == 3
        assert all(isinstance(r, RerankResult) for r in results)


class TestRunRerankingRag:
    def test_returns_answer_scores_and_improvements(
        self, mock_cross_encoder_cls, mock_vector_store, mock_llm, sample_docs
    ):
        _, mock_ce = mock_cross_encoder_cls
        mock_ce.predict.return_value = np.array([0.9, 0.3, 0.7, 0.5, 0.1] * 4)
        from src.techniques.reranking import CrossEncoderReranker

        reranker = CrossEncoderReranker(top_n=3)
        retriever = RerankerRetriever(mock_vector_store, reranker)
        result = run_reranking_rag("What is reranking?", retriever, mock_llm)
        assert "answer" in result
        assert "rerank_scores" in result
        assert "rank_improvements" in result

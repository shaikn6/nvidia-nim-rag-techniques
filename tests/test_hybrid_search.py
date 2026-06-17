"""Tests for Technique 1: Hybrid Search / RRF."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.techniques.hybrid_search import (
    HybridSearchResult,
    reciprocal_rank_fusion,
    run_hybrid_rag,
)


class TestRRF:
    def test_single_list_preserves_order(self, sample_docs):
        result = reciprocal_rank_fusion([sample_docs])
        assert [d for d, _ in result] == sample_docs

    def test_higher_rank_gets_higher_score(self, sample_docs):
        result = reciprocal_rank_fusion([sample_docs])
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_shared_doc_scores_summed(self, sample_docs):
        list1 = sample_docs[:3]
        list2 = [sample_docs[0], sample_docs[3], sample_docs[4]]
        result = reciprocal_rank_fusion([list1, list2])
        score_map = {d.page_content: s for d, s in result}
        assert score_map[sample_docs[0].page_content] > score_map[sample_docs[3].page_content]

    def test_custom_k_affects_score_gap(self, sample_docs):
        result_k60 = reciprocal_rank_fusion([sample_docs], k=60)
        result_k1 = reciprocal_rank_fusion([sample_docs], k=1)
        gap_k60 = result_k60[0][1] - result_k60[-1][1]
        gap_k1 = result_k1[0][1] - result_k1[-1][1]
        assert gap_k1 > gap_k60

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([]) == []

    def test_deduplicates_documents(self, sample_docs):
        result = reciprocal_rank_fusion([sample_docs, sample_docs])
        contents = [d.page_content for d, _ in result]
        assert len(contents) == len(set(contents))


class TestHybridRetriever:
    @pytest.fixture()
    def retriever(self, mock_vector_store, sample_docs):
        # BM25Okapi is imported lazily inside __init__ — patch at rank_bm25 module level
        with patch("rank_bm25.BM25Okapi") as MockBM25:
            mock_bm25 = MagicMock()
            mock_bm25.get_scores.return_value = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
            MockBM25.return_value = mock_bm25
            from src.techniques.hybrid_search import HybridRetriever

            return HybridRetriever(mock_vector_store, sample_docs, top_k=3)

    def test_retrieve_returns_hybrid_results(self, retriever):
        results = retriever.retrieve("NVIDIA NIM inference")
        assert len(results) == 3
        assert all(isinstance(r, HybridSearchResult) for r in results)

    def test_rrf_scores_are_positive(self, retriever):
        results = retriever.retrieve("RAG retrieval")
        assert all(r.rrf_score > 0 for r in results)

    def test_missing_bm25_raises_import_error(self, mock_vector_store, sample_docs):
        with patch.dict("sys.modules", {"rank_bm25": None}):
            import src.techniques.hybrid_search as mod

            with pytest.raises(ImportError, match="rank-bm25"):
                mod.HybridRetriever(mock_vector_store, sample_docs)


class TestRunHybridRag:
    def test_returns_answer_and_metadata(self, mock_vector_store, mock_llm, sample_docs):
        with patch("rank_bm25.BM25Okapi") as MockBM25:
            mock_bm25 = MagicMock()
            mock_bm25.get_scores.return_value = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
            MockBM25.return_value = mock_bm25
            from src.techniques.hybrid_search import HybridRetriever

            retriever = HybridRetriever(mock_vector_store, sample_docs, top_k=3)
            result = run_hybrid_rag("What is NIM?", retriever, mock_llm)

        assert "answer" in result
        assert "sources" in result
        assert "rrf_scores" in result
        assert result["answer"] == "This is a mock LLM answer."

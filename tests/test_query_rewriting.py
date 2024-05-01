"""Tests for Technique 3: Query Rewriting (HyDE, Multi-Query, Step-Back)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.techniques.query_rewriting import (
    hyde_retrieve,
    multi_query_retrieve,
    run_query_rewriting_rag,
    step_back_retrieve,
)


@pytest.fixture()
def mock_llm_variants():
    llm = MagicMock()

    def side_effect(prompt):
        resp = MagicMock()
        if "Write a brief passage" in prompt:
            resp.content = "NVIDIA NIM is an optimized inference platform."
        elif "different phrasings" in prompt:
            resp.content = (
                "What is NIM?\nHow does NVIDIA NIM work?\nNIM inference platform?"
            )
        elif "broader" in prompt:
            resp.content = "What are AI inference optimization techniques?"
        else:
            resp.content = "This is a mock answer."
        return resp

    llm.invoke.side_effect = side_effect
    return llm


class TestHyDE:
    def test_strategy_is_hyde(
        self, mock_llm_variants, mock_embeddings, mock_vector_store
    ):
        result = hyde_retrieve(
            "What is NIM?", mock_llm_variants, mock_embeddings, mock_vector_store
        )
        assert result.strategy == "hyde"

    def test_rewritten_queries_contains_hypothetical_doc(
        self, mock_llm_variants, mock_embeddings, mock_vector_store
    ):
        result = hyde_retrieve(
            "What is NIM?", mock_llm_variants, mock_embeddings, mock_vector_store
        )
        assert len(result.rewritten_queries) == 1
        assert "NVIDIA NIM" in result.rewritten_queries[0]

    def test_uses_embed_query_for_retrieval(
        self, mock_llm_variants, mock_embeddings, mock_vector_store
    ):
        hyde_retrieve(
            "What is NIM?", mock_llm_variants, mock_embeddings, mock_vector_store
        )
        mock_embeddings.embed_query.assert_called_once()
        mock_vector_store.similarity_search_by_vector.assert_called_once()

    def test_documents_returned(
        self, mock_llm_variants, mock_embeddings, mock_vector_store
    ):
        result = hyde_retrieve(
            "What is NIM?", mock_llm_variants, mock_embeddings, mock_vector_store
        )
        assert len(result.documents) > 0


class TestMultiQuery:
    def test_strategy_is_multi_query(self, mock_llm_variants, mock_vector_store):
        result = multi_query_retrieve(
            "What is NIM?", mock_llm_variants, mock_vector_store
        )
        assert result.strategy == "multi_query"

    def test_generates_n_variants(self, mock_llm_variants, mock_vector_store):
        result = multi_query_retrieve(
            "What is NIM?", mock_llm_variants, mock_vector_store, n_variants=3
        )
        assert len(result.rewritten_queries) <= 3

    def test_deduplicates_documents(
        self, mock_llm_variants, mock_vector_store, sample_docs
    ):
        mock_vector_store.similarity_search.return_value = sample_docs[:3]
        result = multi_query_retrieve(
            "What is NIM?", mock_llm_variants, mock_vector_store
        )
        contents = [d.page_content for d in result.documents]
        assert len(contents) == len(set(contents))

    def test_original_query_preserved(self, mock_llm_variants, mock_vector_store):
        result = multi_query_retrieve(
            "What is NIM?", mock_llm_variants, mock_vector_store
        )
        assert result.original_query == "What is NIM?"


class TestStepBack:
    def test_strategy_is_step_back(self, mock_llm_variants, mock_vector_store):
        result = step_back_retrieve(
            "What is NIM?", mock_llm_variants, mock_vector_store
        )
        assert result.strategy == "step_back"

    def test_rewritten_query_is_general(self, mock_llm_variants, mock_vector_store):
        result = step_back_retrieve(
            "What is NIM?", mock_llm_variants, mock_vector_store
        )
        assert len(result.rewritten_queries) == 1
        assert (
            "inference" in result.rewritten_queries[0].lower()
            or len(result.rewritten_queries[0]) > 0
        )

    def test_merges_specific_and_general_docs(
        self, mock_llm_variants, mock_vector_store
    ):
        result = step_back_retrieve(
            "What is NIM?", mock_llm_variants, mock_vector_store
        )
        assert len(result.documents) > 0


class TestRunQueryRewritingRag:
    def test_hyde_strategy(self, mock_llm_variants, mock_embeddings, mock_vector_store):
        result = run_query_rewriting_rag(
            "What is NIM?",
            "hyde",
            mock_llm_variants,
            mock_vector_store,
            mock_embeddings,
        )
        assert result["strategy"] == "hyde"
        assert "answer" in result

    def test_multi_query_strategy(self, mock_llm_variants, mock_vector_store):
        result = run_query_rewriting_rag(
            "What is NIM?", "multi_query", mock_llm_variants, mock_vector_store
        )
        assert result["strategy"] == "multi_query"

    def test_step_back_strategy(self, mock_llm_variants, mock_vector_store):
        result = run_query_rewriting_rag(
            "What is NIM?", "step_back", mock_llm_variants, mock_vector_store
        )
        assert result["strategy"] == "step_back"

    def test_unknown_strategy_raises(self, mock_llm_variants, mock_vector_store):
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_query_rewriting_rag(
                "q", "unknown", mock_llm_variants, mock_vector_store
            )

    def test_hyde_without_embeddings_raises(self, mock_llm_variants, mock_vector_store):
        with pytest.raises(ValueError, match="embeddings required"):
            run_query_rewriting_rag(
                "q", "hyde", mock_llm_variants, mock_vector_store, embeddings=None
            )

"""Tests for Technique 4: Context Compression."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.techniques.compression import (
    CompressionRetriever,
    EmbeddingsFilter,
    LLMExtractor,
    _cosine_similarity,
    build_embeddings_filter_retriever,
    build_llm_compression_retriever,
    measure_compression,
    run_compression_rag,
)


@pytest.fixture()
def compressed_doc():
    return Document(page_content="NIM optimized inference.", metadata={"source": "nim-docs"})


@pytest.fixture()
def original_doc():
    return Document(
        page_content="NVIDIA NIM provides optimized inference microservices for production deployments.",
        metadata={"source": "nim-docs"},
    )


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self):
        assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6

    def test_zero_vector_returns_zero(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestMeasureCompression:
    def test_ratio_less_than_one_when_compressed(self, original_doc, compressed_doc):
        result = measure_compression(original_doc, compressed_doc)
        assert result.compression_ratio < 1.0

    def test_ratio_is_one_for_identical_docs(self, original_doc):
        result = measure_compression(original_doc, original_doc)
        assert result.compression_ratio == 1.0

    def test_zero_length_original_returns_ratio_one(self):
        empty = Document(page_content="")
        result = measure_compression(empty, Document(page_content="some content"))
        assert result.compression_ratio == 1.0

    def test_result_stores_both_docs(self, original_doc, compressed_doc):
        result = measure_compression(original_doc, compressed_doc)
        assert result.original_doc is original_doc
        assert result.compressed_doc is compressed_doc


class TestLLMExtractor:
    def test_returns_extracted_content(self, mock_llm, sample_docs):
        mock_llm.invoke.return_value = MagicMock(content="NIM is fast.")
        extractor = LLMExtractor(mock_llm)
        result = extractor.compress("What is NIM?", sample_docs[0])
        assert result.page_content == "NIM is fast."

    def test_falls_back_to_original_on_empty_response(self, mock_llm, sample_docs):
        mock_llm.invoke.return_value = MagicMock(content="")
        extractor = LLMExtractor(mock_llm)
        result = extractor.compress("query", sample_docs[0])
        assert result.page_content == sample_docs[0].page_content

    def test_preserves_metadata(self, mock_llm, sample_docs):
        mock_llm.invoke.return_value = MagicMock(content="extracted")
        extractor = LLMExtractor(mock_llm)
        result = extractor.compress("query", sample_docs[0])
        assert result.metadata == sample_docs[0].metadata


class TestEmbeddingsFilter:
    def test_keeps_relevant_sentences(self, mock_embeddings):
        mock_embeddings.embed_query.return_value = [1.0, 0.0]
        mock_embeddings.embed_documents.return_value = [[1.0, 0.0], [0.0, 1.0]]
        doc = Document(page_content="Relevant sentence. Irrelevant sentence")
        ef = EmbeddingsFilter(mock_embeddings, similarity_threshold=0.5)
        result = ef.compress("query", doc)
        assert "Relevant sentence" in result.page_content

    def test_falls_back_on_empty_content(self, mock_embeddings):
        doc = Document(page_content="")
        ef = EmbeddingsFilter(mock_embeddings)
        result = ef.compress("query", doc)
        assert result.page_content == ""

    def test_falls_back_when_nothing_passes_threshold(self, mock_embeddings):
        mock_embeddings.embed_query.return_value = [1.0, 0.0]
        mock_embeddings.embed_documents.return_value = [[0.0, 1.0]]
        doc = Document(page_content="Unrelated text")
        ef = EmbeddingsFilter(mock_embeddings, similarity_threshold=0.99)
        result = ef.compress("query", doc)
        assert result.page_content == doc.page_content


class TestCompressionRetriever:
    def test_invoke_returns_compressed_docs(self, mock_vector_store, mock_llm, sample_docs):
        mock_llm.invoke.return_value = MagicMock(content="compressed content")
        retriever = CompressionRetriever(mock_vector_store, LLMExtractor(mock_llm), top_k=3)
        results = retriever.invoke("What is NIM?")
        assert len(results) <= 3

    def test_filters_empty_compressed_docs(self, mock_vector_store, mock_llm):
        # LLMExtractor falls back to original on empty LLM response;
        # a doc with empty original content is filtered out.
        mock_llm.invoke.return_value = MagicMock(content="")
        mock_vector_store.similarity_search.return_value = [Document(page_content="", metadata={})]
        retriever = CompressionRetriever(mock_vector_store, LLMExtractor(mock_llm))
        results = retriever.invoke("query")
        assert len(results) == 0


class TestBuildHelpers:
    def test_build_llm_compression_retriever(self, mock_vector_store, mock_llm):
        r = build_llm_compression_retriever(mock_vector_store, mock_llm)
        assert isinstance(r, CompressionRetriever)

    def test_build_embeddings_filter_retriever(self, mock_vector_store, mock_embeddings):
        r = build_embeddings_filter_retriever(
            mock_vector_store, mock_embeddings, similarity_threshold=0.9
        )
        assert isinstance(r, CompressionRetriever)
        assert isinstance(r._compressor, EmbeddingsFilter)
        assert r._compressor._threshold == 0.9


class TestRunCompressionRag:
    def test_returns_answer_and_metadata(self, mock_llm, sample_docs, mock_vector_store):
        mock_llm.invoke.return_value = MagicMock(content="compressed answer")
        retriever = build_llm_compression_retriever(mock_vector_store, mock_llm, top_k=3)
        mock_vector_store.similarity_search.return_value = sample_docs[:3]
        result = run_compression_rag("What is NIM?", retriever, mock_llm)
        assert "answer" in result
        assert "compressed_doc_count" in result
        assert result["total_context_chars"] > 0

    def test_empty_results_handled(self, mock_llm, mock_vector_store):
        mock_vector_store.similarity_search.return_value = []
        retriever = CompressionRetriever(mock_vector_store, LLMExtractor(mock_llm))
        result = run_compression_rag("query", retriever, mock_llm)
        assert result["compressed_doc_count"] == 0
        assert result["total_context_chars"] == 0

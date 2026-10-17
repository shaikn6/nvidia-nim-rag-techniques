"""Tests for the naive baseline + technique adapters + demo stand-ins."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from langchain_core.documents import Document

from src.pipeline.base import BaseRAGPipeline
from src.pipeline.pipelines import (
    CompressionPipeline,
    CorrectiveRAGPipeline,
    DemoEmbeddings,
    DemoLLM,
    DemoReranker,
    HybridSearchPipeline,
    NaiveRAGPipeline,
    QueryRewritingPipeline,
    RerankingPipeline,
)
from src.techniques.compression import CompressionRetriever, EmbeddingsFilter
from src.techniques.hybrid_search import HybridRetriever
from src.techniques.reranking import RerankerRetriever


class TestNaiveRAGPipeline:
    def test_conforms_to_base(self):
        assert issubclass(NaiveRAGPipeline, BaseRAGPipeline)

    def test_name(self, mock_vector_store, mock_llm):
        p = NaiveRAGPipeline(mock_vector_store, mock_llm)
        assert p.name == "naive_baseline"

    def test_run_returns_answer_and_sources(self, mock_vector_store, mock_llm):
        p = NaiveRAGPipeline(mock_vector_store, mock_llm)
        result = p.run("what is nim?")
        assert result["answer"] == "This is a mock LLM answer."
        assert len(result["sources"]) == 5

    def test_run_calls_similarity_search_with_top_k(self, mock_vector_store, mock_llm):
        p = NaiveRAGPipeline(mock_vector_store, mock_llm, top_k=3)
        p.run("query")
        mock_vector_store.similarity_search.assert_called_once_with("query", k=3)


class TestHybridSearchPipeline:
    def test_name(self, mock_vector_store, sample_docs, mock_llm):
        retriever = HybridRetriever(mock_vector_store, sample_docs)
        p = HybridSearchPipeline(retriever, mock_llm)
        assert p.name == "hybrid_search"

    def test_run_delegates_to_run_hybrid_rag(self, mock_vector_store, sample_docs, mock_llm):
        retriever = HybridRetriever(mock_vector_store, sample_docs)
        p = HybridSearchPipeline(retriever, mock_llm)
        result = p.run("what is bm25?")
        assert "answer" in result
        assert "rrf_scores" in result


class TestRerankingPipeline:
    def test_name(self, mock_vector_store, mock_llm):
        retriever = RerankerRetriever(mock_vector_store, DemoReranker())
        p = RerankingPipeline(retriever, mock_llm)
        assert p.name == "reranking"

    def test_run_delegates_to_run_reranking_rag(self, mock_vector_store, mock_llm):
        retriever = RerankerRetriever(mock_vector_store, DemoReranker())
        p = RerankingPipeline(retriever, mock_llm)
        result = p.run("what is a cross-encoder?")
        assert "answer" in result
        assert "rerank_scores" in result


class TestQueryRewritingPipeline:
    def test_name_includes_strategy(self, mock_vector_store, mock_llm):
        p = QueryRewritingPipeline("multi_query", mock_llm, mock_vector_store)
        assert p.name == "query_rewriting_multi_query"

    def test_run_delegates_to_run_query_rewriting_rag(self, mock_vector_store, mock_llm):
        p = QueryRewritingPipeline("step_back", mock_llm, mock_vector_store)
        result = p.run("what is rag?")
        assert result["strategy"] == "step_back"

    def test_run_hyde_uses_embeddings(self, mock_vector_store, mock_llm, mock_embeddings):
        p = QueryRewritingPipeline("hyde", mock_llm, mock_vector_store, mock_embeddings)
        result = p.run("what is rag?")
        assert result["strategy"] == "hyde"


class TestCompressionPipeline:
    def test_name(self, mock_vector_store, mock_embeddings, mock_llm):
        retriever = CompressionRetriever(mock_vector_store, EmbeddingsFilter(mock_embeddings))
        p = CompressionPipeline(retriever, mock_llm)
        assert p.name == "compression"

    def test_run_delegates_to_run_compression_rag(self, mock_vector_store, mock_embeddings, mock_llm):
        retriever = CompressionRetriever(mock_vector_store, EmbeddingsFilter(mock_embeddings))
        p = CompressionPipeline(retriever, mock_llm)
        result = p.run("what is nim?")
        assert "compressed_doc_count" in result


class TestCorrectiveRAGPipeline:
    def test_name(self, mock_vector_store, mock_llm):
        p = CorrectiveRAGPipeline(mock_vector_store, mock_llm)
        assert p.name == "corrective_rag"

    def test_run_delegates_to_run_corrective_rag(self, mock_vector_store, mock_llm):
        mock_llm.invoke.return_value.content = "yes"
        p = CorrectiveRAGPipeline(mock_vector_store, mock_llm)
        result = p.run("what is crag?")
        assert "used_web_search" in result


class TestDemoLLM:
    def test_grade_yes_on_overlap(self):
        llm = DemoLLM()
        prompt = (
            "Is the following document relevant to the question?\n\n"
            "Question: what is rag?\n\nDocument:\nrag combines retrieval and generation\n\n"
            "Answer with exactly one word: yes or no."
        )
        assert llm.invoke(prompt).content == "yes"

    def test_grade_no_without_overlap(self):
        llm = DemoLLM()
        prompt = (
            "Is the following document relevant to the question?\n\n"
            "Question: zzz completely unrelated topic\n\nDocument:\nsomething else entirely\n\n"
            "Answer with exactly one word: yes or no."
        )
        assert llm.invoke(prompt).content == "no"

    def test_generate_answer_from_context(self):
        llm = DemoLLM()
        prompt = "Answer the question using only the context below.\n\nContext:\nsome fact\n\nQuestion: what?"
        result = llm.invoke(prompt).content
        assert "demo-llm" in result

    def test_extract_returns_first_sentence(self):
        llm = DemoLLM()
        prompt = (
            "Extract only the sentences from the passage below that are directly relevant "
            "to the question. Return extracted text only, no commentary.\n\n"
            "Question: q\n\nPassage:\nFirst sentence here. Second sentence."
        )
        assert llm.invoke(prompt).content == "First sentence here."

    def test_hyde_generates_hypothetical_passage(self):
        llm = DemoLLM()
        prompt = (
            "Write a brief passage (2-3 sentences) that would directly answer: what is rag?\n"
            "Do not reference the question. Write as if excerpted from an authoritative document."
        )
        assert "rag" in llm.invoke(prompt).content.lower()

    def test_multi_query_generates_variants(self):
        llm = DemoLLM()
        prompt = (
            "Generate 3 different phrasings of the following question. "
            "Return only the questions, one per line, no numbering.\n\nQuestion: what is rag?"
        )
        content = llm.invoke(prompt).content
        assert len(content.splitlines()) == 3

    def test_step_back_generates_general_question(self):
        llm = DemoLLM()
        prompt = (
            "Rewrite the following question as a broader, more general question "
            "about the underlying principle or concept.\n\nQuestion: what is rag?\n\nGeneral question:"
        )
        assert "principles" in llm.invoke(prompt).content

    def test_fallback_mock_response(self):
        llm = DemoLLM()
        assert llm.invoke("some unrecognized prompt shape").content == "[demo-llm] mock response"


class TestDemoEmbeddings:
    def test_embed_query_returns_unit_vector(self):
        emb = DemoEmbeddings()
        vec = emb.embed_query("hello world")
        assert len(vec) == 64
        assert pytest.approx(sum(v * v for v in vec), abs=1e-6) == 1.0

    def test_embed_documents_returns_one_vector_per_doc(self):
        emb = DemoEmbeddings()
        vecs = emb.embed_documents(["a b", "c d", "e f"])
        assert len(vecs) == 3

    def test_embed_query_empty_string_no_zero_division(self):
        emb = DemoEmbeddings()
        vec = emb.embed_query("")
        assert vec == [0.0] * 64

    def test_same_text_embeds_identically_across_instances(self):
        text = "reciprocal rank fusion merges rankings"
        assert DemoEmbeddings().embed_query(text) == DemoEmbeddings().embed_query(text)

    def test_embedding_is_stable_across_processes(self):
        # str hash() is salted per process, so two interpreters with different
        # PYTHONHASHSEED values would disagree if the embedding relied on it.
        code = (
            "from src.pipeline.pipelines import DemoEmbeddings;"
            "print(DemoEmbeddings().embed_query('reciprocal rank fusion'))"
        )
        outputs = {
            subprocess.run(
                [sys.executable, "-c", code],
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            for seed in ("1", "2")
        }
        assert len(outputs) == 1


class TestDemoReranker:
    def test_rerank_orders_by_overlap(self):
        docs = [
            Document(page_content="totally unrelated content"),
            Document(page_content="rag combines retrieval and generation"),
        ]
        reranker = DemoReranker(top_n=2)
        results = reranker.rerank("rag retrieval generation", docs)
        assert results[0].document.page_content == "rag combines retrieval and generation"

    def test_rerank_respects_top_n(self):
        docs = [Document(page_content=f"doc {i} rag") for i in range(5)]
        reranker = DemoReranker(top_n=2)
        results = reranker.rerank("rag", docs)
        assert len(results) == 2

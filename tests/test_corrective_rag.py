"""Tests for Technique 5: Corrective RAG (CRAG)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.techniques.corrective_rag import (
    CRAGState,
    _grade_document,
    _web_search,
    run_corrective_rag,
)


@pytest.fixture()
def relevant_llm():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="yes")
    return llm


@pytest.fixture()
def irrelevant_llm():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="no")
    return llm


class TestGradeDocument:
    def test_relevant_doc_returns_true(self, relevant_llm, sample_docs):
        assert _grade_document("NIM query", sample_docs[0], relevant_llm) is True

    def test_irrelevant_doc_returns_false(self, irrelevant_llm, sample_docs):
        assert _grade_document("NIM query", sample_docs[0], irrelevant_llm) is False

    def test_partial_yes_response_is_true(self, sample_docs):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="yes, it is relevant")
        assert _grade_document("query", sample_docs[0], llm) is True

    def test_empty_response_returns_false(self, sample_docs):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="")
        assert _grade_document("query", sample_docs[0], llm) is False


class TestWebSearch:
    def test_returns_empty_list_without_api_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        result = _web_search("NIM inference")
        assert result == []

    def test_returns_empty_list_on_exception(self):
        with patch(
            "langchain_community.tools.TavilySearchResults",
            side_effect=Exception("import error"),
        ):
            result = _web_search("query")
        assert result == []

    def test_returns_documents_when_tavily_available(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = [{"content": "NIM is fast.", "url": "https://nvidia.com"}]
        with patch("langchain_community.tools.TavilySearchResults", return_value=mock_tool):
            result = _web_search("NIM inference")
        assert len(result) == 1
        assert result[0].page_content == "NIM is fast."


class TestRunCorrectiveRag:
    def test_all_relevant_skips_web_search(self, mock_vector_store, sample_docs):
        llm = MagicMock()
        llm.invoke.side_effect = [MagicMock(content="yes")] * 5 + [
            MagicMock(content="Final answer.")
        ]
        result = run_corrective_rag("What is NIM?", mock_vector_store, llm)
        assert result["used_web_search"] is False
        assert "retrieve" in result["steps"]
        assert "generate" in result["steps"]

    def test_no_relevant_triggers_web_fallback(self, mock_vector_store):
        llm = MagicMock()
        llm.invoke.side_effect = [MagicMock(content="no")] * 5 + [MagicMock(content="Answer.")]
        with patch("src.techniques.corrective_rag._web_search", return_value=[]):
            result = run_corrective_rag(
                "obscure query", mock_vector_store, llm, relevance_threshold=0.5
            )
        assert "web_search_fallback" in result["steps"]

    def test_web_search_returns_docs_used(self, mock_vector_store):
        llm = MagicMock()
        llm.invoke.side_effect = [MagicMock(content="no")] * 5 + [MagicMock(content="Web answer.")]
        web_doc = Document(page_content="web result", metadata={"source": "web"})
        with patch("src.techniques.corrective_rag._web_search", return_value=[web_doc]):
            result = run_corrective_rag("query", mock_vector_store, llm, relevance_threshold=0.5)
        assert result["used_web_search"] is True

    def test_relevance_ratio_in_result(self, mock_vector_store):
        llm = MagicMock()
        llm.invoke.side_effect = [MagicMock(content="yes")] * 5 + [MagicMock(content="Answer.")]
        result = run_corrective_rag("query", mock_vector_store, llm)
        assert 0.0 <= result["relevance_ratio"] <= 1.0

    def test_answer_in_result(self, mock_vector_store):
        llm = MagicMock()
        llm.invoke.side_effect = [MagicMock(content="yes")] * 5 + [
            MagicMock(content="Final answer.")
        ]
        result = run_corrective_rag("What is RAG?", mock_vector_store, llm)
        assert result["answer"] == "Final answer."

    def test_crag_state_dataclass(self):
        state = CRAGState(query="test")
        assert state.query == "test"
        assert state.used_web_search is False
        assert state.answer == ""


class TestBuildCragGraph:
    def test_builds_graph_with_langgraph(self, mock_vector_store, relevant_llm):
        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        with patch("langgraph.graph.StateGraph", return_value=mock_graph):
            with patch("langgraph.graph.END", "END"):
                from src.techniques.corrective_rag import build_crag_graph

                graph = build_crag_graph(mock_vector_store, relevant_llm)

        assert graph is mock_compiled

    def test_missing_langgraph_raises(self, mock_vector_store, relevant_llm):
        with patch.dict("sys.modules", {"langgraph": None, "langgraph.graph": None}):
            from src.techniques.corrective_rag import build_crag_graph

            with pytest.raises(ImportError, match="langgraph"):
                build_crag_graph(mock_vector_store, relevant_llm)

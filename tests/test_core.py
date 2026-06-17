"""Tests for core NIM client factories."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestBuildNimLlm:
    def test_returns_chat_openai_instance(self):
        with patch("src.core.llm.ChatOpenAI") as MockLLM:
            MockLLM.return_value = MagicMock()
            from src.core.llm import build_nim_llm

            llm = build_nim_llm()
        MockLLM.assert_called_once()
        assert llm is MockLLM.return_value

    def test_custom_model_passed(self):
        with patch("src.core.llm.ChatOpenAI") as MockLLM:
            MockLLM.return_value = MagicMock()
            from src.core.llm import build_nim_llm

            build_nim_llm(model="meta/llama-3.1-8b-instruct")
        kwargs = MockLLM.call_args.kwargs
        assert kwargs["model"] == "meta/llama-3.1-8b-instruct"

    def test_uses_env_api_key(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_NIM_API_KEY", "test-key-123")
        with patch("src.core.llm.ChatOpenAI") as MockLLM:
            MockLLM.return_value = MagicMock()
            from src.core.llm import build_nim_llm

            build_nim_llm()
        kwargs = MockLLM.call_args.kwargs
        assert kwargs["api_key"] == "test-key-123"

    def test_fallback_placeholder_key(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
        with patch("src.core.llm.ChatOpenAI") as MockLLM:
            MockLLM.return_value = MagicMock()
            from src.core.llm import build_nim_llm

            build_nim_llm()
        kwargs = MockLLM.call_args.kwargs
        assert kwargs["api_key"] == "nim-placeholder"


class TestBuildNimEmbeddings:
    def test_returns_openai_embeddings_instance(self):
        with patch("src.core.embeddings.OpenAIEmbeddings") as MockEmb:
            MockEmb.return_value = MagicMock()
            from src.core.embeddings import build_nim_embeddings

            emb = build_nim_embeddings()
        MockEmb.assert_called_once()
        assert emb is MockEmb.return_value

    def test_custom_model_passed(self):
        with patch("src.core.embeddings.OpenAIEmbeddings") as MockEmb:
            MockEmb.return_value = MagicMock()
            from src.core.embeddings import build_nim_embeddings

            build_nim_embeddings(model="nvidia/custom-embed")
        kwargs = MockEmb.call_args.kwargs
        assert kwargs["model"] == "nvidia/custom-embed"


class TestBuildVectorStore:
    def test_build_faiss_store(self, sample_docs, mock_embeddings):
        with patch("src.core.vector_store.FAISS") as MockFAISS:
            MockFAISS.from_documents.return_value = MagicMock()
            from src.core.vector_store import build_faiss_store

            build_faiss_store(sample_docs, mock_embeddings)
        MockFAISS.from_documents.assert_called_once_with(sample_docs, mock_embeddings)

    def test_build_pinecone_raises_without_package(self, sample_docs, mock_embeddings, monkeypatch):
        monkeypatch.setenv("PINECONE_API_KEY", "test")
        with patch.dict("sys.modules", {"langchain_pinecone": None}):
            from src.core.vector_store import build_pinecone_store

            with pytest.raises(ImportError, match="langchain-pinecone"):
                build_pinecone_store(sample_docs, mock_embeddings)

    def test_build_pinecone_raises_without_api_key(self, sample_docs, mock_embeddings, monkeypatch):
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        # Mock the package as importable so we get past the ImportError guard
        mock_pc_module = MagicMock()
        with patch.dict("sys.modules", {"langchain_pinecone": mock_pc_module}):
            from src.core.vector_store import build_pinecone_store

            with pytest.raises(KeyError):
                build_pinecone_store(sample_docs, mock_embeddings)

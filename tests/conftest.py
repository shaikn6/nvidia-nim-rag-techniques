"""Shared fixtures — all external services mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


@pytest.fixture()
def sample_docs() -> list[Document]:
    return [
        Document(
            page_content="NVIDIA NIM provides optimized inference microservices.",
            metadata={"source": "nim-docs"},
        ),
        Document(
            page_content="RAG combines retrieval with language model generation.",
            metadata={"source": "rag-paper"},
        ),
        Document(
            page_content="LangGraph enables stateful multi-step agent workflows.",
            metadata={"source": "lg-docs"},
        ),
        Document(
            page_content="Cross-encoders jointly score query and document pairs.",
            metadata={"source": "rerank-paper"},
        ),
        Document(
            page_content="BM25 is a probabilistic sparse retrieval ranking function.",
            metadata={"source": "bm25-paper"},
        ),
        Document(
            page_content="HyDE generates a hypothetical answer before retrieval.",
            metadata={"source": "hyde-paper"},
        ),
    ]


@pytest.fixture()
def mock_llm():
    llm = MagicMock()
    response = MagicMock()
    response.content = "This is a mock LLM answer."
    llm.invoke.return_value = response
    return llm


@pytest.fixture()
def mock_embeddings():
    emb = MagicMock()
    emb.embed_query.return_value = [0.1] * 384
    emb.embed_documents.return_value = [[0.1] * 384] * 6
    return emb


@pytest.fixture()
def mock_vector_store(sample_docs):
    vs = MagicMock()
    vs.similarity_search.return_value = sample_docs[:5]
    vs.similarity_search_by_vector.return_value = sample_docs[:5]
    vs.as_retriever.return_value = MagicMock()
    return vs

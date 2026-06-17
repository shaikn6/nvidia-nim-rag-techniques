"""Vector store factory — FAISS (local) or Pinecone (cloud)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore


def build_faiss_store(docs: list[Document], embeddings: Embeddings) -> FAISS:
    return FAISS.from_documents(docs, embeddings)


def build_pinecone_store(
    docs: list[Document],
    embeddings: Embeddings,
    index_name: str | None = None,
) -> "VectorStore":
    try:
        from langchain_pinecone import PineconeVectorStore
    except ImportError as exc:
        raise ImportError("pip install langchain-pinecone pinecone-client") from exc

    api_key = os.environ["PINECONE_API_KEY"]
    idx = index_name or os.environ.get("PINECONE_INDEX", "nim-rag")
    return PineconeVectorStore.from_documents(
        docs,
        embeddings,
        index_name=idx,
        pinecone_api_key=api_key,
    )

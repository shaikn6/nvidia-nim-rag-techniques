"""Small bundled corpus + query set for the benchmark demo.

Used by `python -m src.pipeline.benchmark` so the quickstart command works
out of the box without requiring a real corpus to be supplied.
"""

from __future__ import annotations

from langchain_core.documents import Document

DEMO_CORPUS: list[Document] = [
    Document(
        page_content=(
            "NVIDIA NIM provides optimized inference microservices for deploying "
            "large language models and embedding models in production."
        ),
        metadata={"source": "nim-docs"},
    ),
    Document(
        page_content=(
            "Retrieval-Augmented Generation (RAG) combines a retriever that finds "
            "relevant documents with a language model that generates an answer "
            "grounded in that retrieved context."
        ),
        metadata={"source": "rag-paper"},
    ),
    Document(
        page_content=(
            "LangGraph enables stateful, multi-step agent workflows by modeling "
            "them as graphs of nodes and conditional edges."
        ),
        metadata={"source": "lg-docs"},
    ),
    Document(
        page_content=(
            "Cross-encoder rerankers jointly encode a query and a document pair, "
            "producing more precise relevance scores than bi-encoder similarity."
        ),
        metadata={"source": "rerank-paper"},
    ),
    Document(
        page_content=(
            "BM25 is a probabilistic sparse retrieval ranking function based on "
            "term frequency and inverse document frequency."
        ),
        metadata={"source": "bm25-paper"},
    ),
    Document(
        page_content=(
            "HyDE generates a hypothetical answer document before retrieval, then "
            "embeds that hypothetical document to search for real supporting text."
        ),
        metadata={"source": "hyde-paper"},
    ),
    Document(
        page_content=(
            "Reciprocal Rank Fusion merges multiple ranked lists by summing "
            "1 / (k + rank) across lists, favoring documents that rank well in "
            "more than one retrieval strategy."
        ),
        metadata={"source": "rrf-paper"},
    ),
    Document(
        page_content=(
            "Corrective RAG grades each retrieved document for relevance and "
            "falls back to a web search when retrieval quality is too low."
        ),
        metadata={"source": "crag-paper"},
    ),
    Document(
        page_content=(
            "Context compression removes irrelevant sentences from retrieved "
            "documents so the language model sees a smaller, more focused prompt."
        ),
        metadata={"source": "compression-paper"},
    ),
    Document(
        page_content=(
            "FAISS is a library for efficient similarity search and clustering of "
            "dense vectors, commonly used as a local vector store for RAG."
        ),
        metadata={"source": "faiss-docs"},
    ),
]

DEMO_QUERIES: list[str] = [
    "What is Retrieval-Augmented Generation?",
    "How does reciprocal rank fusion work?",
    "What does a cross-encoder reranker do?",
    "How does Corrective RAG decide to use web search?",
]

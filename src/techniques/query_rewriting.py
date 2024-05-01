"""Technique 3: Query Rewriting — HyDE, Multi-Query, Step-Back.

Three complementary strategies:
- HyDE: generate a hypothetical document, embed it, use that for retrieval
- Multi-Query: expand to N query variants, union results
- Step-Back: reformulate as a more general principle question
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

_HYDE_PROMPT = (
    "Write a brief passage (2-3 sentences) that would directly answer: {query}\n"
    "Do not reference the question. Write as if excerpted from an authoritative document."
)

_MULTI_QUERY_PROMPT = (
    "Generate {n} different phrasings of the following question. "
    "Return only the questions, one per line, no numbering.\n\nQuestion: {query}"
)

_STEP_BACK_PROMPT = (
    "Rewrite the following question as a broader, more general question "
    "about the underlying principle or concept.\n\nQuestion: {query}\n\nGeneral question:"
)


@dataclass
class QueryRewriteResult:
    strategy: str
    original_query: str
    rewritten_queries: list[str]
    documents: list[Document]


def hyde_retrieve(
    query: str,
    llm: BaseChatModel,
    embeddings: Embeddings,
    vector_store: VectorStore,
    top_k: int = 5,
) -> QueryRewriteResult:
    hypothetical_doc = llm.invoke(_HYDE_PROMPT.format(query=query)).content
    query_vec = embeddings.embed_query(hypothetical_doc)
    docs = vector_store.similarity_search_by_vector(query_vec, k=top_k)
    return QueryRewriteResult(
        strategy="hyde",
        original_query=query,
        rewritten_queries=[hypothetical_doc],
        documents=docs,
    )


def multi_query_retrieve(
    query: str,
    llm: BaseChatModel,
    vector_store: VectorStore,
    n_variants: int = 3,
    top_k: int = 5,
) -> QueryRewriteResult:
    raw = llm.invoke(_MULTI_QUERY_PROMPT.format(n=n_variants, query=query)).content
    variants = [q.strip() for q in raw.strip().splitlines() if q.strip()][:n_variants]

    seen: set[str] = set()
    merged: list[Document] = []
    for variant in variants:
        for doc in vector_store.similarity_search(variant, k=top_k):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)

    return QueryRewriteResult(
        strategy="multi_query",
        original_query=query,
        rewritten_queries=variants,
        documents=merged[:top_k],
    )


def step_back_retrieve(
    query: str,
    llm: BaseChatModel,
    vector_store: VectorStore,
    top_k: int = 5,
) -> QueryRewriteResult:
    general_q = llm.invoke(_STEP_BACK_PROMPT.format(query=query)).content.strip()

    specific_docs = vector_store.similarity_search(query, k=top_k // 2 + 1)
    general_docs = vector_store.similarity_search(general_q, k=top_k // 2 + 1)

    seen: set[str] = set()
    merged: list[Document] = []
    for doc in specific_docs + general_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append(doc)

    return QueryRewriteResult(
        strategy="step_back",
        original_query=query,
        rewritten_queries=[general_q],
        documents=merged[:top_k],
    )


def run_query_rewriting_rag(
    query: str,
    strategy: str,
    llm: BaseChatModel,
    vector_store: VectorStore,
    embeddings: Embeddings | None = None,
) -> dict[str, Any]:
    if strategy == "hyde":
        if embeddings is None:
            raise ValueError("embeddings required for HyDE strategy")
        result = hyde_retrieve(query, llm, embeddings, vector_store)
    elif strategy == "multi_query":
        result = multi_query_retrieve(query, llm, vector_store)
    elif strategy == "step_back":
        result = step_back_retrieve(query, llm, vector_store)
    else:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Choose from: hyde, multi_query, step_back"
        )

    context = "\n\n".join(doc.page_content for doc in result.documents)
    prompt = f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {query}"
    answer = llm.invoke(prompt).content

    return {
        "answer": answer,
        "strategy": result.strategy,
        "rewritten_queries": result.rewritten_queries,
        "sources": [doc.metadata for doc in result.documents],
    }

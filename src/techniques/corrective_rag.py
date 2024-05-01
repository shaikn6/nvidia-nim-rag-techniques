"""Technique 5: Corrective RAG (CRAG) via LangGraph.

Workflow:
  retrieve → grade_documents → (if relevant) generate
                             → (if not relevant) web_search → generate

Grades each retrieved doc; if all fail relevance, falls back to web search
(or returns a "no relevant docs" signal when web search is unavailable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

_GRADE_PROMPT = (
    "Is the following document relevant to the question?\n\n"
    "Question: {query}\n\nDocument:\n{doc}\n\n"
    "Answer with exactly one word: yes or no."
)

_GENERATE_PROMPT = (
    "Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {query}"
)


@dataclass
class CRAGState:
    query: str
    documents: list[Document] = field(default_factory=list)
    graded_docs: list[tuple[Document, bool]] = field(default_factory=list)
    used_web_search: bool = False
    answer: str = ""
    steps: list[str] = field(default_factory=list)


def _grade_document(query: str, doc: Document, llm: BaseChatModel) -> bool:
    response = llm.invoke(_GRADE_PROMPT.format(query=query, doc=doc.page_content))
    return response.content.strip().lower().startswith("yes")


def _web_search(query: str) -> list[Document]:
    """Real implementation would call Tavily or SerpAPI. Returns empty list as fallback."""
    try:
        from langchain_community.tools import TavilySearchResults

        import os

        if not os.environ.get("TAVILY_API_KEY"):
            return []
        tool = TavilySearchResults(max_results=3)
        results = tool.invoke(query)
        return [
            Document(page_content=r["content"], metadata={"source": r["url"]})
            for r in results
            if isinstance(r, dict) and "content" in r
        ]
    except Exception:
        return []


def run_corrective_rag(
    query: str,
    vector_store: VectorStore,
    llm: BaseChatModel,
    top_k: int = 5,
    relevance_threshold: float = 0.5,
) -> dict[str, Any]:
    state = CRAGState(query=query)
    state.steps.append("retrieve")

    state.documents = vector_store.similarity_search(query, k=top_k)

    state.steps.append("grade_documents")
    for doc in state.documents:
        relevant = _grade_document(query, doc, llm)
        state.graded_docs.append((doc, relevant))

    relevant_docs = [doc for doc, ok in state.graded_docs if ok]
    relevance_ratio = len(relevant_docs) / max(len(state.graded_docs), 1)

    if relevance_ratio < relevance_threshold:
        state.steps.append("web_search_fallback")
        web_docs = _web_search(query)
        if web_docs:
            relevant_docs = web_docs
            state.used_web_search = True
        elif not relevant_docs:
            relevant_docs = state.documents

    state.steps.append("generate")
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    response = llm.invoke(_GENERATE_PROMPT.format(context=context, query=query))
    state.answer = response.content

    return {
        "answer": state.answer,
        "steps": state.steps,
        "used_web_search": state.used_web_search,
        "relevance_ratio": relevance_ratio,
        "relevant_doc_count": len(relevant_docs),
        "sources": [doc.metadata for doc in relevant_docs],
    }


def build_crag_graph(
    vector_store: VectorStore,
    llm: BaseChatModel,
    top_k: int = 5,
) -> Any:
    """Build a LangGraph StateGraph for CRAG (returns compiled graph)."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise ImportError("pip install langgraph") from exc

    from typing import TypedDict

    class GraphState(TypedDict):
        query: str
        documents: list[Document]
        graded: list[tuple[Document, bool]]
        used_web: bool
        answer: str

    def retrieve_node(state: GraphState) -> GraphState:  # pragma: no cover
        docs = vector_store.similarity_search(state["query"], k=top_k)
        return {**state, "documents": docs}

    def grade_node(state: GraphState) -> GraphState:  # pragma: no cover
        graded = [(d, _grade_document(state["query"], d, llm)) for d in state["documents"]]
        return {**state, "graded": graded}

    def route_node(state: GraphState) -> Literal["generate", "web_search"]:  # pragma: no cover
        ok_ratio = sum(1 for _, ok in state["graded"] if ok) / max(len(state["graded"]), 1)
        return "generate" if ok_ratio >= 0.5 else "web_search"

    def web_search_node(state: GraphState) -> GraphState:  # pragma: no cover
        docs = _web_search(state["query"]) or state["documents"]
        return {**state, "documents": docs, "used_web": True}

    def generate_node(state: GraphState) -> GraphState:  # pragma: no cover
        relevant = [d for d, ok in state.get("graded", []) if ok] or state["documents"]
        context = "\n\n".join(d.page_content for d in relevant)
        ans = llm.invoke(_GENERATE_PROMPT.format(context=context, query=state["query"])).content
        return {**state, "answer": ans}

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", route_node, {"generate": "generate", "web_search": "web_search"}
    )
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

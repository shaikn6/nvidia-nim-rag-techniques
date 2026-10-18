"""Retrieval-quality evaluation: recall@k and MRR against labeled relevance judgments.

The latency benchmark in `benchmark.py` says nothing about whether a technique retrieves
the right passages. This module scores each pipeline's ranked `sources` against the
qrels in `eval/data/tiny_ir.json`. Only retrieval is scored: the generation step is a
no-op stub, so no LLM (and no NVIDIA NIM key) is needed.

Run with `python -m src.pipeline.retrieval_eval` to embed with a local open model
(BAAI/bge-small-en-v1.5), rerank with the repo's default cross-encoder, and write
`eval/retrieval_results.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.core.vector_store import build_faiss_store
from src.pipeline.base import BaseRAGPipeline
from src.pipeline.pipelines import (
    CompressionPipeline,
    HybridSearchPipeline,
    NaiveRAGPipeline,
    RerankingPipeline,
)
from src.techniques.compression import build_embeddings_filter_retriever
from src.techniques.hybrid_search import HybridRetriever
from src.techniques.reranking import RerankerRetriever

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = _REPO_ROOT / "eval" / "data" / "tiny_ir.json"
RESULTS_PATH = _REPO_ROOT / "eval" / "retrieval_results.json"

KS = (1, 3, 5)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # the repo's default cross-encoder

# Pipelines whose retrieval step needs a real LLM; scoring them with a stub would be
# meaningless, so they are reported as skipped rather than given a number.
SKIPPED = {
    "query_rewriting_hyde": "HyDE retrieves with an LLM-written hypothetical answer; needs a real LLM.",
    "query_rewriting_multi_query": "Retrieves with LLM-generated query variants; needs a real LLM.",
    "query_rewriting_step_back": "Retrieves with an LLM-generated step-back question; needs a real LLM.",
    "corrective_rag": "LLM grades retrieved docs and falls back to Tavily web search (not a corpus ranking).",
}


class RetrievalOnlyLLM:
    """Generation stub: the pipelines call `invoke` after retrieval, but only `sources` is scored."""

    def invoke(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content="")


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain Embeddings over any object with a sentence-transformers style `encode`."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def load_dataset(path: Path = DATASET_PATH) -> tuple[list[Document], list[dict[str, Any]]]:
    data = json.loads(path.read_text())
    docs = [Document(page_content=d["text"], metadata={"source": d["id"]}) for d in data["corpus"]]
    return docs, data["queries"]


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    return len(relevant & set(ranked_ids[:k])) / len(relevant)


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _ranked_ids(pipeline: BaseRAGPipeline, query: str) -> list[str]:
    ids = [src["source"] for src in pipeline.run(query)["sources"]]
    return list(dict.fromkeys(ids))  # a doc counts once, at its best rank


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: round(_mean([r[key] for r in rows]), 4) for key in rows[0]}


def evaluate_pipeline(pipeline: BaseRAGPipeline, queries: list[dict[str, Any]]) -> dict[str, Any]:
    per_query: list[tuple[str, dict[str, float]]] = []
    for q in queries:
        ranked = _ranked_ids(pipeline, q["text"])
        relevant = set(q["relevant"])
        scores = {f"recall@{k}": recall_at_k(ranked, relevant, k) for k in KS}
        scores["mrr"] = reciprocal_rank(ranked, relevant)
        per_query.append((q["type"], scores))

    result: dict[str, Any] = _aggregate([s for _, s in per_query])
    result["by_query_type"] = {
        kind: {**_aggregate([s for t, s in per_query if t == kind]),
               "num_queries": sum(1 for t, _ in per_query if t == kind)}
        for kind in sorted({t for t, _ in per_query})
    }
    return result


def build_pipelines(docs: list[Document], embeddings: Embeddings, reranker: Any) -> list[BaseRAGPipeline]:
    """The pipelines whose retrieval step runs without an LLM, all over one shared index."""
    llm = RetrievalOnlyLLM()
    store = build_faiss_store(docs, embeddings)
    return [
        NaiveRAGPipeline(store, llm),
        HybridSearchPipeline(HybridRetriever(store, docs), llm),
        RerankingPipeline(RerankerRetriever(store, reranker), llm),
        CompressionPipeline(build_embeddings_filter_retriever(store, embeddings), llm),
    ]


def run_eval(
    docs: list[Document],
    queries: list[dict[str, Any]],
    embeddings: Embeddings,
    reranker: Any,
) -> dict[str, Any]:
    pipelines = build_pipelines(docs, embeddings, reranker)
    return {
        "num_passages": len(docs),
        "num_queries": len(queries),
        "pipelines": {p.name: evaluate_pipeline(p, queries) for p in pipelines},
        "skipped": SKIPPED,
    }


def main() -> None:  # pragma: no cover - loads real models; exercised by running the module
    import platform

    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    from src.techniques.reranking import CrossEncoderReranker

    docs, queries = load_dataset()
    reranker = CrossEncoderReranker(model_name=RERANKER_MODEL)
    embeddings = SentenceTransformerEmbeddings(SentenceTransformer(EMBEDDING_MODEL))

    results = run_eval(docs, queries, embeddings, reranker)
    results["dataset"] = str(DATASET_PATH.relative_to(_REPO_ROOT))
    results["models"] = {
        "embeddings": EMBEDDING_MODEL,
        "reranker": RERANKER_MODEL,
        "sentence_transformers": sentence_transformers.__version__,
        "python": platform.python_version(),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")

    print(f"{len(docs)} passages, {len(queries)} queries -> {RESULTS_PATH.relative_to(_REPO_ROOT)}\n")
    header = f"{'pipeline':>16}  " + "  ".join(f"recall@{k}" for k in KS) + "     mrr"
    print(header)
    for name, m in results["pipelines"].items():
        print(f"{name:>16}  " + "  ".join(f"{m[f'recall@{k}']:8.3f}" for k in KS) + f"  {m['mrr']:6.3f}")


if __name__ == "__main__":  # pragma: no cover
    main()

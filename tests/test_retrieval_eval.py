"""Retrieval-quality harness: metric maths plus an end-to-end run on the tiny labeled set."""

from __future__ import annotations

import json

import numpy as np

from src.pipeline.pipelines import DemoEmbeddings, DemoReranker
from src.pipeline.retrieval_eval import (
    DATASET_PATH,
    KS,
    SKIPPED,
    RetrievalOnlyLLM,
    SentenceTransformerEmbeddings,
    load_dataset,
    recall_at_k,
    reciprocal_rank,
    run_eval,
)


def test_recall_at_k():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"c"}, 1) == 0.0
    assert recall_at_k(ranked, {"c"}, 3) == 1.0
    assert recall_at_k(ranked, {"a", "d"}, 3) == 0.5
    assert recall_at_k(ranked, {"z"}, 4) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"c", "b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_dataset_qrels_reference_existing_passages():
    docs, queries = load_dataset()
    ids = {d.metadata["source"] for d in docs}
    assert len(ids) == len(docs) >= 30
    assert len(queries) >= 15
    for q in queries:
        assert q["relevant"] and set(q["relevant"]) <= ids
    assert json.loads(DATASET_PATH.read_text())["license"]


def test_eval_runs_on_tiny_set_with_sane_metrics():
    docs, queries = load_dataset()
    out = run_eval(docs, queries, DemoEmbeddings(), DemoReranker())

    assert out["num_passages"] == len(docs)
    assert out["num_queries"] == len(queries)
    assert set(out["pipelines"]) == {"naive_baseline", "hybrid_search", "reranking", "compression"}
    assert "corrective_rag" in out["skipped"] and out["skipped"] == SKIPPED

    for metrics in out["pipelines"].values():
        recalls = [metrics[f"recall@{k}"] for k in KS]
        assert all(0.0 <= r <= 1.0 for r in recalls)
        assert recalls == sorted(recalls)  # recall can only grow with k
        assert 0.0 <= metrics["mrr"] <= 1.0
        assert metrics["mrr"] >= metrics["recall@1"]  # reciprocal rank >= hit@1 >= recall@1
        assert sum(t["num_queries"] for t in metrics["by_query_type"].values()) == len(queries)
        # hashed bag-of-words still finds something on a lexically easy set
        assert metrics["recall@5"] > 0.0


def test_sentence_transformer_embeddings_adapter():
    class FakeModel:
        def encode(self, texts, normalize_embeddings):
            assert normalize_embeddings
            return np.array([[float(len(t)), 1.0] for t in texts])

    emb = SentenceTransformerEmbeddings(FakeModel())
    assert emb.embed_documents(["ab", "abc"]) == [[2.0, 1.0], [3.0, 1.0]]
    assert emb.embed_query("abcd") == [4.0, 1.0]


def test_stub_llm_returns_empty_content():
    assert RetrievalOnlyLLM().invoke("anything").content == ""


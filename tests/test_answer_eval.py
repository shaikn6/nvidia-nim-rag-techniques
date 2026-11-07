"""Answer-overlap proxy: metric maths plus scoring the demo pipelines on the labeled set."""

from __future__ import annotations

import pytest

from src.pipeline.answer_eval import answer_overlap_proxy, content_tokens
from src.pipeline.pipelines import DemoEmbeddings, DemoReranker
from src.pipeline.retrieval_eval import load_dataset, run_eval


def test_content_tokens_lowercases_and_drops_stopwords():
    assert content_tokens("The VACUUM command, and a B-tree!") == {"vacuum", "command", "b", "tree"}
    assert content_tokens("the an of to") == set()


def test_answer_overlap_proxy_math():
    # every content word of the reference appears in the answer -> 1.0
    assert answer_overlap_proxy("vacuum reclaims dead tuples", "vacuum reclaims dead tuples") == 1.0
    # two of the four reference content words appear -> 0.5
    assert answer_overlap_proxy("vacuum reclaims nothing here", "vacuum reclaims dead tuples") == 0.5
    # no overlap -> 0.0
    assert answer_overlap_proxy("kubernetes pods", "vacuum reclaims dead tuples") == 0.0


def test_answer_overlap_proxy_empty_reference_is_zero():
    assert answer_overlap_proxy("anything at all", "") == 0.0
    assert answer_overlap_proxy("anything", "the an of to") == 0.0  # only stopwords


def test_answer_overlap_proxy_ignores_stopwords_in_reference():
    # stopwords in the reference are not counted, so adding them does not change the score
    assert answer_overlap_proxy("vacuum", "the vacuum") == 1.0


@pytest.mark.parametrize("answer,reference", [("a b c", "a b"), ("x", "x y z")])
def test_answer_overlap_proxy_in_unit_range(answer, reference):
    assert 0.0 <= answer_overlap_proxy(answer, reference) <= 1.0


def test_demo_pipeline_answers_scored_against_labeled_set():
    """End-to-end, fully offline: the demo pipelines answer the labeled queries and the
    proxy scores each answer against its gold passage(s)."""
    docs, queries = load_dataset()
    out = run_eval(docs, queries, DemoEmbeddings(), DemoReranker())

    for name, metrics in out["pipelines"].items():
        proxy = metrics["answer_overlap_proxy"]
        assert 0.0 < proxy <= 1.0, f"{name} proxy out of range: {proxy}"

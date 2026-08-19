"""Tests for the bundled demo corpus/query set."""

from __future__ import annotations

from langchain_core.documents import Document

from src.pipeline.demo_data import DEMO_CORPUS, DEMO_QUERIES


class TestDemoData:
    def test_corpus_is_nonempty(self):
        assert len(DEMO_CORPUS) >= 5

    def test_corpus_entries_are_documents_with_source_metadata(self):
        for doc in DEMO_CORPUS:
            assert isinstance(doc, Document)
            assert "source" in doc.metadata

    def test_queries_is_nonempty(self):
        assert len(DEMO_QUERIES) >= 1

    def test_queries_are_strings(self):
        assert all(isinstance(q, str) and q for q in DEMO_QUERIES)

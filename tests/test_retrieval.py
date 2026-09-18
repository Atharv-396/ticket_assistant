"""Tests for RAG retrieval — chunking, cosine similarity, retrieval."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts.ingest_knowledge_base import _chunk_text, _hash_text
from src.retrieval import (
    RetrievedChunk, _cosine_similarity, format_chunks_for_prompt,
    invalidate_cache, retrieve_relevant_chunks,
)


# ── Chunking ──────────────────────────────────────────────────────────────────

def test_chunk_basic():
    text = " ".join([f"w{i}" for i in range(100)])
    chunks = _chunk_text(text, chunk_size=30)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunk_short_doc_gives_one():
    assert len(_chunk_text("Short doc.", chunk_size=300)) == 1


def test_chunk_empty_string():
    assert _chunk_text("") == []


def test_chunk_overlap_increases_total_words():
    text = " ".join([f"w{i}" for i in range(60)])
    chunks = _chunk_text(text, chunk_size=20)
    total = sum(len(c.split()) for c in chunks)
    assert total >= 60


# ── Cosine similarity ─────────────────────────────────────────────────────────

def test_cosine_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert _cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_zero_vector():
    assert _cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 2.0])) == pytest.approx(0.0)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _make_store(tmp_path: Path) -> Path:
    chunks = [
        {"source": "damaged_goods.md", "text": "Damaged goods policy.", "embedding": [1.0, 0.0, 0.0], "doc_hash": "a"},
        {"source": "shipping.md",      "text": "Shipping delay policy.", "embedding": [0.0, 1.0, 0.0], "doc_hash": "b"},
    ]
    p = tmp_path / "embeddings.json"
    p.write_text(json.dumps(chunks), encoding="utf-8")
    return p


def test_retrieval_returns_most_similar(tmp_path):
    store = _make_store(tmp_path)
    with (
        patch("src.retrieval.settings.EMBEDDINGS_PATH", store),
        patch("src.services.gemini.embed_text", return_value=[1.0, 0.05, 0.0]),
    ):
        invalidate_cache()
        results = retrieve_relevant_chunks("damaged product", top_k=1)
    assert results[0].source == "damaged_goods.md"
    assert results[0].score > 0.9


def test_retrieval_preserves_source_names(tmp_path):
    store = _make_store(tmp_path)
    with (
        patch("src.retrieval.settings.EMBEDDINGS_PATH", store),
        patch("src.services.gemini.embed_text", return_value=[0.5, 0.5, 0.0]),
    ):
        invalidate_cache()
        results = retrieve_relevant_chunks("issue", top_k=2)
    sources = {r.source for r in results}
    assert "damaged_goods.md" in sources
    assert "shipping.md" in sources


def test_retrieval_top_k_limits(tmp_path):
    store = _make_store(tmp_path)
    with (
        patch("src.retrieval.settings.EMBEDDINGS_PATH", store),
        patch("src.services.gemini.embed_text", return_value=[1.0, 0.0, 0.0]),
    ):
        invalidate_cache()
        results = retrieve_relevant_chunks("test", top_k=1)
    assert len(results) == 1


def test_format_chunks_for_prompt():
    chunks = [
        RetrievedChunk(source="refunds.md", text="Refund policy text.", score=0.9),
    ]
    out = format_chunks_for_prompt(chunks)
    assert "refunds.md" in out
    assert "Refund policy text." in out


def test_format_empty_chunks():
    assert "No relevant" in format_chunks_for_prompt([])


def test_kb_files_discoverable():
    from src.config import settings
    kb = settings.KB_DIR
    assert kb.exists(), f"KB not found: {kb}"
    files = list(kb.glob("*.md"))
    assert len(files) >= 4
    names = {f.name for f in files}
    assert "damaged_goods.md" in names
    assert "returns.md" in names
    assert "shipping.md" in names

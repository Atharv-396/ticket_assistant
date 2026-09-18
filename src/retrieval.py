"""
Knowledge-base retrieval using cosine similarity.

How it works:
  1. The ingestion script (scripts/ingest_knowledge_base.py) reads the .md
     policy files, splits them into chunks, embeds each chunk with Gemini,
     and persists everything to data/embeddings.json.

  2. At query time this module:
       a. Loads the persisted chunks from disk (cached in memory after first load).
       b. Embeds the incoming ticket text as a query vector.
       c. Computes cosine similarity between the query and every chunk vector.
       d. Returns the top-k most similar chunks.

Why cosine similarity?
  It measures the angle between two vectors regardless of their magnitude,
  which works well for semantic embeddings where direction encodes meaning.

Why NumPy?
  Fast vectorised dot products with no extra framework required.
  The formula:  cos(θ) = (A · B) / (‖A‖ · ‖B‖)

Why a JSON file instead of a vector DB?
  The knowledge base is tiny (< 20 chunks).  A JSON file is transparent,
  portable, and trivially explainable in an interview.
"""
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    source: str        # filename of the policy document, e.g. "returns.md"
    text: str          # chunk content
    score: float       # cosine similarity 0.0–1.0


# ── Embedding store ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_chunks() -> list[dict]:
    """
    Load persisted chunk embeddings from disk.  The result is cached so we
    only read the file once per process lifetime.

    Raises FileNotFoundError if the ingestion script has not been run yet.
    """
    path: Path = settings.EMBEDDINGS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found at {path}. "
            "Run:  python scripts/ingest_knowledge_base.py"
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def invalidate_cache() -> None:
    """Clear the in-memory chunk cache (useful after re-ingestion)."""
    _load_chunks.cache_clear()


# ── Cosine similarity ─────────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve_relevant_chunks(
    ticket_text: str,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Return the *top_k* most relevant knowledge-base chunks for *ticket_text*.

    Steps:
      1. Embed *ticket_text* with Gemini (query embedding).
      2. Load persisted chunk embeddings.
      3. Compute cosine similarity for each chunk.
      4. Sort descending and return top_k.

    Parameters
    ----------
    ticket_text : str
        The support ticket message to match against the knowledge base.
    top_k : int, optional
        How many chunks to return.  Defaults to settings.RETRIEVAL_TOP_K.

    Returns
    -------
    list[RetrievedChunk]
        Ordered from most to least relevant.
    """
    from src.services.gemini import embed_text  # imported here for easy mocking

    k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K

    chunks = _load_chunks()
    if not chunks:
        logger.warning("No chunks found in embeddings store — retrieval will be empty")
        return []

    # Embed the query
    query_vec = np.array(embed_text(ticket_text), dtype=np.float32)

    # Score every chunk
    scored: list[tuple[float, dict]] = []
    for chunk in chunks:
        chunk_vec = np.array(chunk["embedding"], dtype=np.float32)
        score = _cosine_similarity(query_vec, chunk_vec)
        scored.append((score, chunk))

    # Sort descending by similarity
    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        RetrievedChunk(
            source=chunk["source"],
            text=chunk["text"],
            score=round(score, 4),
        )
        for score, chunk in scored[:k]
    ]


def format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a single string for inclusion in the
    Gemini prompt.  Each chunk is labelled with its source filename.
    """
    if not chunks:
        return "No relevant policy excerpts found."
    parts = []
    for chunk in chunks:
        parts.append(f"### Source: {chunk.source}\n{chunk.text}")
    return "\n\n".join(parts)

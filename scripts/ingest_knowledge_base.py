"""
Knowledge-base ingestion script.

Usage
-----
    python scripts/ingest_knowledge_base.py

What it does
────────────
  1. Discovers all .md files in candidate_pack/knowledge_base/.
  2. Splits each document into overlapping word-based chunks.
  3. Embeds each chunk using the Gemini text-embedding-004 model.
  4. Persists chunks + embeddings to data/embeddings.json.
  5. Prints a progress summary.

Re-ingestion
────────────
  Running the script again overwrites the existing embeddings.json.
  A simple content-hash check avoids re-embedding unchanged files.
  To force full re-embedding, delete data/embeddings.json first.

Why word-based chunking?
  Simple, deterministic, and easy to explain.  Sentence-based splitting
  would be more accurate but adds NLTK as a dependency.  For a small
  policy knowledge base (~6 documents, < 30 sentences each) word windows
  work well.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

# Allow running as  python scripts/ingest_knowledge_base.py  from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.services.gemini import embed_document_chunk


def _hash_text(text: str) -> str:
    """MD5 fingerprint of document text — used to skip unchanged files."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _chunk_text(text: str, chunk_size: int = settings.CHUNK_SIZE) -> list[str]:
    """
    Split *text* into overlapping word-window chunks.

    chunk_size  words per chunk  (~300 words ≈ ~400 tokens for English)
    overlap     20% of chunk_size to preserve sentence context at boundaries
    """
    words = text.split()
    if not words:
        return []

    overlap = max(1, chunk_size // 5)  # 20% overlap
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap  # slide forward with overlap

    return chunks


def load_existing_embeddings(path: Path) -> list[dict]:
    """Load existing embeddings from disk (returns [] if file doesn't exist)."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return []


def ingest() -> None:
    kb_dir: Path = settings.KB_DIR
    out_path: Path = settings.EMBEDDINGS_PATH

    # ── Validate KB directory ─────────────────────────────────────────────────
    if not kb_dir.exists():
        print(f"ERROR: Knowledge-base directory not found: {kb_dir}", file=sys.stderr)
        sys.exit(1)

    md_files = sorted(kb_dir.glob("*.md"))
    if not md_files:
        print(f"ERROR: No .md files found in {kb_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(md_files)} document(s) in {kb_dir}")
    for f in md_files:
        print(f"  - {f.name}")

    # ── Load existing embeddings for incremental update ───────────────────────
    existing = load_existing_embeddings(out_path)
    # Build a lookup: (source, text) → embedding, so we can skip unchanged chunks
    existing_lookup: dict[str, list[float]] = {
        f"{e['source']}::{e['text']}": e["embedding"] for e in existing
    }
    # Track which source hashes we've seen before
    existing_hashes: dict[str, str] = {
        e["source"]: e.get("doc_hash", "") for e in existing
    }

    # ── Process documents ─────────────────────────────────────────────────────
    all_chunks: list[dict] = []
    total_new = 0

    for md_file in md_files:
        source_name = md_file.name
        text = md_file.read_text(encoding="utf-8").strip()
        doc_hash = _hash_text(text)

        chunks = _chunk_text(text)
        print(f"\n{source_name}: {len(chunks)} chunk(s)", end="", flush=True)

        for chunk_text in chunks:
            cache_key = f"{source_name}::{chunk_text}"
            if cache_key in existing_lookup and existing_hashes.get(source_name) == doc_hash:
                # Reuse existing embedding — no API call needed
                embedding = existing_lookup[cache_key]
                print(".", end="", flush=True)
            else:
                # New or changed chunk — embed it
                try:
                    embedding = embed_document_chunk(chunk_text)
                    total_new += 1
                    print("E", end="", flush=True)
                    # Small delay to stay within free-tier rate limits
                    time.sleep(0.1)
                except Exception as exc:
                    print(f"\nERROR embedding chunk from {source_name}: {exc}", file=sys.stderr)
                    sys.exit(1)

            all_chunks.append(
                {
                    "source": source_name,
                    "text": chunk_text,
                    "embedding": embedding,
                    "doc_hash": doc_hash,
                }
            )

    print()  # newline after progress dots

    # ── Persist ───────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh)

    print(f"\n✓ Stored {len(all_chunks)} chunk(s) ({total_new} newly embedded)")
    print(f"  → {out_path}")


if __name__ == "__main__":
    ingest()

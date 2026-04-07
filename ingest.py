#!/usr/bin/env python3
"""
PDF ingestion pipeline: ./source_docs -> chunk -> embed (MiniLM) -> ChromaDB ./db
Uses PyMuPDF for text extraction and recursive character splitting for context windows.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- Paths (relative to script / cwd) ---
SOURCE_DIR = Path(__file__).resolve().parent / "source_docs"
DB_DIR = Path(__file__).resolve().parent / "db"
COLLECTION_NAME = "meeting_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_pdf_texts(source: Path) -> list[tuple[str, str]]:
    """Extract plain text from each PDF under source (recursive). Returns [(path_str, text), ...]."""
    if not source.is_dir():
        source.mkdir(parents=True, exist_ok=True)
        print(f"Created empty folder: {source}")
        return []

    pairs: list[tuple[str, str]] = []
    for pdf_path in sorted(source.rglob("*.pdf")):
        try:
            doc = fitz.open(pdf_path)
            parts: list[str] = []
            for page in doc:
                parts.append(page.get_text("text") or "")
            doc.close()
            full = "\n\n".join(parts).strip()
            if full:
                pairs.append((str(pdf_path.resolve()), full))
            else:
                print(f"Warning: no text extracted from {pdf_path.name}")
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}", file=sys.stderr)
    return pairs


def chunk_text(text: str) -> list[str]:
    """Recursive character split (chunk_size / overlap per spec)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    pieces = splitter.split_text(text)
    return [c for c in pieces if c.strip()]


def stable_id(source: str, chunk_index: int, content: str) -> str:
    """Deterministic ID for re-ingestion (content-hash avoids collisions)."""
    h = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"{Path(source).stem}_{chunk_index}_{h}"


def main() -> None:
    print(f"Embedding model: {EMBED_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=device)
    if device == "cpu":
        print("Note: embeddings on CPU. Install CUDA-enabled PyTorch for GPU speedup.")

    file_texts = load_pdf_texts(SOURCE_DIR)
    if not file_texts:
        print(f"No PDFs with text in {SOURCE_DIR}. Add .pdf files and run again.")
        return

    all_chunks: list[str] = []
    all_metas: list[dict] = []
    all_ids: list[str] = []

    for path_str, text in file_texts:
        pieces = chunk_text(text)
        for i, chunk in enumerate(pieces):
            meta = {"source": Path(path_str).name, "chunk_index": i}
            all_chunks.append(chunk)
            all_metas.append(meta)
            all_ids.append(stable_id(path_str, i, chunk))

    print(f"Encoding {len(all_chunks)} chunks...")
    embeddings = model.encode(
        all_chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine-friendly; matches Chroma cosine space
    ).tolist()

    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    # Replace collection contents for a clean snapshot (simplest UX for "re-run ingest")
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(ids=all_ids, documents=all_chunks, metadatas=all_metas, embeddings=embeddings)
    print(f"Stored {len(all_ids)} vectors in {DB_DIR} (collection '{COLLECTION_NAME}').")


if __name__ == "__main__":
    main()

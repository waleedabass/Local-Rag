#!/usr/bin/env python3
"""
FastAPI RAG server: Chroma top-3 retrieval + Ollama (llama3.2:3b).
Skips all inference when is_active=False to avoid GPU/CPU work per request.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import chromadb
import httpx
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths & models (keep in sync with ingest.py)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DB_DIR = ROOT / "db"
STATIC_DIR = ROOT / "static"
COLLECTION_NAME = "meeting_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
# Whisper (lazy-loaded on first /transcribe). See https://github.com/SYSTRAN/faster-whisper
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "")  # empty = auto (cuda if available)
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "")  # empty = float16 on cuda, int8 on cpu
WHISPER_LANG = os.environ.get("WHISPER_LANG", "")  # e.g. en; empty = auto-detect

SYSTEM_PROMPT = (
    "You are a professional meeting assistant. Using ONLY the provided context, "
    "answer the user's question in 3-5 concise bullet points. If the answer isn't in the context, "
    "say 'Information not found in documents.' Keep it brief for live reading."
)

# Globals set in lifespan (loaded once for low latency)
_embed_model: SentenceTransformer | None = None
_chroma_collection: Any = None
_whisper_model: Any = None
_whisper_lock = threading.Lock()


def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _whisper_compute_type(device: str) -> str:
    if WHISPER_COMPUTE_TYPE:
        return WHISPER_COMPUTE_TYPE
    return "float16" if device == "cuda" else "int8"


def _get_whisper_model() -> Any:
    """Load faster-whisper once (thread-safe). Kept separate from chat startup cost."""
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel

            device = WHISPER_DEVICE or _pick_device()
            compute_type = _whisper_compute_type(device)
            _whisper_model = WhisperModel(
                WHISPER_MODEL,
                device=device,
                compute_type=compute_type,
            )
        return _whisper_model


def transcribe_audio_path(path: str) -> str:
    """Run Whisper on a file path (blocking; call via asyncio.to_thread)."""
    model = _get_whisper_model()
    lang = WHISPER_LANG.strip() or None
    segments, _info = model.transcribe(
        path,
        language=lang,
        beam_size=5,
        vad_filter=True,
    )
    parts: list[str] = []
    for seg in segments:
        t = seg.text.strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embed_model, _chroma_collection
    device = _pick_device()
    # Single load at startup: amortizes model load cost across requests
    _embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    _chroma_collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    yield
    _embed_model = None
    _chroma_collection = None
    global _whisper_model
    _whisper_model = None


app = FastAPI(title="Meeting RAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """User query and live-mode gate."""

    query: str = Field(..., description="Transcription snippet or question")
    is_active: bool = Field(True, description="If false, skip retrieval and LLM")


class ChatResponse(BaseModel):
    response: str = Field("", description="Bullet-point answer; empty when live mode off")
    chunks_used: int = Field(0, description="Number of context chunks retrieved")


class TranscribeResponse(BaseModel):
    text: str = Field("", description="Whisper transcript")


def retrieve_top_k(query: str, k: int = 3) -> tuple[list[str], int]:
    """Embed query and return top-k document texts from Chroma (cosine)."""
    assert _embed_model is not None and _chroma_collection is not None
    q = query.strip()
    if not q:
        return [], 0

    # normalize_embeddings=True matches ingest.py for cosine similarity
    emb = _embed_model.encode(
        [q],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    res = _chroma_collection.query(
        query_embeddings=emb,
        n_results=k,
        include=["documents"],
    )
    docs = (res.get("documents") or [[]])[0]
    docs = [d for d in docs if d]
    return docs, len(docs)


def call_ollama(user_message: str, timeout_s: float = 120.0) -> str:
    """Non-streaming chat completion from local Ollama."""
    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        # Lower temperature for steadier meeting notes
        "options": {"temperature": 0.2, "num_ctx": 4096},
    }
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(OLLAMA_CHAT_URL, json=payload)
        r.raise_for_status()
        data = r.json()
    msg = (data.get("message") or {}).get("content")
    return (msg or "").strip()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    When is_active is False, return immediately (no embedding, DB, or LLM).
    When True, retrieve top-3 chunks and answer via Ollama.
    """
    if not req.is_active:
        return ChatResponse(response="", chunks_used=0)

    q = req.query.strip()
    if not q:
        return ChatResponse(response="", chunks_used=0)

    chunks, n = retrieve_top_k(q, k=3)
    if not chunks:
        return ChatResponse(
            response="Information not found in documents.",
            chunks_used=0,
        )

    context_block = "\n\n---\n\n".join(chunks)
    user_message = f"Context:\n{context_block}\n\nQuestion:\n{q}"
    try:
        llm_text = call_ollama(user_message)
    except httpx.HTTPError as e:
        return ChatResponse(
            response=f"LLM error: {e!s}. Is Ollama running with `{OLLAMA_MODEL}`?",
            chunks_used=n,
        )

    return ChatResponse(response=llm_text, chunks_used=n)


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscribeResponse:
    """
    Local Whisper transcription (faster-whisper). Upload recorded audio (e.g. webm from the browser).
    Requires FFmpeg on PATH. First call downloads/loads the model (see WHISPER_MODEL).
    """
    raw = await file.read()
    if len(raw) < 256:
        raise HTTPException(status_code=400, detail="Audio too short or empty.")

    suffix = Path(file.filename or "audio").suffix.lower()
    if suffix not in {".webm", ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4", ".mpeg", ".mpga", ".oga"}:
        suffix = ".webm"

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        text = await asyncio.to_thread(transcribe_audio_path, tmp_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {e!s}. Is FFmpeg installed and on PATH?",
        ) from e
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return TranscribeResponse(text=text)


# Single-page UI without shadowing /docs or /openapi.json
STATIC_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def serve_ui():
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="static/index.html missing")
    return FileResponse(index)

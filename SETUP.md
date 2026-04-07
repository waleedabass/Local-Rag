# Local PDF RAG Assistant — Setup Guide

This project runs a **local** meeting assistant: PDFs in `source_docs` are embedded with **sentence-transformers/all-MiniLM-L6-v2**, stored in **ChromaDB** (`./db`), and answers are generated with **Ollama** using **llama3.2:3b**. A **FastAPI** server exposes `/chat`, **`/transcribe`** (local **Whisper** via faster-whisper), and serves a small HTML UI at `/`.

---

## What you need

| Requirement | Notes |
|-------------|--------|
| **Windows 10/11** (this repo) | `start.bat` is for Windows; on Linux/macOS run the same Python commands manually. |
| **Python 3.10+** | 64-bit recommended. Check with `python --version`. |
| **Ollama** | [https://ollama.com](https://ollama.com) — must be installed and running locally. |
| **NVIDIA GPU (optional)** | RTX 2060 6GB works; embeddings and the LLM benefit from GPU. CPU works but is slower. |
| **~32GB RAM** | Comfortable for MiniLM + 3B model; less RAM may still work with smaller batches. |
| **FFmpeg** | Required on **PATH** for Whisper to decode browser recordings (WebM, etc.). Install from [ffmpeg.org](https://ffmpeg.org/download.html) or `winget install FFmpeg`. |

---

## 1. Install Ollama and pull the chat model

1. Install Ollama from the official site and start it (it usually runs as a background service).
2. In a terminal, pull the model used by the app:

   ```bash
   ollama pull llama3.2:3b
   ```

3. Confirm Ollama is listening (default): `http://127.0.0.1:11434`.

If Ollama runs on another host or port, set environment variables before starting the server (see [Environment variables](#environment-variables)).

---

## 2. Python virtual environment (recommended)

From the project root (`LocalRag`):

```powershell
cd e:\LocalRag
python -m venv .venv
.\.venv\Scripts\activate
```

`start.bat` automatically activates `.venv` if it exists, so creating the venv is optional but recommended.

---

## 3. Install PyTorch (GPU strongly recommended)

`requirements.txt` pins `torch>=2.0.0`, which often installs a **CPU** wheel from PyPI. On an RTX 2060 you should install a **CUDA** build of PyTorch **first**, then install the rest of the requirements (so pip does not downgrade torch incorrectly).

1. Open [PyTorch — Get Started](https://pytorch.org/get-started/locally/).
2. Choose your OS, **CUDA** version that matches your NVIDIA driver, and **pip**.
3. Run the command PyTorch gives you (example shape only — use the site’s exact command):

   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

4. Then install project dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

If you skip this and use CPU-only PyTorch, the app still runs; embedding and generation will be slower.

---

## 4. Add your PDFs

1. Create or use the folder **`source_docs`** in the project root (same folder as `ingest.py`).
2. Copy **`.pdf`** files there. Subfolders are supported; ingestion scans recursively.

The first run of `ingest.py` can create `source_docs` if it is missing (empty folder → no chunks until you add PDFs).

---

## 5. Run ingestion and the server

### Option A — One click (Windows)

Double-click or run:

```powershell
.\start.bat
```

This script:

1. Activates `.venv` if present.
2. Runs `python ingest.py` (rebuilds the Chroma collection from PDFs).
3. Starts **uvicorn** at `http://127.0.0.1:8000`.

### Option B — Manual steps

```powershell
cd e:\LocalRag
.\.venv\Scripts\activate
python ingest.py
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

---

## 6. Use the app

1. Open a browser: **http://127.0.0.1:8000**
2. **Live Mode** toggle:
   - **On** (`is_active: true`): runs embedding search + Ollama (normal Q&A).
   - **Off**: returns an **empty** response immediately to save GPU/CPU when you are not actively asking questions.
3. Enter text in the box (type, paste) or use **Record (Whisper)**: the browser records audio, uploads it to **`/transcribe`**, and the server runs **local Whisper** (first use downloads the model). Allow the microphone when prompted.
4. Click **Ask** to run RAG on the current text.

Interactive API docs: **http://127.0.0.1:8000/docs**

---

## 7. API quick reference

**POST** `/chat`

JSON body:

```json
{
  "query": "What did we decide about the budget?",
  "is_active": true
}
```

- If `is_active` is `false`, the server skips retrieval and the LLM and returns an empty `response` (and no chunk usage).
- If nothing relevant is in the index, the model is instructed to answer with *Information not found in documents.*

**POST** `/transcribe` — multipart form field **`file`** (audio: WebM, WAV, MP3, etc.). Returns JSON `{"text": "..."}`. Requires FFmpeg on PATH.

---

## 8. Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model name for `/api/chat` |
| `WHISPER_MODEL` | `base` | Whisper model size: `tiny`, `base`, `small`, … (VRAM vs quality tradeoff) |
| `WHISPER_DEVICE` | *(auto)* | `cuda` or `cpu` (defaults to CUDA if PyTorch sees a GPU) |
| `WHISPER_COMPUTE_TYPE` | *(auto)* | e.g. `float16` (GPU) or `int8` (CPU); override if load fails |
| `WHISPER_LANG` | *(auto)* | Optional ISO code (e.g. `en`) to skip language detection |

**PowerShell example** (same session as uvicorn):

```powershell
$env:OLLAMA_HOST = "http://127.0.0.1:11434"
$env:OLLAMA_MODEL = "llama3.2:3b"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

---

## 9. Project layout (reference)

| Path | Role |
|------|------|
| `source_docs/` | Input PDFs (recursive `*.pdf`) |
| `db/` | Persistent Chroma data (created by ingestion) |
| `ingest.py` | PDF → chunks → embeddings → Chroma |
| `app.py` | FastAPI app, retrieval, Ollama call |
| `static/index.html` | Web UI |
| `requirements.txt` | Python dependencies |
| `start.bat` | Ingest + uvicorn on Windows |

---

## 10. After you change PDFs

Re-run ingestion whenever documents change:

```powershell
python ingest.py
```

Current behavior: ingestion **rebuilds** the `meeting_docs` collection so the index matches the files on disk. Restarting uvicorn is not strictly required after ingest if the server already loaded Chroma from `./db`, but if you see stale behavior, restart the server.

---

## 11. Troubleshooting

| Problem | What to check |
|---------|----------------|
| **Connection refused to Ollama** | Ollama is running; `OLLAMA_HOST` matches where it listens. |
| **Model not found** | Run `ollama pull llama3.2:3b` (or set `OLLAMA_MODEL` to a model you have). |
| **Empty answers with Live Mode on** | PDFs in `source_docs`; run `ingest.py` without errors; check `db/` exists and has data. |
| **Scanned PDFs, no text** | PyMuPDF extracts text layers only; OCR’d PDFs or images need OCR first. |
| **Slow responses** | Install CUDA PyTorch; ensure GPU is used (`torch.cuda.is_available()` in Python). LLM latency dominates; smaller prompts and fewer chunks help (this app already uses top-3 chunks). |
| **Port 8000 in use** | Run uvicorn on another port: `python -m uvicorn app:app --host 127.0.0.1 --port 8001` |
| **Transcription failed / FFmpeg** | Install FFmpeg and ensure `ffmpeg` works in a terminal; restart the app. |
| **Whisper out of VRAM** | Set `WHISPER_MODEL=tiny` or `WHISPER_DEVICE=cpu` (slower). |

---

## 12. Sub-2 second responses

End-to-end latency is mostly **Ollama generation** (3B on 6GB VRAM) plus network stack. This project keeps retrieval light (one small query embedding, **top 3** chunks). For the best chance of staying under ~2 seconds on your hardware: use GPU for embeddings, keep queries short, and avoid overloading Ollama with other jobs.

---

You are set when: Ollama has `llama3.2:3b`, `pip install` completed, FFmpeg is on PATH, PDFs are in `source_docs`, `ingest.py` succeeds, and **http://127.0.0.1:8000** loads the UI.

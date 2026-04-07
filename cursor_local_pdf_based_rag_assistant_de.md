# Local PDF-based RAG assistant development
_Exported on 4/7/2026 at 03:09:30 GMT+5 from Cursor (2.5.20)_

---

**User**

Role: Expert AI Engineer specializing in Local RAG and High-Performance Model Serving.

Objective: Build a local PDF-based RAG assistant that runs on Ollama (Llama 3.2 3B), uses ChromaDB, and provides real-time bullet-point answers for meetings.

Hardware Constraints: > * GPU: NVIDIA RTX 2060 (6GB VRAM)

RAM: 32GB

Target: Sub-2 second response time.

Task 1: Ingestion Pipeline (ingest.py)

Create a script using PyMuPDF (fitz) to load PDFs from a folder named ./source_docs.

Implement recursive character splitting (chunk size 1000, overlap 200).

Use sentence-transformers/all-MiniLM-L6-v2 for local embeddings.

Store vectors in a local ChromaDB instance in the ./db directory.

Task 2: Retrieval & LLM Logic (app.py)

Connect to Ollama running locally. Use the llama3.2:3b model.

Implement a Similarity Search to retrieve the top 3 most relevant chunks from ChromaDB based on the user query.

System Prompt: "You are a professional meeting assistant. Using ONLY the provided context, answer the user's question in 3-5 concise bullet points. If the answer isn't in the context, say 'Information not found in documents.' Keep it brief for live reading."

Task 3: The Toggle & Meeting Integration

Create a simple FastAPI backend with a /chat endpoint.

Add a is_active boolean check. If False, the system should return an empty response to save GPU resources.

Provide a simple HTML/JS frontend with a "Live Mode" Toggle Button and a text area for the live transcription/query input.

Task 4: Ease of Use

Generate a requirements.txt with all necessary local libraries.

Create a start.bat file to run the ingestion and start the local server in one click.

Formatting: Ensure all code is modular, well-commented, and optimized for low-latency GPU execution.

---

**Cursor**


Exploring the workspace and building the RAG assistant per your specifications.




Creating a self-contained RAG project at the workspace root: implementing ingestion, FastAPI app with Ollama integration, static frontend, `requirements.txt`, and `start.bat`.




Fixing ingestion chunk indexing and CUDA fallback, then adding `app.py`, static frontend, `start.bat`, and `source_docs/.gitkeep`.




Updating the frontend to always POST with `is_active`, renaming the response field to `response` for clarity, and adding `start.bat` plus `source_docs/.gitkeep`.




Replacing the root `StaticFiles` mount with a `FileResponse` for `/` to avoid routing conflicts with FastAPI's `/docs` and `/openapi.json`.




Here is what was added under `e:\Local RAG` and how it fits your spec.

### Layout

| Path | Role |
|------|------|
| `ingest.py` | PyMuPDF → `RecursiveCharacterTextSplitter` (1000 / 200) → `all-MiniLM-L6-v2` → Chroma in `./db` |
| `app.py` | FastAPI `/chat`, top-3 Chroma query, Ollama `llama3.2:3b`, `is_active` short-circuit |
| `static/index.html` | Live Mode toggle + textarea + POST `/chat` |
| `requirements.txt` | Dependencies (see CUDA note below) |
| `start.bat` | Runs ingest then `uvicorn` |
| `source_docs/` | Put PDFs here (`.gitkeep` only for now) |

### Behavior

- **Ingestion**: Recursively finds `*.pdf` under `./source_docs`, wipes and rebuilds the `meeting_docs` collection so re-runs stay predictable. Embeddings are **L2-normalized** to match Chroma’s **cosine** space.
- **`/chat`**: `{"query": "...", "is_active": true|false}`. If `is_active` is `false`, the handler returns immediately with `response: ""` and `chunks_used: 0` (no embedding, no Chroma query, no Ollama). If the index is empty or nothing matches, you get the required *“Information not found in documents.”* line.
- **Ollama**: Defaults to `http://127.0.0.1:11434` and model `llama3.2:3b`. Override with env vars `OLLAMA_HOST` and `OLLAMA_MODEL` if needed.
- **UI**: `GET /` serves the HTML; `/docs` and `/openapi.json` stay available for API checks.

### Sub-2s latency

End-to-end time is usually dominated by **Ollama** (3B on a 6GB card) plus prompt size. The stack keeps the retrieval side light: one small embedding, three chunks, low temperature. For best throughput, use a **CUDA** build of PyTorch for embeddings so MiniLM stays on the GPU; install from [pytorch.org](https://pytorch.org/get-started/locally/) and keep `requirements.txt`’s `torch` in sync with that wheel.

### One-time setup

1. `python -m venv .venv` then `.\.venv\Scripts\activate` (optional; `start.bat` auto-activates `.venv` if it exists).
2. `pip install -r requirements.txt`
3. `ollama pull llama3.2:3b` and leave Ollama running.
4. Copy PDFs into `source_docs\`.
5. Run `start.bat` and open `http://127.0.0.1:8000`.

If you want ingestion only when PDFs change (incremental upserts instead of full rebuild), say so and we can adjust `ingest.py` without touching the API surface.


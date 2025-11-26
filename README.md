## AI4D RAG Demo API

A minimal Retrieval-Augmented Generation (RAG) FastAPI service that lets you:

1. Upload documents (PDF, DOCX, TXT, MD) which are chunked, embedded with a SentenceTransformer model, and stored in Pinecone.
2. Query those documents: the top-k relevant chunks are retrieved and passed as context to a Gemini model to synthesize an answer.

This is intentionally small and transparent so you can extend it for your own experimentation (different embedding models, vector DBs, LLM providers, auth, etc.).

---
### Architecture Overview

Flow:

```
Client -> /upload (store raw files + create chunks)
			-> Embeddings (SentenceTransformer) -> Pinecone upsert (id, vector, metadata)
Client -> /chat (query) -> Embed query -> Pinecone similarity search -> top-k text chunks
			-> Build prompt with retrieved context -> Gemini LLM -> Answer
```

Key components:
- FastAPI app in `main.py` defines endpoints.
- SentenceTransformer (default: `sentence-transformers/all-MiniLM-L6-v2`, 384-dim) for embeddings.
- Pinecone serverless index (cosine similarity) for vector storage.
- Gemini GenerativeModel for answer synthesis.
- Local filesystem under `data/` holds original files + `metadata.json` with per-chunk info.

---
### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/upload` | Upload one or more files into a logical `context` (auto-generated if omitted). Chunks & embeddings are created and stored. |
| POST | `/chat` | Ask a question against a specific `context`; retrieves top-k (5) chunks then calls Gemini. |
| GET | `/contexts` | List available contexts (directory names under `data/`). |
| GET | `/context/{name}/metadata` | Retrieve full chunk metadata for a context. |

---
### Data Layout

```
data/
	<context>/
		files/              # Original uploaded documents
		metadata.json       # Array of chunk metadata records
```

Each metadata entry contains:
```json
{
	"id": "<uuid>",
	"context": "<context name>",
	"filename": "original name",
	"offset_start": <char index>,
	"offset_end": <char index>,
	"text": "chunk contents"
}
```

Chunking parameters (env tunable): `RAG_CHUNK_SIZE` (default 500 chars) and `RAG_CHUNK_OVERLAP` (default 100 chars).

---
### Requirements

See `requirements.txt`:
```
fastapi
python-multipart
dotenv
pinecone
sentence_transformers
google-generativeai
PyPDF2
python-docx
uvicorn
```

---
### Environment Variables

Set these (e.g. in a `.env` file) before running:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PINECONE_API_KEY` | Yes | — | Pinecone API key |
| `PINECONE_INDEX_NAME` | No | `big-rag` | Index name (auto-created if missing) |
| `EMBEDDING_MODEL_NAME` | No | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace SentenceTransformer model |
| `GEMINI_API_KEY` | Yes | — | Google Gemini API key |
| `LLM_MODEL_NAME` | No | `gemini-2.5-flash` | Gemini model identifier |
| `RAG_DATA_DIR` | No | `./data` | Root directory for stored contexts |
| `RAG_CHUNK_SIZE` | No | `500` | Chunk character length |
| `RAG_CHUNK_OVERLAP` | No | `100` | Overlap between consecutive chunks |

Create `.env` example:
```
PINECONE_API_KEY=your_pinecone_key
GEMINI_API_KEY=your_gemini_key
PINECONE_INDEX_NAME=big-rag
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL_NAME=gemini-2.5-flash
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=100
```

---
### Installation & Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ensure .env is present
uvicorn main:app --reload --port 8000
```

Server will be available at: `http://localhost:8000`.

Automatic index creation: On startup if `PINECONE_INDEX_NAME` does not exist, it is created with dimension 384 & cosine metric.

---
### Usage Examples

Upload two files into a new auto-generated context:
```bash
curl -X POST http://localhost:8000/upload \
	-F "files=@/path/to/doc1.pdf" \
	-F "files=@/path/to/notes.md"
```
Response:
```json
{"context": "ctx-abc123ef", "chunks": 42}
```

Upload specifying a custom context (e.g. `tech-entry-level`):
```bash
curl -X POST http://localhost:8000/upload \
	-F "context=tech-entry-level" \
	-F "files=@/path/to/guide.pdf"
```

List contexts:
```bash
curl http://localhost:8000/contexts
```

Inspect metadata:
```bash
curl http://localhost:8000/context/tech-entry-level/metadata
```

Ask a question:
```bash
curl -X POST http://localhost:8000/chat \
	-F "context=tech-entry-level" \
	-F "query=What entry level non-coding roles are listed?"
```

Sample answer payload:
```json
{
	"answer": "The document cites roles such as Product Design, Brand Design, Digital Marketing, Content Creation, Technical Writing, QA/Testing, Tech Support, Data Analyst, etc.",
	"context": ["Product Design ...", "Brand Design ..."]
}
```

---
### Customization Ideas

- Swap embedding model (ensure dimension matches index).
- Change vector DB (local FAISS, Milvus, Qdrant) with a thin adapter.
- Add authentication / API keys per tenant.
- Add streaming responses or citations formatting.
- Implement re-ranking or hybrid (BM25 + dense) retrieval.
- Add evaluation harness (retrieval precision, answer quality).

---
### Error Handling Notes

- Upload of unsupported file types falls back to best-effort UTF-8/Latin-1 decode.
- Missing optional libraries (`PyPDF2`, `python-docx`) return 500 for those file types.
- A missing context in `/context/{name}/metadata` yields 404.

---
### Development Tips

- Regenerate the index if you change embedding dimension (delete & recreate Pinecone index).
- Keep chunk size balanced: larger -> fewer vectors, potential loss of granularity; smaller -> more vectors, higher cost.
- Consider normalizing or cleaning text before chunking for better embeddings.
- Add logging around upsert & query latency to monitor performance.

---
### Security & Privacy

This demo stores raw document contents on disk and sends chunks to third-party services (Pinecone, Gemini). For sensitive data you must:
- Encrypt storage or avoid persisting raw files.
- Apply PII redaction pre-embedding.
- Restrict CORS origins.
- Add authentication & rate limiting.

---
### License

No license specified yet. Add one (e.g., MIT) if you plan to share publicly.

---
### Contributing

Open an issue or submit a PR with focused changes (new retrievers, auth layer, tests). Keep modifications small and well-described.

---
### Roadmap (Suggested)

1. Add test suite (unit tests for chunking & retrieval).
2. Support streaming LLM output.
3. Implement hybrid retrieval.
4. Add evaluation scripts.
5. Containerize with Docker & optional compose for local Pinecone alternative.

---
### Disclaimer

Gemini & Pinecone usage may incur costs; monitor API usage. Ensure compliance with data handling policies.

---
### Quick Start TL;DR

```bash
pip install -r requirements.txt
uvicorn main:app --reload
curl -X POST http://localhost:8000/upload -F "files=@doc.pdf"
curl -X POST http://localhost:8000/chat -F "context=ctx-xxxx" -F "query=Your question"
```

Enjoy experimenting with retrieval-augmented generation!

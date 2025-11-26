from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
import json
import uuid
from io import BytesIO
import dotenv



# Vector DB
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Set up Gemini
import google.generativeai as genai

# File parsing
try:
    import PyPDF2
except:
    PyPDF2 = None

try:
    import docx
except:
    docx = None

dotenv.load_dotenv()

# -------------------- Configuration --------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "big-rag")
EMBED_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemini-2.5-flash")
DATA_DIR = os.environ.get("RAG_DATA_DIR", "./data")
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", 100))

# Create the data directory that holds the documents that form the vector data.
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Demo RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Embedding model
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# LLM model
genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel(LLM_MODEL_NAME)

# Pinecone DB and Index
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        vector_type="dense",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
        deletion_protection="disabled",
        tags={
            "environment": "development"
        }
    )
index = pc.Index(PINECONE_INDEX_NAME)

# ---- Helper functions ------ #

# Chunking the text
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, (start, end)))

        if end == length:
            break
        start = end - overlap if end - overlap > start else end
    return chunks


def extract_text(filename: str, content: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".txt") or lower.endswith(".md"):
        try:
            return content.decode("utf-8")
        except:
            return content.decode("latin-1", errors="ignore")


    if lower.endswith(".pdf"):
        if PyPDF2 is None:
            raise HTTPException(500, "PyPDF2 not installed")

        reader = PyPDF2.PdfReader(BytesIO(content))
        txt = []
        for p in reader.pages:
            try:
                txt.append(p.extract_text() or "")
            except:
                txt.append("")
        
        return "\n".join(txt)


    if lower.endswith(".docx"):
        if docx is None:
            raise HTTPException(500, "python-docx not installed")

        d = docx.Document(BytesIO(content))
        return "\n".join([p.text for p in d.paragraphs])


    try:
        return content.decode("utf-8")
    except:
        return content.decode("latin-1", errors="ignore")


# ---- Endpoints ------- #
@app.post("/upload")
def upload_files(files: List[UploadFile] = File(...), context: Optional[str] = Form(None)):
    if context is None:
        context = f"ctx-{uuid.uuid4().hex[:8]}"

    ctx_dir = os.path.join(DATA_DIR, context)
    os.makedirs(ctx_dir, exist_ok=True)
    file_dir = os.path.join(ctx_dir, "files")
    os.makedirs(file_dir, exist_ok=True)

    metadata_path = os.path.join(ctx_dir, "metadata.json")
    metadata = []
    if os.path.exists(metadata_path):
        metadata = json.load(open(metadata_path))

    new_vectors = []

    for f in files:
        content = f.file.read()
        text = extract_text(f.filename, content)
        chunks = chunk_text(text)

        # save file
        dest = os.path.join(file_dir, f.filename)
        with open(dest, "wb") as out:
            out.write(content)

        # process chunks
        for chunk, (s, e) in chunks:
            vec = embed_model.encode(chunk).tolist()
            cid = uuid.uuid4().hex
            meta = {
                "id": cid,
                "context": context,
                "filename": f.filename,
                "offset_start": s,
                "offset_end": e,
                "text": chunk,
            }
            new_vectors.append((cid, vec, meta))
            metadata.append(meta)

    # Upsert into pinecone
    index.upsert(vectors=new_vectors)

    # Save the metadata.
    json.dump(metadata, open(metadata_path, "w"), indent=2)

    return {"context": context, "chunks": len(new_vectors)}


@app.post("/chat")
def chat(context: str = Form(...), query: str = Form(...)):
    # embed query
    qvec = embed_model.encode(query).tolist()
    results = index.query(vector=qvec, top_k=5, include_metadata=True, filter={"context": context})

    retrieved = [m["metadata"]["text"] for m in results["matches"]]
    context_block = "\n".join(retrieved)


    # With LLM
    prompt = f"""
Context:
    {context_block}

    Question: {query}
    
    Based on the context provided above, generate a succint answer to the query above.
"""
    response = llm_model.generate_content(prompt)

    return {"answer": response.text, "context": retrieved}


@app.get("/contexts")
def list_contexts():
    return [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]


@app.get("/context/{name}/metadata")
def get_metadata(name: str):
    p = os.path.join(DATA_DIR, name, "metadata.json")
    if not os.path.exists(p):
        raise HTTPException(404, "Context not found")
    return json.load(open(p))

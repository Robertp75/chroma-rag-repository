import os
import json
import numpy as np
import httpx
import onnxruntime as ort
from tokenizers import Tokenizer
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

app = FastAPI()

# ==== Chroma / Proxy config ====
CHROMA_BASE = os.environ.get("CHROMA_BASE")  # e.g. http://100.103.222.100:8000
TENANT = os.environ.get("CHROMA_TENANT", "default_tenant")
DATABASE = os.environ.get("CHROMA_DATABASE", "default_database")
API_KEY = os.environ.get("PROXY_API_KEY")

# ==== Model config ====
# We download these files at startup (cached in /tmp). You can change MODEL_ID if needed.
MODEL_ID = os.environ.get("ONNX_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
HF_BASE = f"https://huggingface.co/{MODEL_ID}/resolve/main"

# We will download these files:
# - tokenizer.json
# - model.onnx  (some repos name it differently; we handle fallback)
TOKENIZER_URL = os.environ.get("TOKENIZER_URL", f"{HF_BASE}/tokenizer.json")
ONNX_URL_PRIMARY = os.environ.get("ONNX_URL", f"{HF_BASE}/model.onnx")
ONNX_URL_FALLBACK = os.environ.get("ONNX_URL_FALLBACK", f"{HF_BASE}/onnx/model.onnx")

# Local cache paths
TOKENIZER_PATH = "/tmp/tokenizer.json"
MODEL_PATH = "/tmp/model.onnx"

# ---- Helper: auth ----
def check_auth(req: Request):
    if not API_KEY:
        return
    if req.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---- Helper: download file ----
async def download_file(url: str, path: str):
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to download {url} (status {r.status_code})")
        with open(path, "wb") as f:
            f.write(r.content)

# ---- Load tokenizer + ONNX session (lazy init) ----
tokenizer = None
session = None
input_names = None

async def ensure_model_loaded():
    global tokenizer, session, input_names

    if tokenizer is None:
        # download tokenizer.json
        if not os.path.exists(TOKENIZER_PATH):
            await download_file(TOKENIZER_URL, TOKENIZER_PATH)
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    if session is None:
        # download ONNX model (try primary, fallback)
        if not os.path.exists(MODEL_PATH):
            try:
                await download_file(ONNX_URL_PRIMARY, MODEL_PATH)
            except Exception:
                await download_file(ONNX_URL_FALLBACK, MODEL_PATH)

        # CPU session
        sess_opts = ort.SessionOptions()
        session = ort.InferenceSession(MODEL_PATH, sess_options=sess_opts, providers=["CPUExecutionProvider"])
        input_names = [i.name for i in session.get_inputs()]

# ---- Mean pooling ----
def mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    # last_hidden_state: (batch, seq, hidden)
    # attention_mask: (batch, seq)
    mask = attention_mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)  # (batch, seq, 1)
    summed = np.sum(last_hidden_state * mask, axis=1)  # (batch, hidden)
    counts = np.clip(np.sum(mask, axis=1), 1e-9, None)  # (batch, 1)
    return summed / counts

# ---- Embed text to 384-d vector ----
def embed_text(text: str) -> list[float]:
    # tokenize
    enc = tokenizer.encode(text)
    input_ids = np.array([enc.ids], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask], dtype=np.int64)

    ort_inputs = {}
    if "input_ids" in input_names:
        ort_inputs["input_ids"] = input_ids
    if "attention_mask" in input_names:
        ort_inputs["attention_mask"] = attention_mask
    # some models also take token_type_ids (we set to zeros if required)
    if "token_type_ids" in input_names:
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
        ort_inputs["token_type_ids"] = token_type_ids

    outputs = session.run(None, ort_inputs)

    # Heuristic: first output is usually last_hidden_state
    last_hidden_state = outputs[0]
    pooled = mean_pool(last_hidden_state, attention_mask)  # (1, hidden)

    vec = pooled[0].tolist()

    # Safety check: your DB dimension is 384
    if len(vec) != 384:
        raise HTTPException(status_code=500, detail=f"Embedding dim {len(vec)} != 384. Wrong ONNX model?")
    return vec

# ---- Chroma helpers ----
async def list_collections(client: httpx.AsyncClient):
    url = f"{CHROMA_BASE.rstrip('/')}/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections"
    r = await client.get(url)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail={"chroma_status": r.status_code, "body": r.text})
    return r.json()

async def get_collection_id(client: httpx.AsyncClient, name: str) -> str:
    cols = await list_collections(client)
    col = next((c for c in cols if c.get("name") == name), None)
    if not col:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return col["id"]

# ==== API ====
class RagQuery(BaseModel):
    collection: str
    question: str
    n_results: int = 3

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/rag/query")
async def rag_query(payload: RagQuery, request: Request):
    check_auth(request)

    if not CHROMA_BASE:
        raise HTTPException(status_code=500, detail="CHROMA_BASE not set")

    await ensure_model_loaded()

    # 1) embed
    vec = embed_text(payload.question)

    timeout = httpx.Timeout(connect=5.0, read=60.0, write=60.0, pool=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        collection_id = await get_collection_id(client, payload.collection)

        q_url = f"{CHROMA_BASE.rstrip('/')}/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections/{collection_id}/query"
        body = {
            "query_embeddings": [vec],
            "n_results": payload.n_results,
            "include": ["documents", "metadatas", "distances"]
        }

        r = await client.post(q_url, json=body)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail={"chroma_status": r.status_code, "body": r.text})

        result = r.json()

    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]

    hits = []
    for i in range(len(ids)):
        hits.append({
            "id": ids[i],
            "distance": dists[i] if i < len(dists) else None,
            "metadata": metas[i] if i < len(metas) else None,
            "document": docs[i] if i < len(docs) else None,
        })

    return {
        "collection": payload.collection,
        "question": payload.question,
        "n_results": payload.n_results,
        "hits": hits
    }

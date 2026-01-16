import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import httpx
from sentence_transformers import SentenceTransformer

app = FastAPI()

# === Env Vars ===
CHROMA_BASE = os.environ.get("CHROMA_BASE")  # e.g. http://100.103.222.100:8000
TENANT = os.environ.get("CHROMA_TENANT", "default_tenant")
DATABASE = os.environ.get("CHROMA_DATABASE", "default_database")
API_KEY = os.environ.get("PROXY_API_KEY")  # recommended
MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Load embedding model once at startup
model = SentenceTransformer(MODEL_NAME)

class RagQuery(BaseModel):
    collection: str
    question: str
    n_results: int = 3

def check_auth(req: Request):
    if not API_KEY:
        return
    if req.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/health")
async def health():
    return {"ok": True}

async def get_collection_id(client: httpx.AsyncClient, collection_name: str) -> str:
    if not CHROMA_BASE:
        raise HTTPException(status_code=500, detail="CHROMA_BASE not set")

    url = f"{CHROMA_BASE.rstrip('/')}/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections"
    r = await client.get(url)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail={"chroma_status": r.status_code, "body": r.text})

    cols = r.json()
    col = next((c for c in cols if c.get("name") == collection_name), None)
    if not col:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    return col["id"]

@app.post("/rag/query")
async def rag_query(payload: RagQuery, request: Request):
    check_auth(request)

    if not CHROMA_BASE:
        raise HTTPException(status_code=500, detail="CHROMA_BASE not set")

    # 1) Create embedding (must match DB dimension = 384)
    vec = model.encode([payload.question])[0].tolist()
    if len(vec) != 384:
        raise HTTPException(status_code=500, detail=f"Embedding dim {len(vec)} != 384 (wrong model?)")

    timeout = httpx.Timeout(connect=5.0, read=60.0, write=60.0, pool=60.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        # 2) Resolve collection name -> id (IDs can change, names are stable)
        collection_id = await get_collection_id(client, payload.collection)

        # 3) Query Chroma v2
        q_url = f"{CHROMA_BASE.rstrip('/')}/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections/{collection_id}/query"
        body = {
            "query_embeddings": [vec],
            "n_results": payload.n_results,
            # IMPORTANT: v2 include does NOT accept "ids"
            "include": ["documents", "metadatas", "distances"]
        }

        r = await client.post(q_url, json=body)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail={"chroma_status": r.status_code, "body": r.text})

        result = r.json()

    # 4) Flatten first query result (since we send one embedding)
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

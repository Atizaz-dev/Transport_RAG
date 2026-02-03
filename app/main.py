"""
FastAPI app: /query (NL question -> answer + sources), /ingest (trigger ingest), health.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import ingest as ingest_module
from app.config import SHIPMENT_CSV
from app.query import query

app = FastAPI(
    title="TMS Demo: NL Query over Shipments",
    description="Ask questions about your shipment data in plain language.",
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def post_query(body: QueryRequest):
    try:
        result = query(body.question)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
def run_ingest():
    """Run ingest once (CSV -> Pinecone). Requires CSV at SHIPMENT_CSV."""
    if not SHIPMENT_CSV.exists():
        raise HTTPException(
            status_code=400,
            detail=f"CSV not found at {SHIPMENT_CSV}. Download from Kaggle and put in data/.",
        )
    try:
        ingest_module.main()
        return {"status": "ok", "message": "Ingest completed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve simple frontend from project root
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")
else:
    @app.get("/")
    def index():
        return {
            "message": "TMS Demo API. Use POST /query with {\"question\": \"...\"} or add static/index.html for UI.",
        }

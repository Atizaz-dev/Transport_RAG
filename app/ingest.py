"""
Ingest shipment CSV: load rows, build text per row, embed with OpenAI, upsert to Pinecone.
Run once after putting CSV in data/ and setting env vars.
Optimized: larger embedding batches + parallel requests to reduce time.
"""
import concurrent.futures
from app.config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    SHIPMENT_CSV,
)
from app.data_loader import load_shipments, shipments_to_documents
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

BATCH_SIZE = 500
MAX_EMBED_WORKERS = 5
UPSERT_BATCH = 100
EMBED_DIM = 1536


def _embed_one_batch(args: tuple) -> tuple[int, list[list[float]]]:
    """Embed a single batch. Returns (start_index, list of embeddings)."""
    client, start_idx, texts_chunk, model = args
    resp = client.embeddings.create(input=texts_chunk, model=model)
    embeddings = [e.embedding for e in resp.data]
    return (start_idx, embeddings)


def get_embeddings(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    """Embed texts with OpenAI: large batches + parallel requests."""
    batch_tasks = [
        (client, i, texts[i : i + BATCH_SIZE], model)
        for i in range(0, len(texts), BATCH_SIZE)
    ]
    results: list[tuple[int, list[list[float]]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_EMBED_WORKERS) as executor:
        futures = [executor.submit(_embed_one_batch, t) for t in batch_tasks]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Embedding", unit="batch"):
            results.append(f.result())
    results.sort(key=lambda x: x[0])
    out: list[list[float]] = []
    for _, embs in results:
        out.extend(embs)
    return out


def main() -> None:
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        raise SystemExit("Set OPENAI_API_KEY and PINECONE_API_KEY in .env")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = pc.list_indexes()
    index_names = [idx.name for idx in (existing.indexes if hasattr(existing, "indexes") else [])]
    if PINECONE_INDEX not in index_names:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(PINECONE_INDEX)

    df = load_shipments(SHIPMENT_CSV)
    docs = shipments_to_documents(df)
    texts = [t for t, _ in docs]
    metas = [m for _, m in docs]
    total = len(texts)

    print(f"Loaded {total:,} rows from {SHIPMENT_CSV.name}. Embedding and upserting…")

    client = OpenAI(api_key=OPENAI_API_KEY)
    embeddings = get_embeddings(client, texts, OPENAI_EMBEDDING_MODEL)

    # Upsert in batches (Pinecone limit). Progress: how many inserted, how many left.
    upsert_batches = list(range(0, len(embeddings), UPSERT_BATCH))
    for i in tqdm(upsert_batches, desc="Upserting", unit="batch", total=len(upsert_batches)):
        end = min(i + UPSERT_BATCH, len(embeddings))
        vectors = [
            {"id": f"row_{metas[j]['row_id']}", "values": embeddings[j], "metadata": metas[j]}
            for j in range(i, end)
        ]
        index.upsert(vectors=vectors)

    print(f"Done. Ingested {total:,} shipment rows into index {PINECONE_INDEX}.")


if __name__ == "__main__":
    main()

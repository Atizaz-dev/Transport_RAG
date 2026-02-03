"""
Natural language query: embed question, search Pinecone, build context, ask OpenAI for answer.
"""
from app.config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    PINECONE_HOST,
    TOP_K,
)
from openai import OpenAI
from pinecone import Pinecone


def query(question: str) -> dict:
    """
    Run RAG: question -> embed -> Pinecone top_k -> build context -> OpenAI completion.
    Returns {"answer": str, "sources": list[dict]}.
    """
    if not question or not question.strip():
        return {"answer": "Please ask a question about the shipment data.", "sources": []}

    client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX) if not PINECONE_HOST else pc.Index(PINECONE_INDEX, host=PINECONE_HOST)

    # Embed question
    emb = client.embeddings.create(input=[question.strip()], model=OPENAI_EMBEDDING_MODEL)
    qvec = emb.data[0].embedding

    # Search Pinecone (response may be dict or object in different SDK versions)
    res = index.query(vector=qvec, top_k=TOP_K, include_metadata=True)
    hits = getattr(res, "matches", None) if not isinstance(res, dict) else res.get("matches")
    if hits is None:
        hits = []
    hits = [m for m in hits if m is not None]
    def _get(m: dict, key: str, default=None):
        return m.get(key, default) if isinstance(m, dict) else getattr(m, key, default)

    sources = [{"id": _get(m, "id") or "", "score": _get(m, "score"), "metadata": _get(m, "metadata") or {}} for m in hits]

    if not hits:
        return {
            "answer": "No relevant shipments found in the index. Run the ingest step first and ensure the index has data.",
            "sources": [],
        }

    # Build context from retrieved rows (use stored text in metadata when available)
    context_parts = []
    for m in hits:
        meta = _get(m, "metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        row_text = (meta.get("text") if isinstance(meta, dict) else getattr(meta, "text", None)) or str(meta)
        context_parts.append(row_text)
    context = "\n\n".join(context_parts)

    # Ask LLM for a short answer with evidence
    system = (
        "You are a helpful assistant for a Transportation Management System. "
        "Answer the user's question about shipment data in 2-4 sentences. "
        "Use only the provided context (list of shipment rows). "
        "The data typically has: order ID, customer region, product category, order/ship/delivery dates, shipping mode, shipping cost, delivery status (Delivered/Delayed), delivery days. "
        "If the user asks about something not in the context (e.g. 'customer care calls'), say this dataset does not include that field and briefly suggest what they can ask instead (e.g. delivery status, region, product category, cost, delivery days)."
    )
    user_msg = f"Context (relevant shipment rows):\n{context}\n\nQuestion: {question.strip()}\n\nAnswer:"

    comp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        max_tokens=300,
    )
    answer = (comp.choices[0].message.content or "").strip()

    return {"answer": answer, "sources": sources}

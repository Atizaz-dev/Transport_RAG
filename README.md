# TMS Improvement Demo: Natural Language Query over Shipments

**Assessment:** Propose and demonstrate a practical improvement to Transportation Management Systems (TMS).

## Idea: Ask Shipment Data in Plain Language

Instead of building reports or learning query languages, users ask questions like:
- * Which shipments were delayed? *
- * Show me orders from the West region.*
- * What product categories have the most delayed deliveries?*
- * Shipments using Express shipping.*
- * High shipping cost orders.*
- * Orders with long delivery days.*

The system uses **semantic search** (Pinecone) and **LLM** (OpenAI) over your shipment dataset to return an answer with evidence (relevant rows).

## Dataset

We use a **Kaggle shipment dataset** so the demo works with real-world-style data.
datset url: https://www.kaggle.com/datasets/yashch05/e-com-shipping-dataset?resource=download

**Supported datasets (column names are normalized automatically):**

- **E-Commerce Order Fulfillment (50K Records)** — Columns: `Order_ID`, `Customer_Region`, `Product_Category`, `Order_Date`, `Ship_Date`, `Delivery_Date`, `Shipping_Mode`, `Shipping_Cost`, `Delivery_Status`, `Delivery_Days`. Set `SHIPMENT_CSV=data/E-Commerce Order Fulfillment Dataset (50K Records).csv`.
- **E-Com Shipping Dataset** (Kaggle) — Warehouse block, Mode of Shipment, Customer care calls, Cost, Weight, Reached on Time, etc. Set `SHIPMENT_CSV=data/train.csv` after downloading.

The code maps both schemas to a common internal format so the same queries work across datasets.

## Tech Stack

- **Backend:** Python, FastAPI  
- **Vector DB:** Pinecone (embeddings index)  
- **LLM & embeddings:** OpenAI  
- **Data:** Pandas (CSV load), optional Kaggle API for download  

## Assumptions

- Each **row** in the CSV is one shipment (or one logical record). We embed a text summary of the row (e.g. "Warehouse C, Ship, 2 care calls, weight 2.5, cost $100, reached on time: No, ...").
- **"Relevant"** = embedding similarity between the user question and these row summaries. The LLM then turns the top-k rows into a short answer with citations.
- No auth, no production hardening—focused on demonstrating the improvement in a small, runnable demo.

## Setup

1. **Clone and install**
   ```bash
   cd assessment-transport
   pip install -r requirements.txt
   ```

2. **Environment**
   - Copy the template and add your keys: `cp env.example .env`, then edit `.env` with:
     - `OPENAI_API_KEY` – your OpenAI API key
     - `PINECONE_API_KEY` – your Pinecone API key
     - `PINECONE_INDEX` – index name (e.g. `tms-shipments`), dimension **1536**, metric **cosine**
     - `SHIPMENT_CSV` – path to CSV (e.g. `data/train.csv`)
     - `OPENAI_EMBEDDING_MODEL=text-embedding-3-small`, `OPENAI_CHAT_MODEL=gpt-4o-mini`

3. **Dataset**
   - Download the E-Com Shipping (or similar) CSV from Kaggle and put it in `data/`.
   - Set `SHIPMENT_CSV` in `.env` to the path (e.g. `data/train.csv`).

4. **Pinecone**
   - Create an index in Pinecone: dimension = **1536** (OpenAI `text-embedding-3-small`), metric = cosine.
   - Put index name (and optionally env/host) in `.env`.

5. **Ingest once**
   ```bash
   python -m app.ingest
   ```
   This reads the CSV, builds text per row, embeds with OpenAI, and upserts to Pinecone.

6. **Run the app**
   ```bash
   uvicorn app.main:app --reload
   ```
   - API: `http://localhost:8000`
   - Try `POST /query` with `{"question": "Which shipments might be late?"}` or use the simple UI at the root.

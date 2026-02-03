"""
Load shipment CSV (e.g. Kaggle E-Com Shipping) and turn each row into a text
snippet for embedding. Handles common column name variants.
"""
from pathlib import Path
from typing import Any

import pandas as pd

# Map known column names (E-Com Shipping, E-Commerce Order Fulfillment, etc.) to common internal names
COLUMN_ALIASES = {
    # E-Com Shipping Dataset
    "warehouse_block": ["Warehouse_block", "warehouse_block", "Warehouse Block"],
    "mode_of_shipment": ["Mode_of_Shipment", "mode_of_shipment", "Mode of Shipment"],
    "customer_care_calls": ["Customer_care_calls", "customer_care_calls", "Customer care calls"],
    "customer_rating": ["Customer_rating", "customer_rating", "Customer_rating", "Customer rating"],
    "cost": ["Cost_of_the_Product", "cost_of_the_product", "Cost of the Product", "cost", "Shipping_Cost", "Shipping Cost"],
    "prior_purchases": ["Prior_purchases", "prior_purchases", "Prior purchases"],
    "product_importance": ["Product_importance", "product_importance", "Product_importance", "Product importance"],
    "weight": ["Weight_in_gms", "weight_in_gms", "Weight_in_gms", "Weight (gms)", "weight"],
    "reached_on_time": ["Reached.on.Time_Y.N", "reached.on.time_y.n", "Reached on Time_Y.N", "Reached on Time"],
    "discount_offered": ["Discount_offered", "discount_offered", "Discount offered"],
    "gender": ["Gender", "gender"],
    "id": ["ID", "id"],
    "order_id": ["Order_ID", "Order_ID", "order_id"],
    "customer_region": ["Customer_Region", "Customer_Region", "customer_region"],
    "product_category": ["Product_Category", "Product_Category", "product_category"],
    "order_date": ["Order_Date", "Order_Date", "order_date"],
    "ship_date": ["Ship_Date", "Ship_Date", "ship_date"],
    "delivery_date": ["Delivery_Date", "Delivery_Date", "delivery_date"],
    "shipping_mode": ["Shipping_Mode", "Shipping_Mode", "shipping_mode"],
    "delivery_status": ["Delivery_Status", "Delivery_Status", "delivery_status"],
    "delivery_days": ["Delivery_Days", "Delivery_Days", "delivery_days"],
}

COLUMN_ALIASES["mode_of_shipment"] = COLUMN_ALIASES["mode_of_shipment"] + ["Shipping_Mode", "Shipping Mode"]
COLUMN_ALIASES["reached_on_time"] = COLUMN_ALIASES["reached_on_time"] + ["Delivery_Status", "Delivery Status"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to a common set of names when possible."""
    out = df.copy()
    for canonical, aliases in COLUMN_ALIASES.items():
        for col in df.columns:
            if col.strip() in aliases or col == canonical:
                out = out.rename(columns={col: canonical})
                break
    return out


def _row_to_text(row: pd.Series, row_id: int) -> str:
    """Build a single text block for one shipment row for embedding."""
    parts = [f"Shipment row {row_id}."]
    # Canonical keys: E-Com Shipping + E-Commerce Order Fulfillment (order_id, region, category, dates, mode, cost, status, days)
    keys_order = [
        "order_id", "customer_region", "product_category", "order_date", "ship_date", "delivery_date",
        "warehouse_block", "mode_of_shipment", "customer_care_calls", "customer_rating", "cost",
        "prior_purchases", "product_importance", "weight", "reached_on_time", "delivery_status",
        "delivery_days", "discount_offered", "gender",
    ]
    for key in keys_order:
        if key not in row.index or pd.isna(row.get(key)):
            continue
        val = row[key]
        if key == "reached_on_time":
            # Support both 0/1/Y/N and "Delivered"/"Delayed" (from Delivery_Status)
            if val in (1, "1", "Y", "y", True, "Delivered", "delivered"):
                parts.append("Reached on time: Yes. Delivery status: Delivered.")
            else:
                parts.append("Reached on time: No. Delivery status: Delayed.")
        elif key == "delivery_status":
            # Already covered if column was mapped to reached_on_time; if separate column, add it
            parts.append(f"Delivery status: {val}.")
        else:
            parts.append(f"{key.replace('_', ' ').title()}: {val}.")
    # Append any remaining columns not in our list
    for col in row.index:
        if col not in COLUMN_ALIASES and pd.notna(row.get(col)):
            parts.append(f"{str(col).replace('_', ' ').title()}: {row[col]}.")
    return " ".join(parts)


def load_shipments(csv_path: Path) -> pd.DataFrame:
    """Load CSV and normalize column names."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Shipment CSV not found: {csv_path}. Download from Kaggle and put in data/.")
    df = pd.read_csv(csv_path)
    return _normalize_columns(df)


def shipments_to_documents(df: pd.DataFrame) -> list[tuple[str, dict[str, Any]]]:
    """
    Convert DataFrame to list of (text_for_embedding, metadata).
    metadata includes row_id, text (for RAG context), and display fields.
    """
    documents: list[tuple[str, dict[str, Any]]] = []
    for i, row in df.iterrows():
        text = _row_to_text(row, row_id=i)
        meta: dict[str, Any] = {"row_id": int(i), "text": text[:40_000]}  # Pinecone metadata size limit
        display_keys = [
            "order_id", "customer_region", "product_category", "mode_of_shipment",
            "cost", "weight", "reached_on_time", "delivery_status", "delivery_days",
            "warehouse_block", "customer_care_calls",
        ]
        for k in display_keys:
            if k in row.index and pd.notna(row.get(k)):
                v = row[k]
                meta[k] = str(v) if not isinstance(v, (int, float)) else v
        documents.append((text, meta))
    return documents

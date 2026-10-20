"""
catalogue.py
------------
Structured product catalogue backed by SQLite (see db.py).

The catalogue JSON (settings.BU_JSON_PATH) is the import source. On first use,
and whenever the file changes, it is flattened into the `products` table and
demo stock is regenerated. Prices are parsed once ("54/-" -> 54.0), so pricing
never has to read a number out of LLM-generated text.

Public API
----------
get_catalogue()               cached Catalogue for the configured files
Catalogue.get(product_id)     -> ProductRecord | None
Catalogue.find_by_sap(code)   -> list[ProductRecord]  (SAP codes can repeat)
Catalogue.all_products()      -> list[ProductRecord]
flatten_catalogue(json_data)  -> list[dict]  (pure; used by the loader and tests)
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from db import Meta, Product, Stock, get_engine
from settings import BU_JSON_PATH, DB_PATH, DEMO_STOCK_SEED

logger = logging.getLogger(__name__)

_META_HASH_KEY = "catalogue_sha256"


# ─── Records ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProductRecord:
    """A catalogue product with its (demo) stock, detached from the DB session."""
    id: str
    sap_code: str
    brand: str
    range_name: str
    variant: Optional[str]
    variant_type: Optional[str]
    item_name: str
    hsn_code: str
    packing: str
    unit_price: float
    price_raw: str
    on_hand: Optional[int] = None
    monthly_capacity: Optional[int] = None
    lead_time_days: Optional[int] = None
    stock_is_demo: bool = True

    def display_name(self) -> str:
        variant = f" / {self.variant}" if self.variant else ""
        return f"{self.brand.title()} {self.range_name}{variant} — {self.item_name}"

    def search_text(self) -> str:
        """
        Text used to embed and rerank this product in the BU RAG index.
        Catalogue abbreviations are spelled out on an extra line, because the
        search models don't know that "Ind." means indicator (measured:
        "Switches with Indicator" scored 0.03 against "... with Ind." and
        0.99 once expanded).
        """
        expanded = expand_abbreviations(self.item_name)
        also = f"Also described as: {expanded}\n" if expanded != self.item_name else ""
        return (
            f"Product: {self.item_name}\n"
            f"{also}"
            f"Brand: {self.brand}\n"
            f"Range: {self.range_name}\n"
            f"Variant: {self.variant or 'N/A'}\n"
            f"Price: Rs. {self.unit_price:g} per piece\n"
            f"SAP Code: {self.sap_code}\n"
            f"HSN Code: {self.hsn_code}\n"
            f"Packing: {self.packing}\n"
        )


@dataclass(frozen=True)
class LoadReport:
    loaded: bool                 # False when the DB was already up to date
    product_count: int
    duplicate_sap_codes: List[str]


# ─── Abbreviations used in catalogue item names ───────────────────────────────

_ABBREVIATIONS = [
    (re.compile(r"\bind\b\.*", re.I), "Indicator"),
    (re.compile(r"\breg\b\.*", re.I), "Regulator"),
    (re.compile(r"\bSP\b"), "Single Pole"),
    (re.compile(r"\bDP\b"), "Double Pole"),
    (re.compile(r"\bMCB\b"), "MCB (Miniature Circuit Breaker)"),
]


def expand_abbreviations(name: str) -> str:
    for pattern, replacement in _ABBREVIATIONS:
        name = pattern.sub(replacement, name)
    return re.sub(r"\s+", " ", name).strip()


# ─── Parsing ──────────────────────────────────────────────────────────────────

_PRICE_RE = re.compile(r"^\s*(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)\s*(?:/-)?\s*$", re.I)


def parse_price(raw: str) -> float:
    """Parse catalogue price strings such as "54/-", "1,250/-" or "Rs. 60"."""
    m = _PRICE_RE.match(str(raw))
    if not m:
        raise ValueError(f"Unrecognised price format: {raw!r}")
    return float(m.group(1).replace(",", ""))


def make_product_id(range_name: str, variant: Optional[str], item_name: str) -> str:
    key = f"{range_name}|{variant or ''}|{item_name}".lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def flatten_catalogue(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten brand → ranges → (items | colors | categories) into product rows."""
    brand = data.get("brand", "N/A")
    rows: List[Dict[str, Any]] = []
    for range_entry in data.get("product_data", []):
        range_name = range_entry.get("range", "N/A")
        groups: List[tuple] = []
        if "items" in range_entry:
            groups.append((None, None, range_entry["items"]))
        for colour in range_entry.get("colors", []):
            groups.append((colour.get("color"), "color", colour.get("items", [])))
        for category in range_entry.get("categories", []):
            groups.append((category.get("category"), "category", category.get("items", [])))

        for variant, variant_type, items in groups:
            for item in items:
                rows.append({
                    "id": make_product_id(range_name, variant, item["item_name"]),
                    "sap_code": str(item.get("sap_code", "")).strip(),
                    "brand": brand,
                    "range_name": range_name,
                    "variant": variant,
                    "variant_type": variant_type,
                    "item_name": item["item_name"].strip(),
                    "hsn_code": str(item.get("hsn_code", "")).strip(),
                    "packing": str(item.get("packing", "")).strip(),
                    "unit_price": parse_price(item["unit_price"]),
                    "price_raw": str(item["unit_price"]).strip(),
                })
    return rows


# ─── Demo stock ───────────────────────────────────────────────────────────────

def demo_stock_for(product_id: str, seed: int = DEMO_STOCK_SEED) -> Dict[str, int]:
    """
    Deterministic synthetic stock for one product.

    About 15% of products get a low monthly capacity so the "insufficient
    stock" path shows up in demos. String seeding is stable across machines.
    """
    rng = random.Random(f"{seed}:{product_id}")
    if rng.random() < 0.15:
        monthly_capacity = rng.randrange(50, 301, 10)
    else:
        monthly_capacity = rng.randrange(800, 5001, 50)
    on_hand = int(round(monthly_capacity * rng.uniform(0.25, 1.0), -1))
    return {
        "on_hand": on_hand,
        "monthly_capacity": monthly_capacity,
        "lead_time_days": rng.choice([7, 10, 14, 21]),
    }


# ─── Loader ───────────────────────────────────────────────────────────────────

def _file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_loaded(engine: Engine, json_path: str = BU_JSON_PATH) -> LoadReport:
    """Load the catalogue JSON into the DB if it is new or has changed."""
    digest = _file_sha256(json_path)
    with Session(engine) as session:
        current = session.get(Meta, _META_HASH_KEY)
        if current is not None and current.value == digest:
            count = len(session.scalars(select(Product.id)).all())
            return LoadReport(loaded=False, product_count=count, duplicate_sap_codes=[])

        with open(json_path, "r", encoding="utf-8-sig") as f:
            rows = flatten_catalogue(json.load(f))

        ids = Counter(r["id"] for r in rows)
        clashes = [pid for pid, n in ids.items() if n > 1]
        if clashes:
            raise ValueError(f"Catalogue has repeated range/variant/item rows: {clashes}")

        sap_counts = Counter(r["sap_code"] for r in rows)
        duplicates = sorted(code for code, n in sap_counts.items() if n > 1)
        if duplicates:
            logger.warning("Catalogue SAP codes used by more than one product: %s", duplicates)

        session.execute(delete(Stock))
        session.execute(delete(Product))
        session.add_all(Product(**row) for row in rows)
        session.add_all(
            Stock(product_id=row["id"], is_demo=True, **demo_stock_for(row["id"]))
            for row in rows
        )
        session.merge(Meta(key=_META_HASH_KEY, value=digest))
        session.commit()

    logger.info("Loaded %d catalogue products from %s", len(rows), json_path)
    return LoadReport(loaded=True, product_count=len(rows), duplicate_sap_codes=duplicates)


# ─── Query API ────────────────────────────────────────────────────────────────

class Catalogue:
    def __init__(self, engine: Engine, json_path: str = BU_JSON_PATH) -> None:
        self.engine = engine
        self.load_report = ensure_loaded(engine, json_path)

    def _query(self, *where) -> List[ProductRecord]:
        stmt = (
            select(Product, Stock)
            .join(Stock, Stock.product_id == Product.id, isouter=True)
            .where(*where)
            .order_by(Product.range_name, Product.variant, Product.item_name)
        )
        with Session(self.engine) as session:
            return [_to_record(p, s) for p, s in session.execute(stmt).all()]

    def get(self, product_id: str) -> Optional[ProductRecord]:
        found = self._query(Product.id == product_id)
        return found[0] if found else None

    def find_by_sap(self, sap_code: str) -> List[ProductRecord]:
        return self._query(Product.sap_code == sap_code.strip())

    def all_products(self) -> List[ProductRecord]:
        return self._query()


def _to_record(p: Product, s: Optional[Stock]) -> ProductRecord:
    return ProductRecord(
        id=p.id,
        sap_code=p.sap_code,
        brand=p.brand,
        range_name=p.range_name,
        variant=p.variant,
        variant_type=p.variant_type,
        item_name=p.item_name,
        hsn_code=p.hsn_code,
        packing=p.packing,
        unit_price=p.unit_price,
        price_raw=p.price_raw,
        on_hand=s.on_hand if s else None,
        monthly_capacity=s.monthly_capacity if s else None,
        lead_time_days=s.lead_time_days if s else None,
        stock_is_demo=s.is_demo if s else True,
    )


@lru_cache(maxsize=None)
def get_catalogue(db_path: str = DB_PATH, json_path: str = BU_JSON_PATH) -> Catalogue:
    """Return the Catalogue for these files (one instance per path pair)."""
    return Catalogue(get_engine(db_path), json_path)

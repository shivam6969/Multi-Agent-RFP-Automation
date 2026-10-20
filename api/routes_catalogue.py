"""Catalogue search, used by "Change Product" on the matching screen."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from catalogue import expand_abbreviations, get_catalogue
from db import User

from .context import current_user
from .schemas import ProductOut

router = APIRouter(prefix="/api/catalogue", tags=["catalogue"])


@router.get("/products", response_model=List[ProductOut])
def search_products(q: str = "", limit: int = 25, _: User = Depends(current_user)) -> List[Dict[str, Any]]:
    """
    Plain word search over name, range, variant and SAP code (every word must
    appear; "ind" also finds "Indicator"). Fast and predictable for a picker.
    """
    words = q.lower().split()
    results = []
    for p in get_catalogue().all_products():
        haystack = " ".join(filter(None, [
            p.item_name, expand_abbreviations(p.item_name), p.range_name, p.variant, p.sap_code,
        ])).lower()
        if all(w in haystack for w in words):
            results.append({
                "product_id": p.id, "product_name": p.item_name, "product_range": p.range_name,
                "product_variant": p.variant, "sap_code": p.sap_code, "hsn_code": p.hsn_code,
                "packing": p.packing, "unit_price": p.unit_price,
                "stock_on_hand": p.on_hand, "monthly_available": p.monthly_capacity,
                "lead_time_days": p.lead_time_days, "stock_is_demo": p.stock_is_demo,
            })
            if len(results) >= min(limit, 100):
                break
    return results

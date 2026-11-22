"""
Search keywords from the business's own stock.

Catalogue names mix a product type with ratings, sizes and range names
("6/16A 3 Pin Combined Shuttered Socket", "16 A SP C Mini MCB Coral"). We
strip those to the product type ("shuttered socket", "mini mcb"), group by the
head word, and rank by how much the business stocks, so the tender search asks
for what it can actually supply. Pure rules, no AI: the same stock always gives
the same keywords.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from catalogue import ProductRecord, expand_abbreviations

# Words that describe a rating, size, finish or range, not what the product is.
_NOISE = {
    "a", "an", "the", "and", "&", "of", "for", "with", "to", "my", "do", "not",
    "way", "pin", "pole", "single", "double", "sp", "dp", "tp", "c", "b", "ka", "w", "v", "h", "m",
    "new", "series", "type", "combined", "universal", "international", "metal", "ring",
    "high", "speed", "step", "type", "white", "grey", "silver", "gold", "black", "ebony",
    "coral", "oro", "metallica", "magna", "havells", "outer", "inner", "flush", "gl",
}

# How tenders usually name these products (GeM item categories say "modular").
_SEARCH_FORM = {
    "switch": "modular switch",
    "socket": "modular socket",
    "plate": "modular cover plate",
    "box": "modular switch box",
    "mcb": "mcb",
    "regulator": "fan regulator",
}

# Head words that are not products on their own.
_SKIP_HEADS = {"room", "module", "modules", "indicator", "unit", "units", "range"}


@dataclass
class StockKeyword:
    keyword: str                 # what to type into the search box
    head: str                    # the product word it groups ("socket")
    products: int                # catalogue products of this type
    units_on_hand: int           # their stock, summed
    examples: List[str] = field(default_factory=list)


def product_type(name: str) -> Optional[str]:
    """
    "6/16A 3 Pin Combined Shuttered Socket" -> "shuttered socket";
    "16A 1 way switch with ind." -> "switch"; None if nothing is left.
    """
    text = expand_abbreviations(name).lower()
    text = re.sub(r"\(.*?\)", " ", text)               # "(1M)", "(5X3)"
    text = re.sub(r"\bwith\b.*$", " ", text)            # "... with indicator"
    words = [
        w for w in re.findall(r"[a-z]+(?:'[a-z]+)?|\d\S*", text)
        if not any(ch.isdigit() for ch in w) and w not in _NOISE and len(w) > 1
    ]
    if not words:
        return None
    if words[-1] in ("breaker",):                       # "miniature circuit breaker" -> mcb
        return "mcb"
    phrase = words[-2:] if len(words) >= 2 else words
    return " ".join(phrase)


def extract_keywords(products: Iterable[ProductRecord], limit: int = 8,
                     min_products: int = 2) -> List[StockKeyword]:
    """Product-type keywords, the most-stocked first."""
    by_head: Dict[str, List[ProductRecord]] = defaultdict(list)
    phrases: Dict[str, Counter] = defaultdict(Counter)
    for p in products:
        kind = product_type(p.item_name)
        if not kind:
            continue
        head = kind.split()[-1]
        if head in _SKIP_HEADS:
            continue
        by_head[head].append(p)
        phrases[head][kind] += 1

    out: List[StockKeyword] = []
    for head, items in by_head.items():
        if len(items) < min_products:
            continue
        common = phrases[head].most_common(1)[0][0]
        keyword = _SEARCH_FORM.get(head, common)
        out.append(StockKeyword(
            keyword=keyword,
            head=head,
            products=len(items),
            units_on_hand=sum(p.on_hand or 0 for p in items),
            examples=[k for k, _ in phrases[head].most_common(3)],
        ))
    out.sort(key=lambda k: (k.units_on_hand, k.products), reverse=True)
    return out[:limit]


def stock_keywords(limit: int = 8) -> List[StockKeyword]:
    from catalogue import get_catalogue
    return extract_keywords(get_catalogue().all_products(), limit=limit)

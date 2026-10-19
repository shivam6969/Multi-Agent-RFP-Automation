"""A stand-in for BuRagTool.search_products_scored() using real catalogue rows."""

from __future__ import annotations

from typing import Dict, List, Tuple

from catalogue import ProductRecord, get_catalogue


def product(range_name: str, item_name: str) -> ProductRecord:
    for p in get_catalogue().all_products():
        if (p.range_name, p.item_name) == (range_name, item_name):
            return p
    raise KeyError(f"{range_name} / {item_name}")


def first_named(item_name: str) -> ProductRecord:
    for p in get_catalogue().all_products():
        if p.item_name == item_name:
            return p
    raise KeyError(item_name)


class FakeSearchTool:
    """Returns canned (product, score) lists for queries containing a keyword."""

    def __init__(self, results: Dict[str, List[Tuple[ProductRecord, float]]]) -> None:
        self.results = results
        self.queries: List[str] = []

    def search_products_scored(self, query: str, top_k: int = 8, candidate_k: int = 25):
        self.queries.append(query)
        for keyword, ranked in self.results.items():
            if keyword.lower() in query.lower():
                return ranked[:top_k]
        return []

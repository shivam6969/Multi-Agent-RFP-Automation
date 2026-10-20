"""
GeM (Government e-Marketplace) tender source: OFF by default.

Ported from legacy/gem_scraper (the team's original Streamlit scraper).
The team believes bidplus.gem.gov.in does not allow automated scraping, so
GemSource refuses to run unless settings.DISCOVERY_GEM_ENABLED is set; it
also needs `pip install selenium` and Chrome, which are not project
dependencies. The live scraping path has NOT been run as part of this
project; the document parser below is tested on the 32 archived tenders in
legacy/gem_scraper/.

parse_gem_text() reads the fixed labels of GeM bid / reverse-auction (RA)
documents and is used for uploaded GeM PDFs as well (no LLM needed).
"""

from __future__ import annotations

import importlib.util
import logging
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf

from settings import DISCOVERY_GEM_ENABLED

from .base import DiscoveryQuery, ProgressFn, RfpCandidate, SourceUnavailable

logger = logging.getLogger(__name__)

GEM_SEARCH_URL = "https://bidplus.gem.gov.in/all-bids"
_GEM_REF = re.compile(r"\bGEM/\d{4}/[A-Z]+/\d{4,}\b", re.I)


def looks_like_gem(text: str) -> bool:
    head = text[:1500]
    return bool(_GEM_REF.search(head)) and ("Bid" in head or "RA " in head or "RA Details" in head)


def _label(text: str, *labels: str) -> Optional[str]:
    for label in labels:
        m = re.search(rf"^{re.escape(label)}\s*:?\s*(.+)$", text, re.M)
        if m:
            value = re.sub(r"\s+", " ", m.group(1)).strip()
            if value and not set(value) <= {"*"}:
                return value
    return None


def _date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})", value)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(0), "%d-%m-%Y").date().isoformat()
    except ValueError:
        return None


def parse_gem_text(text: str) -> Dict[str, Any]:
    """Metadata from a GeM bid / RA document's labelled header."""
    ref = _GEM_REF.search(text)
    buyer = _label(text, "Organisation Name", "Department Name", "Ministry/State Name")
    category = _label(text, "Item Category")
    meta: Dict[str, Any] = {
        "reference": ref.group(0).upper() if ref else None,
        "buyer": buyer.title() if buyer and buyer.islower() else buyer,
        "deadline": _date(_label(text, "Bid End Date/Time", "RA End Date/Time")),
        "published": _date(_label(text, "Dated")),
        "category": category[:120] if category else None,
        "location": _label(text, "Consignee Address", "Address", "Office Name"),
        "title": (f"{category[:110]}" if category else None),
    }
    qty = _label(text, "Total Quantity")
    if qty and qty.replace(",", "").isdigit():
        meta["total_quantity"] = int(qty.replace(",", ""))
    return {k: v for k, v in meta.items() if v}


class GemSource:
    name = "gem"
    label = "GeM portal (live)"

    def available(self) -> Optional[str]:
        if not DISCOVERY_GEM_ENABLED:
            return "Switched off: scraping the GeM portal may not be permitted (set DISCOVERY_GEM_ENABLED to enable)."
        if importlib.util.find_spec("selenium") is None:
            return "Needs `pip install selenium` and Google Chrome."
        return None

    def search(self, query: DiscoveryQuery, out_dir: str,
               progress: Optional[ProgressFn] = None) -> List[RfpCandidate]:
        reason = self.available()
        if reason:
            raise SourceUnavailable(reason)

        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        folder = Path(out_dir) / "gem"
        folder.mkdir(parents=True, exist_ok=True)
        options = Options()
        options.add_argument("--headless=new")
        driver = webdriver.Chrome(options=options)
        found: List[RfpCandidate] = []
        try:
            driver.get(GEM_SEARCH_URL)
            wait = WebDriverWait(driver, 20)
            for keyword in query.keywords or ["modular switches"]:
                box = wait.until(EC.visibility_of_element_located((By.XPATH, "//input[@placeholder='Enter Keyword']")))
                box.clear()
                box.send_keys(keyword, Keys.RETURN)
                wait.until(EC.presence_of_element_located((By.XPATH, "//a[contains(text(),'GEM/')]")))
                links = [(a.text, a.get_attribute("href")) for a in driver.find_elements(By.XPATH, "//a[contains(text(),'GEM/')]")]
                for bid_no, href in links:
                    if len(found) >= query.count:
                        return found
                    if not href:
                        continue
                    path = folder / f"{bid_no.replace('/', '_')}.pdf"
                    time.sleep(2)   # be polite: one request every couple of seconds
                    with urllib.request.urlopen(href, timeout=30) as resp:
                        path.write_bytes(resp.read())
                    with pymupdf.open(path) as doc:
                        meta = parse_gem_text("".join(p.get_text() for p in doc))
                    found.append(RfpCandidate(
                        title=meta.get("title") or bid_no, pdf_path=str(path), source=self.name,
                        buyer=meta.get("buyer"), deadline=meta.get("deadline"), published=meta.get("published"),
                        location=meta.get("location"), category=meta.get("category"),
                        reference=meta.get("reference") or bid_no, meta={"keyword": keyword, "url": href},
                    ))
                    if progress:
                        progress(len(found), query.count, f"Downloaded {bid_no}")
        finally:
            driver.quit()
        return found

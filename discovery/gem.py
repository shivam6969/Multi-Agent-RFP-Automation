"""
GeM (Government e-Marketplace) tender source: OFF by default.

Drives a real (headless) Chrome with Selenium on bidplus.gem.gov.in/all-bids,
the way a person would: types each keyword into the search box, reads the bid
cards the page draws, and lets the browser download each bid document PDF.
With no keywords given, the keywords come from the business's own stock
(discovery/keywords.py), so it looks for tenders the business can supply.

It refuses to run unless settings.DISCOVERY_GEM_ENABLED is set: check that
GeM's terms of use allow automated access before enabling it. It also needs
`pip install selenium` and Google Chrome (not core dependencies). It is
deliberately slow: a pause between searches and between downloads.

parse_gem_text() reads the fixed labels of GeM bid / reverse-auction (RA)
documents and is used for uploaded GeM PDFs as well (no LLM needed).
"""

from __future__ import annotations

import importlib.util
import logging
import re
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import pymupdf

from settings import DISCOVERY_GEM_ENABLED

from .base import DiscoveryQuery, ProgressFn, RfpCandidate, SourceUnavailable

logger = logging.getLogger(__name__)

GEM_SEARCH_URL = "https://bidplus.gem.gov.in/all-bids"
PAUSE_SECONDS = 3            # between searches and between downloads
DOWNLOAD_TIMEOUT = 45        # seconds to wait for one PDF
# Bid documents and RA documents are PDFs; "list-ra-schedules" is an HTML page.
_DOC_LINK = re.compile(r"/(showbidDocument|showradocumentPdf|showdirectradocumentPdf)/\d+")
_GEM_REF = re.compile(r"\bGEM/\d{4}/[A-Z]+/\d{4,}\b", re.I)


def looks_like_gem(text: str) -> bool:
    head = text[:1500]
    return bool(_GEM_REF.search(head)) and ("Bid" in head or "RA " in head or "RA Details" in head)


def _label(text: str, *labels: str) -> Optional[str]:
    """
    The value of the first label found. Two layouts:
      older documents  "Item Category : Modular Switch" on one line;
      current ones     "<Hindi>/Item Category" with the value on the next
                       line(s), up to the next bilingual label.
    """
    lines = text.splitlines()
    for label in labels:
        same = re.compile(rf"^(?:.*/\s*)?{re.escape(label)}\s*:?\s*(.*\S)\s*$")
        alone = re.compile(rf"^(?:.*/\s*)?{re.escape(label)}\s*:?\s*$")
        for i, line in enumerate(lines):
            m = same.match(line)
            if m and not alone.match(line):
                value = m.group(1)
            elif alone.match(line):
                # The value runs until the next bilingual label (Hindi text),
                # a blank line, or three lines, whichever comes first.
                parts = []
                for nxt in lines[i + 1:i + 4]:
                    if not nxt.strip() or not _is_ascii(nxt):
                        break
                    parts.append(nxt.strip())
                value = " ".join(parts)
            else:
                continue
            value = re.sub(r"\s+", " ", value).strip(" ,")
            if value and not set(value) <= {"*"} and value.lower() not in ("n/a", "na", "nil", "-"):
                return value
    return None


def _is_ascii(line: str) -> bool:
    return all(ord(ch) < 128 for ch in line)


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

        keywords = query.keywords
        if not keywords:
            from .keywords import stock_keywords
            keywords = [k.keyword for k in stock_keywords(limit=6)]
            logger.info("GeM search keywords from stock: %s", keywords)

        folder = Path(out_dir) / "gem"
        folder.mkdir(parents=True, exist_ok=True)
        downloads = Path(tempfile.mkdtemp(prefix="gem-dl-"))
        driver = _start_browser(downloads)
        found: List[RfpCandidate] = []
        seen: set = set()
        try:
            driver.get(GEM_SEARCH_URL)
            per_keyword = max(1, -(-query.count // len(keywords)))   # share the count across keywords
            for keyword in keywords:
                if len(found) >= query.count:
                    break
                if progress:
                    progress(len(found), query.count, f"Searching GeM for “{keyword}”")
                cards = _search(driver, keyword)
                logger.info("GeM: %d bids for %r", len(cards), keyword)
                taken = 0
                for card in cards:
                    if len(found) >= query.count or taken >= per_keyword:
                        break
                    if card["bid_no"] in seen or not is_relevant(keyword, card.get("items") or ""):
                        continue
                    seen.add(card["bid_no"])
                    time.sleep(PAUSE_SECONDS)
                    path = folder / f"{card['bid_no'].replace('/', '_')}.pdf"
                    if not path.exists():
                        try:
                            _download(driver, card["url"], downloads, path)
                        except Exception as exc:   # one bad bid must not stop the search
                            logger.warning("GeM: could not download %s: %s", card["bid_no"], exc)
                            continue
                    found.append(_candidate(card, path, keyword, self.name))
                    taken += 1
                    if progress:
                        progress(len(found), query.count, f"Downloaded {card['bid_no']}")
                driver.get(GEM_SEARCH_URL)   # back to the list for the next keyword
                time.sleep(PAUSE_SECONDS)
        finally:
            driver.quit()
            shutil.rmtree(downloads, ignore_errors=True)
        return found


# ─── Browser steps ────────────────────────────────────────────────────────────

def _start_browser(download_dir: Path):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1366,900")
    options.add_experimental_option("prefs", {
        "download.default_directory": str(download_dir),
        "download.prompt_for_download": False,
        "plugins.always_open_pdf_externally": True,   # save PDFs instead of showing them
    })
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Browser.setDownloadBehavior",
                           {"behavior": "allow", "downloadPath": str(download_dir)})
    return driver


def _search(driver, keyword: str) -> List[Dict[str, str]]:
    """Type the keyword into the page's search box; return the first page of bid cards."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    wait = WebDriverWait(driver, 30)
    box = wait.until(EC.element_to_be_clickable((By.ID, "searchBid")))
    box.clear()
    box.send_keys(keyword)
    driver.find_element(By.ID, "searchBidRA").click()

    def settled(d) -> bool:   # the loader gif is gone and the list (or "no results") is drawn
        board = d.find_element(By.ID, "bidCard")
        return (not board.find_elements(By.CSS_SELECTOR, "img[src*='gemloader']")
                and bool(board.text.strip()))
    wait.until(settled)

    cards = []
    for el in driver.find_elements(By.CSS_SELECTOR, "#bidCard .card"):
        links = [a for a in el.find_elements(By.CSS_SELECTOR, "a.bid_no_hover")
                 if _DOC_LINK.search(a.get_attribute("href") or "")]
        if not links:
            continue
        link = links[-1]          # an RA card shows its parent bid first, then itself
        text = el.text
        # Long item lists are cut to 30 characters; the full text is in the hover popover.
        popovers = el.find_elements(By.CSS_SELECTOR, "a[data-toggle='popover']")
        items = (popovers[0].get_attribute("data-content") if popovers else None) or _card_field(text, "Items:")
        cards.append({
            "bid_no": link.text.strip(),
            "url": urljoin(GEM_SEARCH_URL, link.get_attribute("href")),
            "items": items,
            "department": _card_department(text),
            "end_date": _date(_card_field(text, "End Date:")),
        })
    return cards


# GeM's search matches a word anywhere in a bid ("modular" finds a "modular
# valve"), so a bid only counts when its item category names the product.
_PRODUCT_WORDS = {
    "mcb": ("mcb", "circuit breaker"),
    "switch": ("switch",),
    "socket": ("socket",),
    "plate": ("plate",),
    "box": ("box",),
    "regulator": ("regulator",),
}


def is_relevant(keyword: str, items: str) -> bool:
    head = keyword.lower().split()[-1]
    words = _PRODUCT_WORDS.get(head, (head,))
    text = items.lower()
    return any(re.search(rf"\b{re.escape(w)}", text) for w in words)


def _download(driver, url: str, downloads: Path, dest: Path) -> None:
    """Let the browser download one PDF, then move it to *dest*."""
    before = set(downloads.iterdir())
    driver.get(url)
    deadline = time.time() + DOWNLOAD_TIMEOUT
    while time.time() < deadline:
        done = [p for p in set(downloads.iterdir()) - before if not p.name.endswith(".crdownload")]
        if done:
            data = done[0].read_bytes()
            if not data.startswith(b"%PDF"):
                done[0].unlink()
                raise ValueError("the download is not a PDF")
            shutil.move(str(done[0]), dest)
            return
        time.sleep(0.5)
    raise TimeoutError(f"no PDF after {DOWNLOAD_TIMEOUT} s")


def _card_field(text: str, label: str) -> Optional[str]:
    m = re.search(rf"{re.escape(label)}\s*(.+)", text)
    return m.group(1).strip() if m else None


def _card_department(text: str) -> Optional[str]:
    m = re.search(r"Department Name And Address:\s*\n(.+?)(?:\n(?:Start Date|End Date):|$)", text, re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None


def _candidate(card: Dict[str, Any], path: Path, keyword: str, source: str) -> RfpCandidate:
    """Details from the PDF's labelled header, falling back to what the card showed."""
    try:
        with pymupdf.open(path) as doc:
            meta = parse_gem_text("".join(page.get_text() for page in doc))
    except Exception as exc:
        logger.warning("GeM: could not read %s: %s", path.name, exc)
        meta = {}
    return RfpCandidate(
        title=meta.get("title") or card.get("items") or card["bid_no"],
        pdf_path=str(path), source=source,
        buyer=meta.get("buyer") or card.get("department"),
        # The card shows today's end date; the PDF keeps the original, before auto-extensions.
        deadline=card.get("end_date") or meta.get("deadline"),
        published=meta.get("published"),
        location=meta.get("location"),
        category=meta.get("category") or card.get("items"),
        reference=meta.get("reference") or card["bid_no"],
        meta={"keyword": keyword, "url": card["url"], "total_quantity": meta.get("total_quantity")},
    )

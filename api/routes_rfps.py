"""RFP documents: upload, list, view."""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from db import Rfp, User
from discovery.metadata import extract_metadata

from .context import AppContext, current_user, get_ctx, get_db
from .schemas import RfpOut

router = APIRouter(prefix="/api/rfps", tags=["rfps"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


_GENERIC_HEADING = re.compile(
    r"^(request for (proposal|quotation)|rfp|rfq|tender( document)?|bid document|notice inviting tender)"
    r"\s*(\([^)]*\))?\s*[:.\-]?$",
    re.I,
)


def _guess_title(text: str, fallback: str) -> str:
    """
    First substantial line of the document, skipping generic headings like
    "Request for Proposal (RFP)" so the title says what is being bought.
    """
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < 12 or line.lower().startswith(("page ", "dated")) or _GENERIC_HEADING.match(line):
            continue
        return line[:200]
    return fallback


def read_pdf(data: bytes) -> Tuple[int, str]:
    """(pages, text) of a PDF, or HTTP 422 with a clear reason."""
    if not data.startswith(b"%PDF"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "The file is not a PDF")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pages, text = doc.page_count, "".join(page.get_text() for page in doc)
    except Exception:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "The PDF could not be opened")
    if len(text.strip()) < 50:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "No readable text in this PDF. If it is a scanned document it needs OCR first.",
        )
    return pages, text


def register_rfp(
    db: Session,
    *,
    data: bytes,
    pdf_path: Path,
    filename: str,
    source: str,
    user_id: Optional[int],
    known: Optional[Dict[str, Any]] = None,
) -> Tuple[Rfp, bool]:
    """
    Save an RFP row for this PDF (deduplicated by content). Metadata comes
    from *known* (discovery already knows it) or is extracted from the text.
    Returns (rfp, created).
    """
    sha = hashlib.sha256(data).hexdigest()
    existing = db.scalar(select(Rfp).where(Rfp.sha256 == sha))
    if existing:
        return existing, False

    pages, text = read_pdf(data)
    if not pdf_path.is_file():
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(data)

    meta = dict(known) if known is not None else extract_metadata(text)
    value = meta.get("estimated_value")
    rfp = Rfp(
        id=uuid.uuid4().hex[:12],
        title=meta.get("title") or _guess_title(text, Path(filename).stem),
        source=source,
        filename=filename,
        pdf_path=str(pdf_path),
        sha256=sha,
        pages=pages,
        buyer=meta.get("buyer"),
        deadline=meta.get("deadline"),
        location=meta.get("location"),
        category=meta.get("category"),
        estimated_value=float(value) if value is not None else None,
        meta={k: v for k, v in meta.items()
              if k not in {"title", "buyer", "deadline", "location", "category", "estimated_value"}},
        created_by=user_id,
    )
    db.add(rfp)
    db.commit()
    return rfp, True


@router.post("/upload", response_model=RfpOut, status_code=status.HTTP_201_CREATED)
def upload_rfp(
    response: Response,
    file: UploadFile = File(...),
    ctx: AppContext = Depends(get_ctx),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> Rfp:
    # A plain (non-async) endpoint: FastAPI runs it on a worker thread, so the
    # metadata LLM call below does not block other requests.
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "PDF is larger than 25 MB")
    sha = hashlib.sha256(data).hexdigest()
    rfp, created = register_rfp(
        db, data=data, pdf_path=ctx.uploads_dir / f"{sha}.pdf",
        filename=Path(file.filename or "rfp.pdf").name, source="upload", user_id=user.id,
    )
    if not created:
        response.status_code = status.HTTP_200_OK   # same document uploaded before
    return rfp


@router.get("", response_model=List[RfpOut])
def list_rfps(
    q: Optional[str] = None,
    source: Optional[str] = None,
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
) -> List[Rfp]:
    stmt = select(Rfp).order_by(Rfp.created_at.desc())
    if source:
        stmt = stmt.where(Rfp.source == source)
    rfps = list(db.scalars(stmt))
    if q:
        needle = q.lower()
        rfps = [r for r in rfps if needle in " ".join(
            filter(None, [r.title, r.buyer, r.location, r.category, r.filename])).lower()]
    return rfps


def _get_rfp(db: Session, rfp_id: str) -> Rfp:
    rfp = db.get(Rfp, rfp_id)
    if rfp is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "RFP not found")
    return rfp


@router.get("/{rfp_id}", response_model=RfpOut)
def get_rfp(rfp_id: str, db: Session = Depends(get_db), _: User = Depends(current_user)) -> Rfp:
    return _get_rfp(db, rfp_id)


@router.get("/{rfp_id}/document")
def get_rfp_document(rfp_id: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
    rfp = _get_rfp(db, rfp_id)
    if not Path(rfp.pdf_path).is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "The PDF file is missing")
    return FileResponse(rfp.pdf_path, media_type="application/pdf", filename=rfp.filename,
                        content_disposition_type="inline")

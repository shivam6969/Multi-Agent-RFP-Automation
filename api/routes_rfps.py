"""RFP documents: upload, list, view."""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import List, Optional

import pymupdf
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from db import Rfp, User

from .context import AppContext, current_user, get_ctx, get_db
from .schemas import RfpOut

router = APIRouter(prefix="/api/rfps", tags=["rfps"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _guess_title(text: str, fallback: str) -> str:
    """First substantial line of the document, e.g. "Request for Proposal (RFP)"."""
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) >= 12 and not line.lower().startswith(("page ", "dated")):
            return line[:200]
    return fallback


@router.post("/upload", response_model=RfpOut, status_code=status.HTTP_201_CREATED)
async def upload_rfp(
    response: Response,
    file: UploadFile = File(...),
    ctx: AppContext = Depends(get_ctx),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> Rfp:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "PDF is larger than 25 MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "The file is not a PDF")

    sha = hashlib.sha256(data).hexdigest()
    existing = db.scalar(select(Rfp).where(Rfp.sha256 == sha))
    if existing:
        response.status_code = status.HTTP_200_OK   # same document uploaded before
        return existing

    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pages = doc.page_count
            text = "".join(page.get_text() for page in doc)
    except Exception:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "The PDF could not be opened")
    if len(text.strip()) < 50:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "No readable text in this PDF. If it is a scanned document it needs OCR first.",
        )

    ctx.uploads_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.uploads_dir / f"{sha}.pdf"
    path.write_bytes(data)

    filename = Path(file.filename or "rfp.pdf").name
    rfp = Rfp(
        id=uuid.uuid4().hex[:12],
        title=_guess_title(text, Path(filename).stem),
        source="upload",
        filename=filename,
        pdf_path=str(path),
        sha256=sha,
        pages=pages,
        meta={},
        created_by=user.id,
    )
    db.add(rfp)
    db.commit()
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

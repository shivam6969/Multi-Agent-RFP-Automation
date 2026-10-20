"""Discovery: list sources and find RFPs (the "Find RFPs" form)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError
from sqlalchemy.orm import Session

from db import Rfp, User
from discovery.base import DiscoveryQuery, SourceUnavailable
from discovery.discovery_agent import list_sources, run_discovery

from .context import AppContext, current_user, get_ctx, get_db
from .routes_rfps import register_rfp
from .schemas import DiscoverySearchIn, RfpOut

router = APIRouter(prefix="/api/discovery", tags=["discovery"])


@router.get("/sources")
def sources(_: User = Depends(current_user)) -> List[Dict[str, Any]]:
    return list_sources()


@router.post("/search", response_model=List[RfpOut])
def search(
    body: DiscoverySearchIn,
    ctx: AppContext = Depends(get_ctx),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> List[Rfp]:
    """Run the Discovery Agent and register every RFP it finds."""
    try:
        query = DiscoveryQuery.model_validate(body.model_dump(exclude={"source"}))
    except ValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    try:
        found = run_discovery(query, body.source, out_dir=str(ctx.discovery_dir))
    except SourceUnavailable as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))

    rfps: List[Rfp] = []
    for c in found:
        path = Path(c.pdf_path)
        known = {k: v for k, v in c.to_dict().items() if k not in {"pdf_path", "source", "meta"} and v is not None}
        known.update(c.meta)
        rfp, _ = register_rfp(db, data=path.read_bytes(), pdf_path=path, filename=path.name,
                              source=c.source, user_id=user.id, known=known)
        rfps.append(rfp)
    return rfps

"""Settings: the company profile used on proposals ("Prepared by")."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends

from company_profile import CompanyProfile, get_profile, save_profile
from db import User

from .context import AppContext, current_user, get_ctx
from .schemas import CompanyProfileIn

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("/company")
def read_company(ctx: AppContext = Depends(get_ctx), _: User = Depends(current_user)) -> Dict[str, Any]:
    return get_profile(ctx.engine).model_dump()


@router.put("/company")
def update_company(body: CompanyProfileIn, ctx: AppContext = Depends(get_ctx),
                   _: User = Depends(current_user)) -> Dict[str, Any]:
    profile = CompanyProfile(**body.model_dump(), is_demo=False)
    return save_profile(profile, ctx.engine).model_dump()

"""
company_profile.py
------------------
The vendor's own details: who the proposal is "Prepared by". Edited on the
dashboard's Settings page and stored in the app_settings table.

The default is a clearly fictitious demo company; replace it in Settings.
A project keeps a snapshot of the profile from when its proposal was
drafted, so editing Settings later doesn't silently change old proposals.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from db import AppSetting, get_engine
from settings import DB_PATH

_KEY = "company_profile"


class CompanyProfile(BaseModel):
    name: str = Field(default="Demo Electricals Pvt. Ltd.", min_length=1, max_length=200)
    tagline: str = Field(default="Supplier of modular switches, sockets and wiring accessories", max_length=200)
    about: str = Field(
        default=("We supply modular switches, sockets, plates and wiring accessories to builders, "
                 "hotels, schools and offices, with stock held locally for quick delivery."),
        max_length=2000,
    )
    brands: str = Field(default="Havells", max_length=200)
    years_in_business: Optional[int] = Field(default=10, ge=0, le=200)
    address: str = Field(default="Plot 12, Sector 18, Gurugram, Haryana 122015", max_length=300)
    gstin: str = Field(default="", max_length=20)
    email: str = Field(default="sales@demo-electricals.example", max_length=200)
    phone: str = Field(default="+91 00000 00000", max_length=40)
    website: str = Field(default="", max_length=200)
    signatory_name: str = Field(default="Authorised Signatory", max_length=120)
    signatory_title: str = Field(default="Sales Manager", max_length=120)
    is_demo: bool = True   # set to False once real details are saved


def get_profile(engine: Optional[Engine] = None) -> CompanyProfile:
    with Session(engine or get_engine(DB_PATH)) as db:
        row = db.get(AppSetting, _KEY)
        return CompanyProfile.model_validate(row.value) if row else CompanyProfile()


def save_profile(profile: CompanyProfile, engine: Optional[Engine] = None) -> CompanyProfile:
    with Session(engine or get_engine(DB_PATH)) as db:
        db.merge(AppSetting(key=_KEY, value=profile.model_dump()))
        db.commit()
    return profile

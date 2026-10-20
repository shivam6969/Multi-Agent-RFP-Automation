"""Request/response models for the API (also documented at /docs)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ─── Auth ─────────────────────────────────────────────────────────────────────

class RegisterIn(BaseModel):
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=254)
    name: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=8, max_length=256)


class LoginIn(BaseModel):
    email: str
    password: str
    remember: bool = False


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str
    name: str


# ─── RFPs ─────────────────────────────────────────────────────────────────────

class RfpOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    source: str
    filename: str
    pages: int
    buyer: Optional[str] = None
    deadline: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None
    estimated_value: Optional[float] = None
    created_at: datetime


# ─── Projects ─────────────────────────────────────────────────────────────────

Stage = Literal["queued", "running", "requirements_review", "matching_review",
                "pricing_review", "failed", "done"]


class ProjectCreateIn(BaseModel):
    rfp_id: str


class ProjectSummaryOut(BaseModel):
    id: str
    rfp: RfpOut
    stage: str
    running: bool
    progress: Optional[str] = None
    error: Optional[str] = None
    grand_total: Optional[float] = None
    fulfillment_score: Optional[float] = None
    outcome: Optional[str] = None
    proposal_available: bool = False
    created_at: datetime
    updated_at: datetime


class ProjectOut(ProjectSummaryOut):
    failed_step: Optional[str] = None
    warnings: List[str] = []
    # Working data for the current screen (whatever the workflow has so far).
    rfp_summary: Optional[str] = None
    requirements: List[Dict[str, Any]] = []
    matched_items: List[Dict[str, Any]] = []
    match_summary: Dict[str, Any] = {}
    line_items: List[Dict[str, Any]] = []
    pricing_summary: Dict[str, Any] = {}
    unpriced: List[Dict[str, Any]] = []
    review_log: List[Dict[str, Any]] = []


class ReviewIn(BaseModel):
    """The reviewer's decision for the current stage; see workflow.DECISION_SCHEMAS."""
    decision: Dict[str, Any] = {}


class PricingPreviewIn(BaseModel):
    margin_pct: Optional[float] = Field(default=None, ge=0)
    advance_payment: Optional[bool] = None
    freight_inr: Optional[float] = Field(default=None, ge=0)
    quantities: Dict[str, int] = {}


class PricingPreviewOut(BaseModel):
    line_items: List[Dict[str, Any]]
    pricing_summary: Dict[str, Any]


class OutcomeIn(BaseModel):
    outcome: Optional[Literal["submitted", "won", "lost"]] = None


# ─── Catalogue ────────────────────────────────────────────────────────────────

class ProductOut(BaseModel):
    product_id: str
    product_name: str
    product_range: str
    product_variant: Optional[str]
    sap_code: str
    hsn_code: str
    packing: str
    unit_price: float
    stock_on_hand: Optional[int]
    monthly_available: Optional[int]
    lead_time_days: Optional[int]
    stock_is_demo: bool

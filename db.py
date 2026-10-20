"""
db.py
-----
SQLite database: the single source of truth for structured data.

Tables
------
products  one row per catalogue item (price stored as a number)
stock     synthetic demo stock per product (is_demo = True)
meta      key/value bookkeeping (e.g. hash of the catalogue file last loaded)
users     dashboard accounts (scrypt password hashes, never plain text)
sessions  login sessions (only a hash of the cookie token is stored)
rfps      RFP documents: uploaded, generated samples or discovered
projects  one review workflow per RFP; summary columns for lists/dashboard.
          The full working state lives in the LangGraph checkpoint
          (thread_id = projects.id), see workflow.py.

FAISS indexes are a cache built from these tables, never the other way round.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from settings import DB_PATH


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"

    # SAP codes are not unique in the source data (3 duplicates), so each
    # product gets a stable id derived from range + variant + item name.
    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    sap_code: Mapped[str] = mapped_column(String(32), index=True)
    brand: Mapped[str] = mapped_column(String(64))
    range_name: Mapped[str] = mapped_column(String(128))
    variant: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    variant_type: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    item_name: Mapped[str] = mapped_column(String(256))
    hsn_code: Mapped[str] = mapped_column(String(16))
    packing: Mapped[str] = mapped_column(String(32))
    unit_price: Mapped[float] = mapped_column(Float)
    price_raw: Mapped[str] = mapped_column(String(32))


class Stock(Base):
    __tablename__ = "stock"

    product_id: Mapped[str] = mapped_column(ForeignKey("products.id"), primary_key=True)
    on_hand: Mapped[int] = mapped_column(Integer)
    monthly_capacity: Mapped[int] = mapped_column(Integer)
    lead_time_days: Mapped[int] = mapped_column(Integer)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=True)


class Meta(Base):
    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(256))


class AppSetting(Base):
    """Small JSON settings edited in the UI (e.g. the company profile)."""
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[Dict[str, Any]] = mapped_column(JSON)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    password_hash: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuthSession(Base):
    __tablename__ = "sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Rfp(Base):
    __tablename__ = "rfps"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(String(512))
    source: Mapped[str] = mapped_column(String(16))          # upload | sample | gem
    filename: Mapped[str] = mapped_column(String(256))
    pdf_path: Mapped[str] = mapped_column(String(1024))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    pages: Mapped[int] = mapped_column(Integer, default=0)
    buyer: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    deadline: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)   # ISO date
    location: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    estimated_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_by: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    rfp_id: Mapped[str] = mapped_column(ForeignKey("rfps.id"), index=True)
    owner_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    # Cached from the workflow so lists don't need to open every checkpoint.
    stage: Mapped[str] = mapped_column(String(32), default="queued")
    running: Mapped[bool] = mapped_column(Boolean, default=False)
    progress: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    grand_total: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fulfillment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    proposal_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    outcome: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)   # submitted | won | lost
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


@lru_cache(maxsize=None)
def get_engine(db_path: str = DB_PATH) -> Engine:
    """Return a (cached) engine for *db_path*, creating the file and tables."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return engine

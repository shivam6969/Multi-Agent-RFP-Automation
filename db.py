"""
db.py
-----
SQLite database: the single source of truth for structured data.

Tables so far
-------------
products  one row per catalogue item (price stored as a number)
stock     synthetic demo stock per product (is_demo = True)
meta      key/value bookkeeping (e.g. hash of the catalogue file last loaded)

Users, RFPs, projects and proposals will be added here in later milestones.
FAISS indexes are a cache built from these tables, never the other way round.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, create_engine
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

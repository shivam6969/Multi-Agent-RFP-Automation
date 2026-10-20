"""Shared types for discovery sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field, field_validator

ProgressFn = Callable[[int, int, str], None]   # (done, total, message)


class DiscoveryQuery(BaseModel):
    """What the user asked for on the New Session screen."""
    keywords: List[str] = []
    count: int = Field(default=6, ge=1, le=20)
    location: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    category: Optional[str] = None
    min_value: Optional[float] = Field(default=None, ge=0)
    max_value: Optional[float] = Field(default=None, ge=0)

    @field_validator("keywords", mode="before")
    @classmethod
    def _clean_keywords(cls, value: Any) -> List[str]:
        if isinstance(value, str):
            value = value.split(",")
        seen: List[str] = []
        for kw in value or []:
            kw = str(kw).strip()
            if kw and kw.lower() not in (s.lower() for s in seen):
                seen.append(kw)
        return seen[:10]


@dataclass
class RfpCandidate:
    """One RFP found by a source, with its PDF saved locally."""
    title: str
    pdf_path: str
    source: str
    buyer: Optional[str] = None
    deadline: Optional[str] = None          # ISO date
    published: Optional[str] = None         # ISO date
    location: Optional[str] = None
    category: Optional[str] = None
    estimated_value: Optional[float] = None
    reference: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SourceUnavailable(RuntimeError):
    """The source is switched off or its dependencies are missing."""


class DiscoverySource(Protocol):
    name: str
    label: str

    def available(self) -> Optional[str]:
        """None if the source can run, else a human-readable reason."""

    def search(self, query: DiscoveryQuery, out_dir: str,
               progress: Optional[ProgressFn] = None) -> List[RfpCandidate]:
        ...

"""
Dashboard data, computed from the projects and rfps tables only (no LLM calls,
no checkpoint reads): KPI cards, reminders, the deadline calendar, monthly
generated-vs-won activity and the pending RFPs list.

build_dashboard() is a pure function of the rows and today's date so it is
easy to test. Projects are shared by the whole team, like the project lists.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from db import Project, Rfp, User

from .context import current_user, get_db

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

CLOSING_SOON_DAYS = 7      # pending RFP badge
NEW_RFP_DAYS = 3           # pending RFP badge
UPCOMING_DAYS = 14         # unstarted RFPs show up as reminders this far ahead
ACTIVITY_MONTHS = 6
MAX_REMINDERS = 6
MAX_PENDING = 6

REVIEW_LABEL = {
    "requirements_review": "Review requirements",
    "matching_review": "Review product matches",
    "pricing_review": "Review pricing",
    "proposal_review": "Review proposal",
}

# Reminder order: the most urgent first, then by date.
STATUS_ORDER = ["overdue", "due_today", "failed", "pending", "ready", "upcoming"]


def local_date(dt: Optional[datetime]) -> Optional[date]:
    """SQLite returns naive UTC datetimes; convert to the laptop's local date."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().date()


def parse_deadline(value: Optional[str]) -> Optional[date]:
    try:
        return date.fromisoformat((value or "")[:10])
    except ValueError:
        return None


def _month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def _months_back(today: date, n: int) -> List[date]:
    """First day of the last *n* months, oldest first, including this month."""
    first = today.replace(day=1)
    months = []
    for _ in range(n):
        months.append(first)
        first = (first - timedelta(days=1)).replace(day=1)
    return months[::-1]


def _deadline_status(deadline: Optional[date], today: date) -> Optional[str]:
    if deadline is None:
        return None
    if deadline < today:
        return "overdue"
    if deadline == today:
        return "due_today"
    return None


def _count_by_month(dates: Iterable[Optional[date]], today: date) -> Dict[str, int]:
    this_month = today.replace(day=1)
    last_month = (this_month - timedelta(days=1)).replace(day=1)
    keys = {"this_month": _month_key(this_month), "last_month": _month_key(last_month)}
    counts = {"this_month": 0, "last_month": 0}
    for d in dates:
        if d is None:
            continue
        for name, key in keys.items():
            if _month_key(d) == key:
                counts[name] += 1
    return counts


def build_dashboard(projects: List[Project], rfps: List[Rfp], today: date) -> Dict[str, Any]:
    rfp_by_id = {r.id: r for r in rfps}
    # One RFP can have several projects (e.g. restarted); the newest one counts.
    latest: Dict[str, Project] = {}
    for p in projects:
        current = latest.get(p.rfp_id)
        if current is None or (p.created_at or datetime.min) > (current.created_at or datetime.min):
            latest[p.rfp_id] = p

    generated = [p for p in projects if p.stage == "done"]
    waiting = [p for p in projects if not p.running and p.stage in REVIEW_LABEL]
    decided = [p for p in generated if p.outcome in ("won", "lost")]
    won = [p for p in decided if p.outcome == "won"]

    def deadline_of(p: Project) -> Optional[date]:
        rfp = rfp_by_id.get(p.rfp_id)
        return parse_deadline(rfp.deadline) if rfp else None

    soon = today + timedelta(days=3)
    kpis = {
        "proposals_generated": {
            "value": len(generated),
            **_count_by_month((local_date(p.completed_at) for p in generated), today),
        },
        "awaiting_review": {
            "value": len(waiting),
            "urgent": sum(1 for p in waiting if (d := deadline_of(p)) is not None and d <= soon),
        },
        "rfps_received": {
            "value": len(rfps),
            **_count_by_month((local_date(r.created_at) for r in rfps), today),
        },
        "win_rate": {
            "value": round(100 * len(won) / len(decided), 1) if decided else None,
            "won": len(won),
            "lost": len(decided) - len(won),
            "awaiting_outcome": sum(1 for p in generated if p.outcome in (None, "submitted")),
            "won_value": round(sum(p.grand_total or 0 for p in won), 2),
        },
    }

    reminders: List[Dict[str, Any]] = []
    for rfp in rfps:
        deadline = parse_deadline(rfp.deadline)
        project = latest.get(rfp.id)
        base = {"rfp_id": rfp.id, "rfp_title": rfp.title, "deadline": deadline.isoformat() if deadline else None,
                "project_id": project.id if project else None}
        overdue_or_today = _deadline_status(deadline, today)
        if project is None:
            if deadline is not None and today <= deadline <= today + timedelta(days=UPCOMING_DAYS):
                reminders.append({**base, "kind": "start", "title": f"Start a response: {rfp.title}",
                                  "status": overdue_or_today or "upcoming"})
        elif project.running:
            continue
        elif project.stage in REVIEW_LABEL:
            reminders.append({**base, "kind": "review",
                              "title": f"{REVIEW_LABEL[project.stage]}: {rfp.title}",
                              "status": overdue_or_today or "pending"})
        elif project.stage == "failed":
            reminders.append({**base, "kind": "failed", "title": f"Retry the failed step: {rfp.title}",
                              "status": "failed"})
        elif project.stage == "done" and project.outcome is None:
            reminders.append({**base, "kind": "submit", "title": f"Submit the proposal: {rfp.title}",
                              "status": overdue_or_today or "ready"})
        elif project.stage == "done" and project.outcome == "submitted":
            reminders.append({**base, "kind": "outcome", "title": f"Record the result: {rfp.title}",
                              "status": "pending" if deadline is None or deadline < today else "upcoming"})
    reminders.sort(key=lambda r: (STATUS_ORDER.index(r["status"]), r["deadline"] or "9999-12-31"))

    deadlines = []
    for rfp in rfps:
        deadline = parse_deadline(rfp.deadline)
        if deadline is None:
            continue
        project = latest.get(rfp.id)
        deadlines.append({
            "date": deadline.isoformat(),
            "rfp_id": rfp.id,
            "title": rfp.title,
            "buyer": rfp.buyer,
            "project_id": project.id if project else None,
            "stage": project.stage if project else None,
            "outcome": project.outcome if project else None,
        })
    deadlines.sort(key=lambda d: d["date"])

    by_month = {_month_key(m): {"month": _month_key(m), "generated": 0, "won": 0}
                for m in _months_back(today, ACTIVITY_MONTHS)}
    for p in generated:
        bucket = by_month.get(_month_key(local_date(p.completed_at) or today))
        if bucket is not None:
            bucket["generated"] += 1
            bucket["won"] += p.outcome == "won"

    pending = []
    for rfp in rfps:
        project = latest.get(rfp.id)
        if project is not None and project.stage == "done":
            continue
        deadline = parse_deadline(rfp.deadline)
        if deadline is not None and deadline < today:
            continue   # missed; it still shows on the calendar
        if deadline is not None and deadline <= today + timedelta(days=CLOSING_SOON_DAYS):
            tag = "closing_soon"
        elif project is not None:
            tag = "in_progress"
        elif (created := local_date(rfp.created_at)) is not None and created >= today - timedelta(days=NEW_RFP_DAYS):
            tag = "new"
        else:
            tag = "open"
        pending.append({
            "rfp_id": rfp.id,
            "title": rfp.title,
            "buyer": rfp.buyer,
            "source": rfp.source,
            "deadline": deadline.isoformat() if deadline else None,
            "tag": tag,
            "project_id": project.id if project else None,
            "stage": project.stage if project else None,
        })
    pending.sort(key=lambda r: (r["deadline"] or "9999-12-31", r["title"]))

    return {
        "today": today.isoformat(),
        "kpis": kpis,
        "reminders": reminders[:MAX_REMINDERS],
        "reminders_total": len(reminders),
        "deadlines": deadlines,
        "activity": list(by_month.values()),
        "pending_rfps": pending[:MAX_PENDING],
        "pending_total": len(pending),
    }


@router.get("")
def get_dashboard(db: Session = Depends(get_db), _: User = Depends(current_user)) -> Dict[str, Any]:
    projects = list(db.scalars(select(Project)))
    rfps = list(db.scalars(select(Rfp)))
    return build_dashboard(projects, rfps, date.today())

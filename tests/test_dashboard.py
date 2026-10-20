from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

from sqlalchemy import inspect

from api.routes_dashboard import build_dashboard, parse_deadline
from db import Project, Rfp, get_engine

TODAY = date(2026, 10, 20)


def at(d: date) -> datetime:
    """06:00 UTC, so the local date is the same day in India and Europe."""
    return datetime(d.year, d.month, d.day, 6, 0)


def rfp(rid: str, deadline=None, created=TODAY - timedelta(days=30)) -> Rfp:
    return Rfp(id=rid, title=f"Tender {rid}", source="sample", filename=f"{rid}.pdf",
               pdf_path=f"{rid}.pdf", sha256=rid, deadline=deadline.isoformat() if deadline else None,
               buyer=f"Buyer {rid}", created_at=at(created))


def project(pid: str, rfp_id: str, stage: str, *, outcome=None, completed=None, total=None,
            running=False, created=TODAY - timedelta(days=20)) -> Project:
    return Project(id=pid, rfp_id=rfp_id, stage=stage, outcome=outcome, running=running,
                   grand_total=total, created_at=at(created),
                   completed_at=at(completed) if completed else None)


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        last_month = TODAY.replace(day=1) - timedelta(days=3)
        self.rfps = [
            rfp("a", deadline=TODAY + timedelta(days=2), created=TODAY - timedelta(days=1)),   # unstarted, soon
            rfp("b", deadline=TODAY),                                                          # pricing review
            rfp("c", deadline=TODAY - timedelta(days=4)),                                      # done, not submitted
            rfp("d", deadline=TODAY + timedelta(days=30)),                                     # won
            rfp("e", deadline=TODAY + timedelta(days=40)),                                     # lost, last month
            rfp("f", deadline=None),                                                           # failed
            rfp("g", deadline=TODAY - timedelta(days=10)),                                     # missed, no project
            rfp("h", deadline=TODAY + timedelta(days=60)),                                     # unstarted, far
        ]
        self.projects = [
            project("pb", "b", "pricing_review"),
            project("pc", "c", "done", completed=TODAY - timedelta(days=5), total=1000),
            project("pd", "d", "done", outcome="won", completed=TODAY - timedelta(days=2), total=250000),
            project("pe", "e", "done", outcome="lost", completed=last_month, total=5000),
            project("pf", "f", "failed"),
        ]
        self.dash = build_dashboard(self.projects, self.rfps, TODAY)

    def test_kpis(self) -> None:
        k = self.dash["kpis"]
        self.assertEqual(k["proposals_generated"], {"value": 3, "this_month": 2, "last_month": 1})
        self.assertEqual(k["awaiting_review"], {"value": 1, "urgent": 1})
        self.assertEqual(k["rfps_received"]["value"], 8)
        self.assertEqual(k["win_rate"]["value"], 50.0)
        self.assertEqual((k["win_rate"]["won"], k["win_rate"]["lost"]), (1, 1))
        self.assertEqual(k["win_rate"]["awaiting_outcome"], 1)
        self.assertEqual(k["win_rate"]["won_value"], 250000)

    def test_no_decided_proposals_means_no_win_rate(self) -> None:
        dash = build_dashboard([], [], TODAY)
        self.assertIsNone(dash["kpis"]["win_rate"]["value"])
        self.assertEqual(dash["reminders"], [])
        self.assertEqual(len(dash["activity"]), 6)

    def test_reminders_are_derived_and_ordered_by_urgency(self) -> None:
        got = [(r["kind"], r["rfp_id"], r["status"]) for r in self.dash["reminders"]]
        self.assertEqual(got, [
            ("submit", "c", "overdue"),
            ("review", "b", "due_today"),
            ("failed", "f", "failed"),
            ("start", "a", "upcoming"),
        ])
        review = self.dash["reminders"][1]
        self.assertEqual(review["project_id"], "pb")
        self.assertIn("Review pricing", review["title"])

    def test_calendar_has_every_deadline(self) -> None:
        ids = [d["rfp_id"] for d in self.dash["deadlines"]]
        self.assertEqual(ids, ["g", "c", "b", "a", "d", "e", "h"])
        d = next(x for x in self.dash["deadlines"] if x["rfp_id"] == "d")
        self.assertEqual((d["project_id"], d["stage"], d["outcome"]), ("pd", "done", "won"))

    def test_activity_is_six_months_generated_vs_won(self) -> None:
        act = self.dash["activity"]
        self.assertEqual([m["month"] for m in act],
                         ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"])
        self.assertEqual(act[-1], {"month": "2026-10", "generated": 2, "won": 1})
        self.assertEqual(act[-2], {"month": "2026-09", "generated": 1, "won": 0})

    def test_pending_rfps(self) -> None:
        got = [(r["rfp_id"], r["tag"]) for r in self.dash["pending_rfps"]]
        # Done (c, d, e) and missed (g) RFPs are left out; soonest deadline first.
        self.assertEqual(got, [("b", "closing_soon"), ("a", "closing_soon"), ("h", "open"), ("f", "in_progress")])

    def test_newest_project_per_rfp_wins(self) -> None:
        old = project("old", "b", "failed", created=TODAY - timedelta(days=40))
        dash = build_dashboard(self.projects + [old], self.rfps, TODAY)
        self.assertNotIn("old", [r["project_id"] for r in dash["reminders"]])

    def test_new_tag_for_recent_uploads(self) -> None:
        dash = build_dashboard([], [rfp("n", deadline=TODAY + timedelta(days=20), created=TODAY)], TODAY)
        self.assertEqual(dash["pending_rfps"][0]["tag"], "new")

    def test_parse_deadline(self) -> None:
        self.assertEqual(parse_deadline("2026-11-02"), date(2026, 11, 2))
        self.assertEqual(parse_deadline("2026-11-02T17:00:00"), date(2026, 11, 2))
        self.assertIsNone(parse_deadline("next Friday"))
        self.assertIsNone(parse_deadline(None))


class MigrationTests(unittest.TestCase):
    def test_old_database_gets_new_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "old.db")
            con = sqlite3.connect(path)
            con.execute("CREATE TABLE projects (id VARCHAR(32) PRIMARY KEY, rfp_id VARCHAR(32), stage VARCHAR(32), "
                        "running BOOLEAN, outcome VARCHAR(16), created_at DATETIME, updated_at DATETIME)")
            con.execute("INSERT INTO projects VALUES ('p1', 'r1', 'done', 0, NULL, "
                        "'2026-09-01 10:00:00', '2026-09-02 10:00:00')")
            con.execute("INSERT INTO projects VALUES ('p2', 'r1', 'pricing_review', 0, NULL, "
                        "'2026-09-01 10:00:00', '2026-09-03 10:00:00')")
            con.commit()
            con.close()

            engine = get_engine(path)
            try:
                cols = {c["name"] for c in inspect(engine).get_columns("projects")}
                self.assertTrue({"completed_at", "outcome_at", "grand_total"} <= cols)
                with engine.connect() as conn:
                    rows = dict(conn.exec_driver_sql("SELECT id, completed_at FROM projects").all())
                self.assertEqual(rows["p1"], "2026-09-02 10:00:00")   # backfilled from updated_at
                self.assertIsNone(rows["p2"])
                engine.dispose()
                get_engine.cache_clear()
                again = get_engine(path)   # running it again is a no-op
                again.dispose()
            finally:
                engine.dispose()
                get_engine.cache_clear()


if __name__ == "__main__":
    unittest.main()

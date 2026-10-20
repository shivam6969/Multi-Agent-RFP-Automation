import { ArrowDownRight, ArrowUpRight, Minus, Plus } from "lucide-react";
import type { ReactNode } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useDashboard, useMe } from "../api/hooks";
import type { Dashboard, MonthCounts, PendingRfp } from "../api/types";
import ActivityChart from "../components/ActivityChart";
import DeadlineCalendar from "../components/DeadlineCalendar";
import { STAGE_LABEL } from "../components/ProjectsTable";
import ReminderList, { reminderLink } from "../components/Reminders";
import { Badge, Button, cx, ErrorBox, PageHeader, Spinner } from "../components/ui";
import { formatDay, formatINRShort } from "../lib/format";

function greeting() {
  const h = new Date().getHours();
  return h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";
}

type Trend = "up" | "down" | "flat" | "warn";

function Kpi({ label, value, trend, note, accent }: {
  label: string;
  value: ReactNode;
  trend: Trend;
  note: ReactNode;
  accent?: boolean;
}) {
  const Arrow = trend === "down" ? ArrowDownRight : trend === "flat" ? Minus : ArrowUpRight;
  return (
    <div className="card p-5">
      <div className="text-sm text-slate-500">{label}</div>
      <div className={cx("mt-2 text-3xl font-semibold", accent ? "text-brand-600" : "text-slate-800")}>{value}</div>
      <div
        className={cx(
          "mt-2 flex items-center gap-1 text-xs",
          { up: "text-emerald-600", down: "text-rose-600", flat: "text-slate-400", warn: "text-amber-600" }[trend],
        )}
      >
        <Arrow className="h-3.5 w-3.5" />
        <span>{note}</span>
      </div>
    </div>
  );
}

/** "+40% vs last month" when there is a last month to compare with, else "+3 this month". */
function monthTrend(c: MonthCounts): { trend: Trend; note: ReactNode } {
  if (c.last_month > 0) {
    const change = Math.round(((c.this_month - c.last_month) / c.last_month) * 100);
    return {
      trend: change > 0 ? "up" : change < 0 ? "down" : "flat",
      note: <><b className="font-semibold">{change > 0 ? "+" : ""}{change}%</b> <span className="text-slate-400">vs last month</span></>,
    };
  }
  return {
    trend: c.this_month > 0 ? "up" : "flat",
    note: <><b className="font-semibold">+{c.this_month}</b> <span className="text-slate-400">this month</span></>,
  };
}

function Kpis({ kpis }: { kpis: Dashboard["kpis"] }) {
  const { proposals_generated: gen, awaiting_review: wait, rfps_received: rfps, win_rate: win } = kpis;
  const decided = win.won + win.lost;
  return (
    <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <Kpi label="Proposals Generated" value={gen.value} {...monthTrend(gen)} />
      <Kpi
        label="Awaiting Review"
        value={wait.value}
        trend={wait.urgent ? "warn" : "flat"}
        note={wait.urgent
          ? <><b className="font-semibold">{wait.urgent}</b> <span className="text-slate-400">due within 3 days</span></>
          : <span>nothing urgent</span>}
      />
      <Kpi label="RFPs Received" value={rfps.value} {...monthTrend(rfps)} />
      <Kpi
        label="Win Rate"
        value={win.value == null ? "—" : `${win.value}%`}
        accent
        trend={win.value == null ? "flat" : "up"}
        note={decided
          ? <><b className="font-semibold">{win.won} won</b> <span className="text-slate-400">of {decided} decided · {formatINRShort(win.won_value)}</span></>
          : <span>mark outcomes in History</span>}
      />
    </div>
  );
}

const TAG: Record<PendingRfp["tag"], { label: string; tone: "blue" | "amber" | "violet" | "slate" }> = {
  new: { label: "New", tone: "blue" },
  closing_soon: { label: "Closing soon", tone: "amber" },
  in_progress: { label: "In progress", tone: "violet" },
  open: { label: "Open", tone: "slate" },
};

function PendingRfps({ rfps, total }: { rfps: PendingRfp[]; total: number }) {
  const navigate = useNavigate();
  return (
    <div className="card">
      <div className="flex items-center justify-between px-5 pt-5 pb-2">
        <h2 className="font-semibold text-slate-800">Pending RFPs</h2>
        <Link to="/rfps" className="text-sm font-medium text-brand-600 hover:underline">View all</Link>
      </div>
      {rfps.length ? (
        <ul className="divide-y divide-slate-100 pb-2">
          {rfps.map((r) => (
            <li key={r.rfp_id}>
              <button
                onClick={() => navigate(reminderLink(r))}
                className="flex w-full items-start gap-3 px-5 py-2.5 text-left hover:bg-slate-50"
                title={r.stage ? STAGE_LABEL[r.stage] : "Not started"}
              >
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-medium text-slate-800">{r.title}</span>
                  <span className="block text-xs text-slate-500">
                    {r.deadline ? `Due ${formatDay(r.deadline)}` : "No deadline"}
                    {r.buyer && ` · ${r.buyer}`}
                  </span>
                </span>
                <Badge tone={TAG[r.tag].tone}>{TAG[r.tag].label}</Badge>
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="px-5 pb-5 text-sm text-slate-500">
          No open RFPs. <Link to="/new" className="font-medium text-brand-600 hover:underline">Find or upload one</Link>.
        </p>
      )}
      {total > rfps.length && (
        <div className="border-t border-slate-100 px-5 py-2 text-xs text-slate-400">{total - rfps.length} more</div>
      )}
    </div>
  );
}

export default function Home() {
  const { data: user } = useMe();
  const { data, isLoading, error } = useDashboard();
  const navigate = useNavigate();

  return (
    <>
      <PageHeader
        title="Dashboard"
        subtitle={`${greeting()}, ${user?.name.split(" ")[0] ?? ""}. Here's what's happening with your RFP workflow.`}
        actions={<Button icon={<Plus className="h-4 w-4" />} onClick={() => navigate("/new")}>Start New Session</Button>}
      />
      {isLoading ? (
        <div className="flex justify-center p-16"><Spinner /></div>
      ) : error || !data ? (
        <ErrorBox title="Couldn't load the dashboard">{error?.message ?? "Unknown error"}</ErrorBox>
      ) : (
        <>
          <Kpis kpis={data.kpis} />
          <div className="grid gap-6 lg:grid-cols-3">
            <div className="space-y-6 lg:col-span-2">
              <div className="card">
                <div className="flex items-center justify-between px-5 pt-5 pb-2">
                  <h2 className="font-semibold text-slate-800">Reminders</h2>
                  {data.reminders_total > data.reminders.length && (
                    <Link to="/projects" className="text-sm font-medium text-brand-600 hover:underline">
                      View all ({data.reminders_total})
                    </Link>
                  )}
                </div>
                <div className="pb-2">
                  <ReminderList reminders={data.reminders} today={data.today} />
                </div>
              </div>
              <ActivityChart activity={data.activity} />
            </div>
            <div className="space-y-6">
              <DeadlineCalendar deadlines={data.deadlines} today={data.today} />
              <PendingRfps rfps={data.pending_rfps} total={data.pending_total} />
            </div>
          </div>
        </>
      )}
    </>
  );
}

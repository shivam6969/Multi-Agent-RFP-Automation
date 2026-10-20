import { AlertTriangle, ClipboardCheck, Clock, PartyPopper, Send, Trophy } from "lucide-react";
import { useNavigate } from "react-router-dom";

import type { Reminder, ReminderStatus } from "../api/types";
import { formatDay, relativeDay } from "../lib/format";
import { Badge } from "./ui";

const ICON = {
  start: Clock,
  review: ClipboardCheck,
  failed: AlertTriangle,
  submit: Send,
  outcome: Trophy,
} as const;

const STATUS: Record<ReminderStatus, { label: string; tone: "red" | "amber" | "blue" | "slate" }> = {
  overdue: { label: "Overdue", tone: "red" },
  due_today: { label: "Due today", tone: "red" },
  failed: { label: "Failed", tone: "red" },
  pending: { label: "Pending", tone: "amber" },
  ready: { label: "Ready to submit", tone: "blue" },
  upcoming: { label: "Upcoming", tone: "slate" },
};

/** Statuses that light up the bell. */
export const URGENT: ReminderStatus[] = ["overdue", "due_today", "failed", "pending"];

export function reminderLink(r: Pick<Reminder, "project_id" | "rfp_id">) {
  return r.project_id ? `/projects/${r.project_id}` : `/rfps?ids=${r.rfp_id}`;
}

function when(r: Reminder, today: string) {
  if (!r.deadline) return "No submission deadline";
  return `Deadline ${formatDay(r.deadline)} · ${relativeDay(r.deadline, today)}`;
}

export default function ReminderList({
  reminders,
  today,
  onNavigate,
  compact = false,
}: {
  reminders: Reminder[];
  today: string;
  onNavigate?: () => void;
  compact?: boolean;
}) {
  const navigate = useNavigate();

  if (!reminders.length) {
    return (
      <div className="flex flex-col items-center gap-2 px-6 py-10 text-center text-sm text-slate-500">
        <PartyPopper className="h-5 w-5 text-brand-500" />
        You're all caught up. Nothing needs your attention.
      </div>
    );
  }

  return (
    <ul className="divide-y divide-slate-100">
      {reminders.map((r) => {
        const Icon = ICON[r.kind];
        const status = STATUS[r.status];
        return (
          <li key={`${r.kind}-${r.rfp_id}`}>
            <button
              onClick={() => {
                navigate(reminderLink(r));
                onNavigate?.();
              }}
              className={`flex w-full items-center gap-3 text-left hover:bg-slate-50 ${compact ? "px-3 py-2.5" : "px-5 py-3"}`}
            >
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-slate-200 text-slate-500">
                <Icon className="h-4 w-4" />
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium text-slate-800">{r.title}</span>
                <span className="block text-xs text-slate-500">{when(r, today)}</span>
              </span>
              <Badge tone={status.tone}>{status.label}</Badge>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

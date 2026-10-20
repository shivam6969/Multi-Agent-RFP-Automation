import { ChevronLeft, ChevronRight } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import type { DeadlineEntry } from "../api/types";
import { dayKey, formatDay, parseDay } from "../lib/format";
import { reminderLink } from "./Reminders";
import { cx } from "./ui";

const WEEKDAYS = ["M", "T", "W", "T", "F", "S", "S"];

/** Is this RFP still open (no finished proposal)? Open deadlines get a violet dot. */
const isOpen = (d: DeadlineEntry) => d.stage !== "done";

/** Monday-first grid of the month containing *month*, padded with blanks. */
function monthGrid(month: Date): (Date | null)[] {
  const first = new Date(month.getFullYear(), month.getMonth(), 1);
  const days = new Date(month.getFullYear(), month.getMonth() + 1, 0).getDate();
  const lead = (first.getDay() + 6) % 7;
  const cells: (Date | null)[] = Array(lead).fill(null);
  for (let d = 1; d <= days; d++) cells.push(new Date(month.getFullYear(), month.getMonth(), d));
  while (cells.length % 7) cells.push(null);
  return cells;
}

/** Today if something is due today, else the next upcoming deadline this month, else today. */
function initialSelection(deadlines: DeadlineEntry[], today: string): string {
  const thisMonth = today.slice(0, 7);
  const next = deadlines.find((d) => d.date >= today && d.date.startsWith(thisMonth) && isOpen(d));
  return next?.date ?? today;
}

export default function DeadlineCalendar({ deadlines, today }: { deadlines: DeadlineEntry[]; today: string }) {
  const navigate = useNavigate();
  const [month, setMonth] = useState(() => parseDay(today));
  const [selected, setSelected] = useState(() => initialSelection(deadlines, today));

  const byDay = useMemo(() => {
    const map = new Map<string, DeadlineEntry[]>();
    for (const d of deadlines) map.set(d.date, [...(map.get(d.date) ?? []), d]);
    return map;
  }, [deadlines]);

  const shift = (n: number) => setMonth(new Date(month.getFullYear(), month.getMonth() + n, 1));
  const due = byDay.get(selected) ?? [];

  return (
    <div className="card p-5">
      <h2 className="font-semibold text-slate-800">Deadlines</h2>
      <div className="mt-3 flex items-center justify-between">
        <span className="text-sm font-medium text-slate-700">
          {month.toLocaleDateString("en-IN", { month: "long", year: "numeric" })}
        </span>
        <div className="flex gap-1">
          <button onClick={() => shift(-1)} className="rounded p-1 text-slate-400 hover:bg-slate-100" title="Previous month">
            <ChevronLeft className="h-4 w-4" />
          </button>
          <button onClick={() => shift(1)} className="rounded p-1 text-slate-400 hover:bg-slate-100" title="Next month">
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-7 gap-y-1 text-center text-xs">
        {WEEKDAYS.map((w, i) => (
          <div key={i} className="pb-1 font-medium text-slate-400">{w}</div>
        ))}
        {monthGrid(month).map((day, i) => {
          if (!day) return <div key={i} />;
          const key = dayKey(day);
          const entries = byDay.get(key) ?? [];
          const open = entries.some(isOpen);
          return (
            <button
              key={i}
              onClick={() => setSelected(key)}
              className="flex flex-col items-center py-0.5"
              title={entries.map((e) => e.title).join("\n") || undefined}
            >
              <span
                className={cx(
                  "flex h-7 w-7 items-center justify-center rounded-full text-[13px]",
                  key === today
                    ? "bg-brand-500 font-semibold text-white"
                    : key === selected
                      ? "ring-1 ring-brand-400 text-brand-700"
                      : "text-slate-700 hover:bg-slate-100",
                )}
              >
                {day.getDate()}
              </span>
              <span
                className={cx(
                  "mt-0.5 h-1 w-1 rounded-full",
                  entries.length ? (open ? "bg-brand-500" : "bg-slate-300") : "bg-transparent",
                )}
              />
            </button>
          );
        })}
      </div>

      <div className="mt-3 space-y-2">
        {due.length ? (
          due.map((d) => (
            <button
              key={d.rfp_id}
              onClick={() => navigate(reminderLink(d))}
              className="block w-full rounded-lg border border-brand-100 bg-brand-50 px-3 py-2 text-left hover:border-brand-300"
            >
              <span className="block text-xs font-semibold text-brand-700">
                Deadline · {formatDay(d.date)}
                {!isOpen(d) && " · proposal ready"}
              </span>
              <span className="block truncate text-sm text-slate-700">{d.title}</span>
            </button>
          ))
        ) : (
          <div className="rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-500">
            No deadlines on {formatDay(selected)}.
          </div>
        )}
      </div>
    </div>
  );
}

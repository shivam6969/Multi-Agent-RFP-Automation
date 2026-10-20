import type { Dashboard } from "../api/types";
import { parseDay } from "../lib/format";

/** Round the axis up to a multiple of 4 so the four gridlines get whole numbers. */
function axisMax(values: number[]) {
  const max = Math.max(4, ...values);
  return Math.ceil(max / 4) * 4;
}

/** Monthly bars (proposals generated) with a line for the ones won. Plain HTML + one SVG, no chart library. */
export default function ActivityChart({ activity }: { activity: Dashboard["activity"] }) {
  const top = axisMax(activity.flatMap((m) => [m.generated, m.won]));
  const ticks = [top, (top * 3) / 4, top / 2, top / 4, 0];
  const pct = (v: number) => (v / top) * 100;
  // Centre of each month's column, in % of the plot width.
  const x = (i: number) => ((i + 0.5) / activity.length) * 100;
  const total = activity.reduce((s, m) => s + m.generated, 0);

  return (
    <div className="card p-5">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold text-slate-800">RFP Proposal Activity</h2>
        <div className="flex items-center gap-4 text-xs text-slate-500">
          <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-brand-500" /> Generated</span>
          <span className="flex items-center gap-1.5"><span className="h-0.5 w-3 bg-emerald-500" /> Won</span>
        </div>
      </div>

      <div className="mt-4 flex gap-3">
        <div className="flex h-48 w-6 shrink-0 flex-col justify-between text-right text-[11px] text-slate-400">
          {ticks.map((t) => <span key={t} className="-my-1.5">{t}</span>)}
        </div>
        <div className="relative h-48 flex-1">
          {ticks.map((t) => (
            <div key={t} className="absolute inset-x-0 border-t border-dashed border-slate-100" style={{ bottom: `${pct(t)}%` }} />
          ))}
          <div className="absolute inset-0 flex">
            {activity.map((m) => (
              <div key={m.month} className="flex flex-1 items-end justify-center" title={`${m.generated} generated, ${m.won} won`}>
                <div className="w-1/3 max-w-10 rounded-t-md bg-brand-500" style={{ height: `${pct(m.generated)}%` }} />
              </div>
            ))}
          </div>
          <svg
            className="pointer-events-none absolute inset-0"
            style={{ width: "100%", height: "100%" }}
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
          >
            <polyline
              points={activity.map((m, i) => `${x(i)},${100 - pct(m.won)}`).join(" ")}
              fill="none"
              stroke="#10b981"
              strokeWidth="2"
              vectorEffect="non-scaling-stroke"
            />
          </svg>
          {activity.map((m, i) => (
            <span
              key={m.month}
              className="absolute h-2 w-2 -translate-x-1/2 translate-y-1/2 rounded-full border-2 border-white bg-emerald-500"
              style={{ left: `${x(i)}%`, bottom: `${pct(m.won)}%` }}
            />
          ))}
        </div>
      </div>
      <div className="ml-9 mt-2 flex text-center text-xs text-slate-500">
        {activity.map((m) => (
          <span key={m.month} className="flex-1">
            {parseDay(`${m.month}-01`).toLocaleDateString("en-IN", { month: "short" })}
          </span>
        ))}
      </div>
      {total === 0 && (
        <p className="mt-3 text-center text-xs text-slate-400">
          No proposals generated in the last six months yet. Finished proposals show up here.
        </p>
      )}
    </div>
  );
}

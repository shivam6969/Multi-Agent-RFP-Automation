import { Check } from "lucide-react";
import { Fragment } from "react";

import type { Stage } from "../api/types";
import { cx } from "./ui";

export const STEPS = ["RFP Discovery", "RFP Selection", "Requirements", "Product Matching", "Pricing", "Proposal"];

// Which step a project's stage corresponds to (0-based index into STEPS).
export function stepForStage(stage: Stage | undefined, runningFrom?: string | null): number {
  switch (stage) {
    case "requirements_review":
      return 2;
    case "matching_review":
      return 3;
    case "pricing_review":
      return 4;
    case "proposal_review":
    case "done":
      return 5;
    case "running":
    case "queued":
      return runningFrom?.startsWith("Matching") ? 3 : runningFrom?.startsWith("Calculating") ? 4
        : runningFrom?.startsWith("Drafting") || runningFrom?.startsWith("Rendering") ? 5 : 2;
    default:
      return 2;
  }
}

export default function Stepper({ current }: { current: number }) {
  return (
    <div className="card mb-6 flex items-center gap-2 overflow-x-auto px-4 py-3">
      {STEPS.map((label, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <Fragment key={label}>
            {i > 0 && <div className={cx("h-px min-w-6 flex-1", i <= current ? "bg-brand-400" : "bg-slate-200")} />}
            <div className="flex shrink-0 items-center gap-2">
              <span
                className={cx(
                  "flex h-6 w-6 items-center justify-center rounded-full text-xs font-semibold",
                  done && "bg-brand-500 text-white",
                  active && "bg-brand-500 text-white ring-4 ring-brand-100",
                  !done && !active && "bg-slate-100 text-slate-500",
                )}
              >
                {done ? <Check className="h-3.5 w-3.5" /> : i + 1}
              </span>
              <span className={cx("text-xs", active ? "font-semibold text-slate-800" : "text-slate-500")}>{label}</span>
            </div>
          </Fragment>
        );
      })}
    </div>
  );
}

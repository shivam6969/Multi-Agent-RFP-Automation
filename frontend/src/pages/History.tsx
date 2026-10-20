import { useMemo, useState } from "react";

import { useProjects } from "../api/hooks";
import type { ProjectSummary } from "../api/types";
import ProjectsTable from "../components/ProjectsTable";
import { Button, cx, PageHeader, Spinner } from "../components/ui";
import { formatINRShort } from "../lib/format";

const PAGE = 10;

const FILTERS = [
  { key: "all", label: "All", match: () => true },
  { key: "awaiting", label: "Not submitted", match: (p: ProjectSummary) => p.outcome == null },
  { key: "submitted", label: "Submitted", match: (p: ProjectSummary) => p.outcome === "submitted" },
  { key: "won", label: "Won", match: (p: ProjectSummary) => p.outcome === "won" },
  { key: "lost", label: "Lost", match: (p: ProjectSummary) => p.outcome === "lost" },
] as const;

type FilterKey = (typeof FILTERS)[number]["key"];

const generatedAt = (p: ProjectSummary) => p.completed_at ?? p.updated_at;

export default function History() {
  const { data: projects, isLoading } = useProjects();
  const [filter, setFilter] = useState<FilterKey>("all");
  const [page, setPage] = useState(0);

  // Only finished proposals; newest first by when they were generated.
  const generated = useMemo(
    () => (projects ?? []).filter((p) => p.stage === "done").sort((a, b) => generatedAt(b).localeCompare(generatedAt(a))),
    [projects],
  );
  const counts = Object.fromEntries(FILTERS.map((f) => [f.key, generated.filter(f.match).length])) as Record<FilterKey, number>;
  const shownAll = generated.filter(FILTERS.find((f) => f.key === filter)!.match);
  const pages = Math.max(1, Math.ceil(shownAll.length / PAGE));
  const shown = shownAll.slice(page * PAGE, page * PAGE + PAGE);
  const wonValue = generated.filter((p) => p.outcome === "won").reduce((s, p) => s + (p.grand_total ?? 0), 0);

  return (
    <>
      <PageHeader title="Completed / History" subtitle="Every proposal you have generated, with its outcome. Set the status as you hear back." />
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-1 rounded-lg bg-slate-100 p-1">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              onClick={() => {
                setFilter(f.key);
                setPage(0);
              }}
              className={cx(
                "rounded-md px-3 py-1.5 text-sm transition",
                filter === f.key ? "bg-white font-medium text-slate-800 shadow-sm" : "text-slate-500 hover:text-slate-700",
              )}
            >
              {f.label} <span className="text-xs text-slate-400">{counts[f.key]}</span>
            </button>
          ))}
        </div>
        {counts.won > 0 && (
          <div className="text-sm text-slate-500">
            Won so far: <span className="font-semibold text-emerald-700">{formatINRShort(wonValue)}</span>
          </div>
        )}
      </div>
      <div className="card">
        {isLoading ? (
          <div className="flex justify-center p-10"><Spinner /></div>
        ) : (
          <>
            <ProjectsTable
              projects={shown}
              showOutcome
              dateColumn="generated"
              emptyText={filter === "all" ? "No proposals generated yet." : "No proposals with this status."}
            />
            {shownAll.length > PAGE && (
              <div className="flex items-center justify-between border-t border-slate-100 px-4 py-3 text-sm text-slate-500">
                <span>
                  Showing {page * PAGE + 1}–{Math.min(shownAll.length, (page + 1) * PAGE)} of {shownAll.length}
                </span>
                <div className="flex gap-2">
                  <Button size="sm" variant="secondary" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</Button>
                  <Button size="sm" variant="secondary" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>Next</Button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}

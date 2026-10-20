import { Download, FolderOpen } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";

import { useSetOutcome } from "../api/hooks";
import type { ProjectSummary, Stage } from "../api/types";
import { formatDate, formatINRShort, projectRef } from "../lib/format";
import { Badge, EmptyState } from "./ui";

export const STAGE_LABEL: Record<Stage, string> = {
  queued: "Queued",
  running: "Working…",
  requirements_review: "Review requirements",
  matching_review: "Review matches",
  pricing_review: "Review pricing",
  failed: "Failed",
  done: "Proposal ready",
};

export function StatusBadge({ project }: { project: ProjectSummary }) {
  if (project.outcome === "won") return <Badge tone="green">Won</Badge>;
  if (project.outcome === "lost") return <Badge tone="red">Lost</Badge>;
  if (project.outcome === "submitted") return <Badge tone="blue">Submitted</Badge>;
  const tone = project.running
    ? "violet"
    : project.stage === "failed"
      ? "red"
      : project.stage === "done"
        ? "green"
        : "amber";
  return <Badge tone={tone}>{project.running ? project.progress ?? "Working…" : STAGE_LABEL[project.stage]}</Badge>;
}

export default function ProjectsTable({
  projects,
  showOutcome = false,
  emptyText = "Nothing here yet.",
}: {
  projects: ProjectSummary[];
  showOutcome?: boolean;
  emptyText?: string;
}) {
  const navigate = useNavigate();
  const setOutcome = useSetOutcome();

  if (!projects.length) {
    return (
      <EmptyState icon={<FolderOpen className="h-5 w-5" />} title={emptyText}>
        <Link to="/new" className="font-medium text-brand-600 hover:underline">
          Start a new session
        </Link>
      </EmptyState>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-100 text-left text-[11px] uppercase tracking-wide text-slate-500">
            <th className="px-4 py-3 font-medium">Reference</th>
            <th className="px-4 py-3 font-medium">Proposal</th>
            <th className="px-4 py-3 font-medium">Organization</th>
            <th className="px-4 py-3 font-medium">Updated</th>
            <th className="px-4 py-3 font-medium">Value</th>
            <th className="px-4 py-3 font-medium">Status</th>
            <th className="px-4 py-3" />
          </tr>
        </thead>
        <tbody>
          {projects.map((p) => (
            <tr
              key={p.id}
              onClick={() => navigate(`/projects/${p.id}`)}
              className="cursor-pointer border-b border-slate-100 last:border-0 hover:bg-slate-50"
            >
              <td className="px-4 py-3 font-mono text-xs text-slate-500">{projectRef(p.id)}</td>
              <td className="max-w-xs truncate px-4 py-3 font-medium text-slate-800">{p.rfp.title}</td>
              <td className="px-4 py-3 text-slate-600">{p.rfp.buyer ?? "—"}</td>
              <td className="px-4 py-3 text-slate-600">{formatDate(p.updated_at)}</td>
              <td className="px-4 py-3 font-semibold text-slate-800">{formatINRShort(p.grand_total)}</td>
              <td className="px-4 py-3" onClick={(e) => showOutcome && p.stage === "done" && e.stopPropagation()}>
                {showOutcome && p.stage === "done" ? (
                  <select
                    className="rounded-md border border-slate-200 bg-white px-2 py-1 text-xs"
                    value={p.outcome ?? ""}
                    onChange={(e) => setOutcome.mutate({ id: p.id, outcome: e.target.value || null })}
                  >
                    <option value="">Proposal ready</option>
                    <option value="submitted">Submitted</option>
                    <option value="won">Won</option>
                    <option value="lost">Lost</option>
                  </select>
                ) : (
                  <StatusBadge project={p} />
                )}
              </td>
              <td className="px-4 py-3 text-right">
                {p.proposal_available && (
                  <a
                    href={`/api/projects/${p.id}/proposal.pdf`}
                    download
                    onClick={(e) => e.stopPropagation()}
                    className="inline-flex items-center gap-1 text-xs font-medium text-slate-600 hover:text-brand-600"
                  >
                    <Download className="h-3.5 w-3.5" /> PDF
                  </a>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

import { ArrowRight, Calendar, Eye, FileText, MapPin, Package, Search, Upload } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";

import { useProjects, useRfps, useStartProject } from "../api/hooks";
import type { ProjectSummary, Rfp } from "../api/types";
import { StatusBadge } from "../components/ProjectsTable";
import Stepper from "../components/Stepper";
import { Badge, Button, cx, EmptyState, PageHeader, Spinner } from "../components/ui";
import { formatDate, formatINRShort } from "../lib/format";

function RfpCard({
  rfp,
  project,
  selected,
  onSelect,
}: {
  rfp: Rfp;
  project?: ProjectSummary;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <div
      onClick={onSelect}
      className={cx(
        "card flex cursor-pointer flex-col p-4 transition",
        selected ? "border-brand-400 bg-brand-50/40 ring-2 ring-brand-100" : "hover:border-brand-200",
      )}
    >
      <div className="mb-2 flex items-center justify-between">
        <input type="radio" readOnly checked={selected} className="h-4 w-4 accent-brand-500" />
        {project ? <StatusBadge project={project} /> : <Badge tone="blue">New</Badge>}
      </div>
      <h3 className="line-clamp-2 font-semibold text-slate-800">{rfp.title}</h3>
      <p className="mt-1 text-sm text-slate-500">{rfp.buyer ?? rfp.filename}</p>
      <ul className="mt-3 space-y-1.5 text-xs text-slate-500">
        <li className="flex items-center gap-2"><Calendar className="h-3.5 w-3.5" /> Added {formatDate(rfp.created_at)}</li>
        {rfp.deadline && <li className="flex items-center gap-2"><Calendar className="h-3.5 w-3.5" /> Deadline <b className="text-slate-700">{formatDate(rfp.deadline)}</b></li>}
        {rfp.location && <li className="flex items-center gap-2"><MapPin className="h-3.5 w-3.5" /> {rfp.location}</li>}
        <li className="flex items-center gap-2">
          <Package className="h-3.5 w-3.5" /> {rfp.pages} page{rfp.pages === 1 ? "" : "s"}
          {rfp.estimated_value != null && <> · <b className="text-slate-700">{formatINRShort(rfp.estimated_value)}</b> est. value</>}
        </li>
      </ul>
      <div className="mt-4 flex items-center justify-between border-t border-slate-100 pt-3">
        <span className="text-xs capitalize text-slate-500">{rfp.category ?? rfp.source}</span>
        <a
          href={`/api/rfps/${rfp.id}/document`}
          target="_blank"
          rel="noreferrer"
          onClick={(e) => e.stopPropagation()}
          className="inline-flex items-center gap-1 rounded-md border border-slate-200 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
        >
          <Eye className="h-3.5 w-3.5" /> View PDF
        </a>
      </div>
    </div>
  );
}

export default function RfpSelection() {
  const [params] = useSearchParams();
  const [q, setQ] = useState("");
  const [selected, setSelected] = useState<string | null>(params.get("selected"));
  const { data: rfps, isLoading } = useRfps(q);
  const { data: projects } = useProjects();
  const start = useStartProject();
  const navigate = useNavigate();

  // Latest project per RFP (projects are newest first).
  const projectByRfp = useMemo(() => {
    const map = new Map<string, ProjectSummary>();
    for (const p of projects ?? []) if (!map.has(p.rfp.id)) map.set(p.rfp.id, p);
    return map;
  }, [projects]);

  const existing = selected ? projectByRfp.get(selected) : undefined;

  return (
    <>
      <Stepper current={1} />
      <PageHeader
        title="RFPs / Documents"
        subtitle="Select the RFP you want to respond to. One RFP is processed at a time."
        actions={<Button variant="secondary" icon={<Upload className="h-4 w-4" />} onClick={() => navigate("/new")}>Upload RFP</Button>}
      />

      <div className="relative mb-5 max-w-md">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
        <input className="input pl-9" placeholder="Search RFPs, organizations, locations" value={q} onChange={(e) => setQ(e.target.value)} />
      </div>

      {isLoading ? (
        <div className="flex justify-center p-10"><Spinner /></div>
      ) : !rfps?.length ? (
        <div className="card">
          <EmptyState icon={<FileText className="h-5 w-5" />} title={q ? "No RFPs match your search" : "No RFPs yet"}>
            <Link to="/new" className="font-medium text-brand-600 hover:underline">Upload your first RFP</Link>
          </EmptyState>
        </div>
      ) : (
        <div className="grid gap-4 pb-24 sm:grid-cols-2 lg:grid-cols-3">
          {rfps.map((rfp) => (
            <RfpCard
              key={rfp.id}
              rfp={rfp}
              project={projectByRfp.get(rfp.id)}
              selected={selected === rfp.id}
              onSelect={() => setSelected(rfp.id)}
            />
          ))}
        </div>
      )}

      {selected && (
        <div className="fixed bottom-6 left-1/2 z-10 flex -translate-x-1/2 items-center gap-4 rounded-xl border border-slate-200 bg-white px-4 py-3 shadow-lg">
          {start.error && <span className="max-w-xs text-sm text-rose-600">{(start.error as Error).message}</span>}
          {existing && (
            <Button variant="secondary" onClick={() => navigate(`/projects/${existing.id}`)}>Open existing project</Button>
          )}
          <Button
            loading={start.isPending}
            icon={<ArrowRight className="h-4 w-4" />}
            onClick={() => start.mutate(selected, { onSuccess: (p) => navigate(`/projects/${p.id}`) })}
          >
            {existing ? "Start again" : "Continue with selected RFP"}
          </Button>
        </div>
      )}
    </>
  );
}

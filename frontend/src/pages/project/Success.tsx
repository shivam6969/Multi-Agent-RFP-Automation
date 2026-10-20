import { CheckCircle2, Download, Eye, FileSpreadsheet, FileText, Plus } from "lucide-react";
import { useNavigate } from "react-router-dom";

import { useSetOutcome } from "../../api/hooks";
import type { Project } from "../../api/types";
import { Button, ErrorBox, LinkButton } from "../../components/ui";
import { formatDate, formatINR, formatNumber, projectRef } from "../../lib/format";

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-slate-500">{label}</div>
      <div className="text-sm font-medium text-slate-800">{value}</div>
    </div>
  );
}

export default function Success({ project }: { project: Project }) {
  const navigate = useNavigate();
  const setOutcome = useSetOutcome();
  const summary = project.pricing_summary;
  const units = project.line_items.reduce((s, li) => s + li.qty, 0);
  const pdf = `/api/projects/${project.id}/proposal.pdf`;

  return (
    <div className="mx-auto max-w-4xl">
      <div className="mb-8 text-center">
        <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-emerald-50 text-emerald-600 ring-8 ring-emerald-50/50">
          <CheckCircle2 className="h-7 w-7" />
        </div>
        <h1 className="text-2xl font-semibold text-slate-800">
          {project.proposal_available ? "Proposal Generated Successfully" : "Quote Ready"}
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          {project.proposal_available
            ? "Your proposal is ready to review and submit."
            : "Pricing is approved. The proposal PDF could not be generated (see below); the quote CSV is available."}
        </p>
        <div className="mt-5 flex flex-wrap justify-center gap-3">
          {project.proposal_available && (
            <>
              <LinkButton href={pdf} download icon={<Download className="h-4 w-4" />}>Download PDF</LinkButton>
              <LinkButton href={pdf} target="_blank" rel="noreferrer" variant="secondary" icon={<Eye className="h-4 w-4" />}>
                View Proposal
              </LinkButton>
            </>
          )}
          <LinkButton href={`/api/projects/${project.id}/quote.csv`} download variant="secondary" icon={<FileSpreadsheet className="h-4 w-4" />}>
            Quote (CSV)
          </LinkButton>
          <Button variant="ghost" icon={<Plus className="h-4 w-4" />} onClick={() => navigate("/new")}>Start New Session</Button>
        </div>
      </div>

      {project.error && !project.proposal_available && (
        <div className="mb-6"><ErrorBox title="Proposal PDF not generated">{project.error}</ErrorBox></div>
      )}

      <div className="grid gap-5 md:grid-cols-[1fr_18rem]">
        <div className="card p-5">
          <h2 className="mb-4 font-semibold text-slate-800">Proposal Summary</h2>
          <div className="grid grid-cols-2 gap-4">
            <Field label="RFP" value={project.rfp.title} />
            <Field label="Reference" value={projectRef(project.id)} />
            <Field label="Organization" value={project.rfp.buyer ?? "—"} />
            <Field label="Generated" value={formatDate(project.updated_at)} />
            <Field label="Products" value={`${project.line_items.length} line items · ${formatNumber(units)} units`} />
            <Field label="Margin applied" value={`${summary.margin_pct ?? "—"}% + ${project.line_items[0]?.gst_pct ?? 18}% GST`} />
          </div>
          <div className="mt-5 flex items-center justify-between border-t border-slate-100 pt-4">
            <div className="flex items-center gap-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-50 text-brand-500"><FileText className="h-4 w-4" /></span>
              <span className="text-sm text-slate-700">Outcome</span>
            </div>
            <select
              className="rounded-md border border-slate-200 bg-white px-2 py-1.5 text-sm"
              value={project.outcome ?? ""}
              onChange={(e) => setOutcome.mutate({ id: project.id, outcome: e.target.value || null })}
            >
              <option value="">Not submitted yet</option>
              <option value="submitted">Submitted</option>
              <option value="won">Won</option>
              <option value="lost">Lost</option>
            </select>
          </div>
        </div>
        <div className="flex flex-col rounded-xl bg-cocoa-900 p-5 text-white">
          <div className="text-sm text-white/70">Proposal value</div>
          <div className="mt-auto pt-10 text-3xl font-semibold">{formatINR(summary.grand_total)}</div>
          <div className="text-xs text-white/60">Inclusive of GST and margin</div>
        </div>
      </div>
    </div>
  );
}

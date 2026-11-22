import { ArrowRight, ChevronDown, ChevronUp, Plus, Trash2 } from "lucide-react";
import { useState } from "react";

import { useSubmitReview } from "../../api/hooks";
import type { Criticality, Project, RequirementSource } from "../../api/types";
import { Badge, Button, cx, ErrorBox, Notice, PageHeader } from "../../components/ui";
import { formatNumber, sourceLabel } from "../../lib/format";

interface Row {
  key: number;
  id?: string;
  requirement: string;
  criticality: Criticality;
  quantity: string;
  // Read-only context from the RFP (kept by the backend for existing ids).
  unit?: string | null;
  source?: RequirementSource;
  breakdown?: Record<string, number>;
}

const CRITICALITY: Criticality[] = ["mandatory", "high", "medium", "low"];
const TONE = { mandatory: "red", high: "amber", medium: "blue", low: "slate" } as const;

// The RFP agent answers in light Markdown; show it as plain text.
const plain = (s: string) => s.replace(/\*\*(.+?)\*\*/g, "$1").replace(/^#+\s*/gm, "");

export default function RequirementsReview({ project }: { project: Project }) {
  const submit = useSubmitReview(project.id);
  const [rows, setRows] = useState<Row[]>(() =>
    project.requirements.map((r, i) => ({
      key: i,
      id: r.id,
      requirement: r.requirement,
      criticality: r.criticality,
      quantity: r.quantity ? String(r.quantity) : "",
      unit: r.unit,
      source: r.source,
      breakdown: r.quantity_breakdown,
    })),
  );
  const fromTables = project.requirements.filter((r) => r.origin === "table").length;
  const [nextKey, setNextKey] = useState(project.requirements.length);
  const [showSummary, setShowSummary] = useState(false);

  const update = (key: number, patch: Partial<Row>) =>
    setRows((rs) => rs.map((r) => (r.key === key ? { ...r, ...patch } : r)));

  const invalid = rows.some((r) => r.requirement.trim().length < 3 || (r.quantity && !(Number(r.quantity) > 0)));

  function approve() {
    submit.mutate({
      requirements: rows.map((r) => ({
        id: r.id,
        requirement: r.requirement.trim(),
        criticality: r.criticality,
        quantity: r.quantity ? Number(r.quantity) : null,
      })),
    });
  }

  return (
    <>
      <PageHeader
        title="Review Requirements"
        subtitle={<>The AI read <b>{project.rfp.title}</b>. Edit, remove or add requirements, and set quantities where you know them.</>}
      />

      {project.warnings.map((w) => (
        <div key={w} className="mb-4"><Notice>{w}</Notice></div>
      ))}

      {project.rfp_summary && (
        <div className="card mb-6">
          <button onClick={() => setShowSummary((s) => !s)} className="flex w-full items-center justify-between px-5 py-3 text-left">
            <span className="font-semibold text-slate-800">RFP overview</span>
            {showSummary ? <ChevronUp className="h-4 w-4 text-slate-400" /> : <ChevronDown className="h-4 w-4 text-slate-400" />}
          </button>
          <div
            className={cx(
              "whitespace-pre-wrap border-t border-slate-100 px-5 py-4 text-sm leading-relaxed text-slate-600",
              !showSummary && "line-clamp-4",
            )}
          >
            {plain(project.rfp_summary).replace(/^Here is a summary of the RFP:\s*/i, "")}
          </div>
        </div>
      )}

      <div className="card">
        <div className="flex items-center justify-between border-b border-slate-100 px-5 py-3">
          <h2 className="font-semibold text-slate-800">Requirements</h2>
          <div className="flex items-center gap-2">
            {fromTables > 0 && (
              <span title="Read directly from the RFP's item tables: quantities come from the table cells">
                <Badge tone="blue">{fromTables} from tables</Badge>
              </span>
            )}
            <Badge>{rows.length} items</Badge>
          </div>
        </div>
        <div className="divide-y divide-slate-100">
          {rows.map((r, i) => (
            <div key={r.key} className="grid grid-cols-[2.5rem_1fr_8.5rem_6.5rem_2rem] items-start gap-3 px-5 py-2.5">
              <span className="pt-2 font-mono text-xs text-slate-400">{r.id ?? `new ${i + 1}`}</span>
              <div className="min-w-0">
                <input
                  className={cx("input w-full py-1.5", r.requirement.trim().length < 3 && "border-rose-300")}
                  value={r.requirement}
                  onChange={(e) => update(r.key, { requirement: e.target.value })}
                  placeholder="Describe the requirement"
                />
                {(r.source || r.unit || r.breakdown) && (
                  <div className="mt-1 flex flex-wrap gap-x-3 text-[11px] text-slate-400">
                    {sourceLabel(r.source) && <span title={r.source?.section || undefined}>{sourceLabel(r.source)}</span>}
                    {r.unit && <span>Unit: {r.unit}</span>}
                    {r.breakdown && (
                      <span>
                        {Object.entries(r.breakdown).map(([k, v]) => `${k} ${formatNumber(v)}`).join(" + ")}
                      </span>
                    )}
                  </div>
                )}
              </div>
              <div className="flex items-center gap-2">
                <span className={cx("h-2 w-2 shrink-0 rounded-full", {
                  mandatory: "bg-rose-500", high: "bg-amber-500", medium: "bg-sky-500", low: "bg-slate-400",
                }[r.criticality])} />
                <select
                  className="input py-1.5 capitalize"
                  value={r.criticality}
                  onChange={(e) => update(r.key, { criticality: e.target.value as Criticality })}
                >
                  {CRITICALITY.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <input
                className={cx("input py-1.5", r.quantity && !(Number(r.quantity) > 0) && "border-rose-300")}
                inputMode="numeric"
                placeholder="Qty"
                value={r.quantity}
                onChange={(e) => update(r.key, { quantity: e.target.value.replace(/[^\d]/g, "") })}
                title="Order quantity, if the RFP states one"
              />
              <button
                onClick={() => setRows((rs) => rs.filter((x) => x.key !== r.key))}
                className="rounded p-1 text-slate-400 hover:bg-rose-50 hover:text-rose-600"
                title="Remove"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
          ))}
        </div>
        <div className="border-t border-slate-100 px-5 py-3">
          <Button
            variant="ghost"
            size="sm"
            icon={<Plus className="h-4 w-4" />}
            onClick={() => {
              setRows((rs) => [...rs, { key: nextKey, requirement: "", criticality: "high", quantity: "" }]);
              setNextKey((k) => k + 1);
            }}
          >
            Add requirement
          </Button>
        </div>
      </div>

      <div className="mt-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex gap-3 text-xs text-slate-500">
          {CRITICALITY.map((c) => (
            <span key={c}><Badge tone={TONE[c]}>{c}</Badge> {rows.filter((r) => r.criticality === c).length}</span>
          ))}
        </div>
        <Button icon={<ArrowRight className="h-4 w-4" />} loading={submit.isPending} disabled={invalid || !rows.length} onClick={approve}>
          Continue to Product Matching
        </Button>
      </div>
      {submit.error && <div className="mt-3"><ErrorBox>{(submit.error as Error).message}</ErrorBox></div>}
    </>
  );
}

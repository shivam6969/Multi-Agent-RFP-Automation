import { AlertTriangle, ChevronDown, ExternalLink, RefreshCw, Save, Sparkles } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  proposalPreviewUrl,
  useRegenerateSection,
  useSaveProposalDraft,
  useSubmitReview,
  type SectionEdit,
} from "../../api/hooks";
import type { Project, ProposalSection } from "../../api/types";
import { Badge, Button, cx, ErrorBox, Spinner } from "../../components/ui";

// Same pattern as the server's check: money amounts belong in the pricing table.
const AMOUNT_RE = /(₹|\bRs\.?|\bINR)\s*\d[\d,]*(\.\d+)?|\b\d+(\.\d+)?\s*(lakh|lac|crore|cr)\b/i;

function SectionCard({
  section,
  open,
  onToggle,
  onChange,
  onRegenerate,
  regenerating,
}: {
  section: ProposalSection;
  open: boolean;
  onToggle: () => void;
  onChange: (body: string) => void;
  onRegenerate: (instruction: string) => void;
  regenerating: boolean;
}) {
  const [instruction, setInstruction] = useState("");
  const mentionsAmount = AMOUNT_RE.test(section.body);

  return (
    <div className="border-b border-slate-100 last:border-0">
      <button onClick={onToggle} className="flex w-full items-center justify-between gap-2 px-4 py-3 text-left">
        <span className="flex min-w-0 items-center gap-2">
          <span className="truncate text-sm font-medium text-slate-800">{section.title}</span>
          {section.ai_generated && <Badge tone="violet"><Sparkles className="h-3 w-3" />AI Generated</Badge>}
          {section.edited && <Badge tone="blue">Edited</Badge>}
          {mentionsAmount && <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-amber-500" />}
        </span>
        <ChevronDown className={cx("h-4 w-4 shrink-0 text-slate-400 transition", open && "rotate-180")} />
      </button>
      {open && (
        <div className="space-y-2 px-4 pb-4">
          <textarea
            className="input min-h-40 resize-y font-[inherit] leading-relaxed"
            value={section.body}
            onChange={(e) => onChange(e.target.value)}
            rows={Math.min(18, Math.max(6, section.body.split("\n").length + 2))}
          />
          {mentionsAmount && (
            <p className="text-xs text-amber-700">
              This section mentions a money amount. Prices come from the pricing table; remove amounts from the text or check
              they match.
            </p>
          )}
          {section.key === "pricing" && (
            <p className="text-xs text-slate-500">The pricing table and totals are added below this text automatically.</p>
          )}
          {section.ai_generated && (
            <div className="flex items-center gap-2">
              <input
                className="input flex-1 py-1.5 text-xs"
                placeholder="Optional instruction, e.g. make it shorter"
                value={instruction}
                onChange={(e) => setInstruction(e.target.value)}
              />
              <Button
                size="sm"
                variant="secondary"
                loading={regenerating}
                icon={<RefreshCw className="h-3.5 w-3.5" />}
                onClick={() => onRegenerate(instruction)}
              >
                Regenerate
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function ProposalEditor({ project }: { project: Project }) {
  const submit = useSubmitReview(project.id);
  const saveDraft = useSaveProposalDraft(project.id);
  const regenerate = useRegenerateSection(project.id);

  const [sections, setSections] = useState<ProposalSection[]>(project.proposal_sections);
  const [open, setOpen] = useState<string | null>("executive_summary");
  const [regeneratingKey, setRegeneratingKey] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const lastUrl = useRef<string | null>(null);

  const edits: SectionEdit[] = useMemo(() => sections.map(({ key, title, body }) => ({ key, title, body })), [sections]);
  const dirty = sections.some((s, i) => s.body !== project.proposal_sections[i]?.body);

  // Re-render the PDF preview shortly after typing stops.
  const editsKey = JSON.stringify(edits);
  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(() => {
      setPreviewing(true);
      proposalPreviewUrl(project.id, JSON.parse(editsKey), controller.signal)
        .then((url) => {
          if (lastUrl.current) URL.revokeObjectURL(lastUrl.current);
          lastUrl.current = url;
          setPreviewUrl(url);
          setPreviewError(null);
        })
        .catch((e: Error) => e.name !== "AbortError" && setPreviewError(e.message))
        .finally(() => setPreviewing(false));
    }, 800);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [editsKey, project.id]);
  useEffect(() => () => void (lastUrl.current && URL.revokeObjectURL(lastUrl.current)), []);

  const setBody = (key: string, body: string) =>
    setSections((all) => all.map((s) => (s.key === key ? { ...s, body, edited: true } : s)));

  return (
    <>
      <h1 className="page-title">Generate Proposal</h1>
      <p className="page-subtitle mb-6">Edit sections on the left; the PDF preview updates as you type.</p>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <div className="space-y-4">
          <div className="card overflow-hidden">
            <div className="border-b border-slate-100 px-4 py-3 font-semibold text-slate-800">Proposal Editor</div>
            {sections.map((s) => (
              <SectionCard
                key={s.key}
                section={s}
                open={open === s.key}
                onToggle={() => setOpen(open === s.key ? null : s.key)}
                onChange={(body) => setBody(s.key, body)}
                regenerating={regeneratingKey === s.key}
                onRegenerate={(instruction) => {
                  setRegeneratingKey(s.key);
                  regenerate.mutate(
                    { key: s.key, instruction, sections: edits },
                    { onSuccess: (r) => setBody(s.key, r.body), onSettled: () => setRegeneratingKey(null) },
                  );
                }}
              />
            ))}
          </div>
          {regenerate.error && <ErrorBox title="Could not regenerate">{(regenerate.error as Error).message}</ErrorBox>}
          {(submit.error || saveDraft.error) && <ErrorBox>{((submit.error || saveDraft.error) as Error).message}</ErrorBox>}
          <div className="flex flex-wrap gap-3">
            <Button className="flex-1" loading={submit.isPending} onClick={() => submit.mutate({ sections: edits })}>
              Complete Generation
            </Button>
            <Button
              variant="secondary"
              icon={<Save className="h-4 w-4" />}
              loading={saveDraft.isPending}
              disabled={!dirty}
              onClick={() => saveDraft.mutate(edits)}
            >
              {saveDraft.isSuccess && !dirty ? "Saved" : "Save draft"}
            </Button>
          </div>
        </div>

        <div className="card flex min-h-[70vh] flex-col overflow-hidden">
          <div className="flex items-center justify-between border-b border-slate-100 px-4 py-2.5 text-sm">
            <span className="text-slate-600">
              Proposal.pdf <span className="text-slate-400">· {project.proposal_meta.reference}</span>
            </span>
            <span className="flex items-center gap-3">
              {previewing && <Spinner className="h-4 w-4" />}
              {previewUrl && (
                <a href={previewUrl} target="_blank" rel="noreferrer" className="text-slate-500 hover:text-brand-600" title="Open in a new tab">
                  <ExternalLink className="h-4 w-4" />
                </a>
              )}
            </span>
          </div>
          <div className="flex-1 bg-slate-100">
            {previewError ? (
              <div className="p-4"><ErrorBox title="Preview failed">{previewError}</ErrorBox></div>
            ) : previewUrl ? (
              <iframe title="Proposal preview" src={previewUrl} className="h-full min-h-[70vh] w-full" />
            ) : (
              <div className="flex h-full items-center justify-center"><Spinner /></div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

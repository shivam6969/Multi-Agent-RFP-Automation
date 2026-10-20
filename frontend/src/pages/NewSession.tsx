import { FileUp, Sparkles, UploadCloud } from "lucide-react";
import { useRef, useState, type DragEvent } from "react";
import { useNavigate } from "react-router-dom";

import { useUploadRfp } from "../api/hooks";
import Stepper from "../components/Stepper";
import { Badge, Button, cx, ErrorBox } from "../components/ui";

export default function NewSession() {
  const upload = useUploadRfp();
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function send(file: File | undefined) {
    if (!file) return;
    upload.mutate(file, { onSuccess: (rfp) => navigate(`/rfps?selected=${rfp.id}`) });
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    send(e.dataTransfer.files[0]);
  }

  return (
    <>
      <Stepper current={0} />
      <Badge tone="violet" className="mb-3">
        <Sparkles className="h-3 w-3" /> AI-powered RFP response
      </Badge>
      <h1 className="page-title">Start New Session</h1>
      <p className="page-subtitle mb-6">Add an RFP document to analyse. The AI will extract its requirements for you to review.</p>

      <div className="card p-6">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => input.current?.click()}
          className={cx(
            "flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-14 text-center transition",
            dragging ? "border-brand-400 bg-brand-50" : "border-slate-200 hover:border-brand-300 hover:bg-slate-50",
          )}
        >
          <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-brand-50 text-brand-500">
            <UploadCloud className="h-6 w-6" />
          </div>
          <div className="font-medium text-slate-700">Drop an RFP PDF here, or click to browse</div>
          <div className="mt-1 text-sm text-slate-500">PDF with selectable text, up to 25 MB</div>
          <input
            ref={input}
            type="file"
            accept="application/pdf,.pdf"
            className="hidden"
            onChange={(e) => send(e.target.files?.[0])}
          />
        </div>

        {upload.isPending && <p className="mt-4 text-center text-sm text-slate-500">Uploading and checking the PDF…</p>}
        {upload.error && (
          <div className="mt-4">
            <ErrorBox title="Upload failed">{(upload.error as Error).message}</ErrorBox>
          </div>
        )}

        <div className="mt-6 flex items-center justify-between border-t border-slate-100 pt-5">
          <p className="text-sm text-slate-500">
            Already uploaded an RFP? Pick it from your documents.
          </p>
          <Button variant="secondary" icon={<FileUp className="h-4 w-4" />} onClick={() => navigate("/rfps")}>
            Choose an existing RFP
          </Button>
        </div>
      </div>

      <p className="mt-4 text-xs text-slate-400">
        Tender discovery by keyword (with generated sample RFPs) arrives with the Discovery Agent.
      </p>
    </>
  );
}

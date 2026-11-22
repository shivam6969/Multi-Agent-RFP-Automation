import { Calendar, ChevronDown, FileUp, MapPin, Search, Sparkles, UploadCloud, Warehouse, X } from "lucide-react";
import { useRef, useState, type DragEvent, type FormEvent, type KeyboardEvent } from "react";
import { useNavigate } from "react-router-dom";

import { fetchStockKeywords, useDiscoverySearch, useDiscoverySources, useUploadRfp } from "../api/hooks";
import Stepper from "../components/Stepper";
import { Badge, Button, cx, ErrorBox } from "../components/ui";

const iso = (d: Date) => d.toISOString().slice(0, 10);

function KeywordInput({ value, onChange }: { value: string[]; onChange: (v: string[]) => void }) {
  const [draft, setDraft] = useState("");
  const add = (text: string) => {
    const parts = text.split(",").map((s) => s.trim()).filter(Boolean);
    const next = [...value];
    for (const p of parts) if (!next.some((k) => k.toLowerCase() === p.toLowerCase())) next.push(p);
    onChange(next.slice(0, 10));
    setDraft("");
  };
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      add(draft);
    } else if (e.key === "Backspace" && !draft && value.length) {
      onChange(value.slice(0, -1));
    }
  };
  return (
    <div className="input flex min-h-24 flex-wrap content-start items-start gap-2 py-2.5">
      {value.map((k) => (
        <span key={k} className="inline-flex items-center gap-1 rounded-full bg-brand-50 px-2.5 py-1 text-xs font-medium text-brand-700">
          {k}
          <button type="button" onClick={() => onChange(value.filter((x) => x !== k))} className="text-brand-400 hover:text-brand-700">
            <X className="h-3 w-3" />
          </button>
        </span>
      ))}
      <input
        className="min-w-48 flex-1 bg-transparent py-1 text-sm outline-none placeholder:text-slate-400"
        placeholder={value.length ? "Add another keyword" : "Enter keywords such as modular switches, sockets, fan regulators…"}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={onKey}
        onBlur={() => draft.trim() && add(draft)}
      />
    </div>
  );
}

function FindRfps() {
  const navigate = useNavigate();
  const { data: sources } = useDiscoverySources();
  const search = useDiscoverySearch();
  const today = new Date();

  const [source, setSource] = useState("sample");
  const [keywords, setKeywords] = useState<string[]>(["modular switches", "sockets"]);
  const [count, setCount] = useState(6);
  const [location, setLocation] = useState("");
  const [dateFrom, setDateFrom] = useState(iso(new Date(today.getTime() - 14 * 864e5)));
  const [dateTo, setDateTo] = useState(iso(today));
  const [advanced, setAdvanced] = useState(false);
  const [category, setCategory] = useState("");
  const [minValue, setMinValue] = useState("");
  const [maxValue, setMaxValue] = useState("");
  const [stockBusy, setStockBusy] = useState(false);
  const [stockError, setStockError] = useState<string | null>(null);

  async function loadStockKeywords() {
    setStockBusy(true);
    setStockError(null);
    try {
      const found = await fetchStockKeywords();
      if (found.length) setKeywords(found.map((k) => k.keyword));
      else setStockError("No product types found in your stock.");
    } catch (e) {
      setStockError((e as Error).message);
    } finally {
      setStockBusy(false);
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    search.mutate(
      {
        source,
        keywords,
        count,
        location: location.trim() || undefined,
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined,
        category: category.trim() || undefined,
        min_value: minValue ? Number(minValue) : undefined,
        max_value: maxValue ? Number(maxValue) : undefined,
      },
      { onSuccess: (rfps) => navigate(`/rfps?ids=${rfps.map((r) => r.id).join(",")}`) },
    );
  }

  return (
    <form onSubmit={submit} className="card overflow-hidden">
      <div className="space-y-5 p-6">
        <div>
          <span className="label">Source</span>
          <div className="flex flex-wrap gap-2">
            {(sources ?? [{ name: "sample", label: "Sample tenders (demo)", enabled: true, reason: null }]).map((s) => (
              <button
                key={s.name}
                type="button"
                disabled={!s.enabled}
                title={s.reason ?? undefined}
                onClick={() => setSource(s.name)}
                className={cx(
                  "rounded-lg border px-3 py-2 text-sm transition",
                  source === s.name ? "border-brand-400 bg-brand-50 font-medium text-brand-700" : "border-slate-200 text-slate-600",
                  !s.enabled && "cursor-not-allowed opacity-50",
                )}
              >
                {s.label}
                {!s.enabled && <span className="ml-1.5 text-xs">(off)</span>}
              </button>
            ))}
          </div>
          {source === "sample" && (
            <p className="mt-1.5 text-xs text-slate-500">
              Generates realistic sample tenders from your catalogue, with fictitious buyers. Every PDF is marked as a sample.
            </p>
          )}
          {source === "gem" && (
            <p className="mt-1.5 text-xs text-slate-500">
              Searches live bids on bidplus.gem.gov.in in a browser and downloads each bid document. It goes slowly on
              purpose, so allow about 15 seconds per RFP. Only bids whose item category names the product are kept.
            </p>
          )}
        </div>

        <div>
          <div className="flex items-end justify-between gap-3">
            <span className="label">Keywords</span>
            <button
              type="button"
              onClick={loadStockKeywords}
              disabled={stockBusy}
              className="mb-1.5 inline-flex items-center gap-1.5 text-xs font-medium text-brand-600 hover:underline disabled:text-slate-400"
            >
              <Warehouse className="h-3.5 w-3.5" />
              {stockBusy ? "Reading your stock…" : "Use keywords from my stock"}
            </button>
          </div>
          <KeywordInput value={keywords} onChange={setKeywords} />
          <p className="mt-1 text-xs text-slate-400">
            Separate keywords with commas. Press Enter to add.
            {stockError && <span className="ml-1 text-rose-600">{stockError}</span>}
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label className="label" htmlFor="count">Number of RFPs</label>
            <input id="count" type="number" min={1} max={20} className="input" value={count}
              onChange={(e) => setCount(Math.max(1, Math.min(20, Number(e.target.value) || 1)))} />
          </div>
          <div>
            <label className="label" htmlFor="location">Location</label>
            <div className="relative">
              <MapPin className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <input id="location" className="input pl-9" placeholder="Any (e.g. Gurugram)" value={location} onChange={(e) => setLocation(e.target.value)} />
            </div>
          </div>
          <div>
            <label className="label" htmlFor="from">Published from</label>
            <div className="relative">
              <Calendar className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <input id="from" type="date" className="input pl-9" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} />
            </div>
          </div>
          <div>
            <label className="label" htmlFor="to">To</label>
            <div className="relative">
              <Calendar className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <input id="to" type="date" className="input pl-9" value={dateTo} onChange={(e) => setDateTo(e.target.value)} />
            </div>
          </div>
        </div>

        <div className="border-t border-slate-100 pt-4">
          <button type="button" onClick={() => setAdvanced((a) => !a)} className="flex items-center gap-1 text-sm font-medium text-slate-700">
            Advanced filters <ChevronDown className={cx("h-4 w-4 transition", advanced && "rotate-180")} />
          </button>
          {advanced && (
            <div className="mt-4 grid gap-4 sm:grid-cols-3">
              <div>
                <label className="label" htmlFor="category">Category</label>
                <input id="category" className="input" placeholder="e.g. Wiring accessories" value={category} onChange={(e) => setCategory(e.target.value)} />
              </div>
              <div>
                <label className="label" htmlFor="min">Min. budget (₹)</label>
                <input id="min" inputMode="numeric" className="input" placeholder="e.g. 500000" value={minValue}
                  onChange={(e) => setMinValue(e.target.value.replace(/[^\d]/g, ""))} />
              </div>
              <div>
                <label className="label" htmlFor="max">Max. budget (₹)</label>
                <input id="max" inputMode="numeric" className="input" placeholder="e.g. 2500000" value={maxValue}
                  onChange={(e) => setMaxValue(e.target.value.replace(/[^\d]/g, ""))} />
              </div>
            </div>
          )}
        </div>

        {search.error && <ErrorBox title="Could not find RFPs">{(search.error as Error).message}</ErrorBox>}
        <div className="flex justify-end">
          <Button type="submit" icon={<Search className="h-4 w-4" />} loading={search.isPending} disabled={!keywords.length}>
            Find RFPs
          </Button>
        </div>
      </div>
      {search.isPending && (
        <div className="border-t border-slate-100 bg-brand-50/50 px-6 py-4">
          <div className="mb-2 text-sm font-medium text-slate-700">Finding tenders and reading their details…</div>
          <div className="progress-indeterminate" />
        </div>
      )}
    </form>
  );
}

function UploadRfp() {
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
        <div className="mt-1 text-sm text-slate-500">PDF with selectable text, up to 25 MB. The AI reads the buyer, deadline and budget.</div>
        <input ref={input} type="file" accept="application/pdf,.pdf" className="hidden" onChange={(e) => send(e.target.files?.[0])} />
      </div>
      {upload.isPending && (
        <div className="mt-4">
          <div className="mb-2 text-center text-sm text-slate-500">Uploading and reading the PDF…</div>
          <div className="progress-indeterminate" />
        </div>
      )}
      {upload.error && <div className="mt-4"><ErrorBox title="Upload failed">{(upload.error as Error).message}</ErrorBox></div>}
    </div>
  );
}

export default function NewSession() {
  const navigate = useNavigate();
  const [tab, setTab] = useState<"find" | "upload">("find");

  return (
    <>
      <Stepper current={0} />
      <Badge tone="violet" className="mb-3">
        <Sparkles className="h-3 w-3" /> AI-powered RFP discovery
      </Badge>
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="page-title">Start New Session</h1>
          <p className="page-subtitle">Find relevant RFPs based on your requirements, or upload one you already have.</p>
        </div>
        <Button variant="secondary" icon={<FileUp className="h-4 w-4" />} onClick={() => navigate("/rfps")}>
          Choose an existing RFP
        </Button>
      </div>

      <div className="mb-4 inline-flex rounded-lg border border-slate-200 bg-white p-1">
        {([["find", "Find RFPs"], ["upload", "Upload a PDF"]] as const).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={cx("rounded-md px-4 py-1.5 text-sm", tab === key ? "bg-brand-500 font-medium text-white" : "text-slate-600 hover:bg-slate-50")}
          >
            {label}
          </button>
        ))}
      </div>
      {tab === "find" ? <FindRfps /> : <UploadRfp />}
    </>
  );
}

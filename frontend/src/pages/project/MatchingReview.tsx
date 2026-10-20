import { ArrowDown, ArrowRight, Check, ChevronDown, Info, RotateCcw, Search, Sparkles, X } from "lucide-react";
import { useMemo, useState } from "react";

import { useCatalogueSearch, useSubmitReview } from "../../api/hooks";
import type { MatchedItem, ProductBrief, Project } from "../../api/types";
import { Badge, Button, cx, ErrorBox, ProgressBar } from "../../components/ui";
import { formatINR, formatNumber } from "../../lib/format";

type Decision =
  | { action: "accept" }
  | { action: "reject" }
  | { action: "change"; product: ProductBrief }
  | { action: "review" }; // moved back to the suggestions column, undecided

type Effective = { status: MatchedItem["status"]; product?: ProductBrief; manual?: boolean };

function briefOf(item: MatchedItem): ProductBrief | undefined {
  if (!item.product_id) return undefined;
  return {
    product_id: item.product_id,
    product_name: item.product_name!,
    product_range: item.product_range!,
    product_variant: item.product_variant ?? null,
    sap_code: item.sap_code!,
    unit_price: item.unit_price!,
    match_score: item.match_score ?? null,
  };
}

function effective(item: MatchedItem, d?: Decision): Effective {
  switch (d?.action) {
    case "accept":
      return { status: "approved", product: briefOf(item) };
    case "reject":
      return { status: "rejected" };
    case "change":
      return { status: "approved", product: d.product, manual: true };
    case "review":
      return { status: "needs_review", product: briefOf(item) };
    default:
      return { status: item.status, product: briefOf(item) };
  }
}

const STATUS_BADGE: Record<string, { tone: "green" | "amber" | "red" | "slate" | "violet"; label: string }> = {
  approved: { tone: "green", label: "Approved" },
  matched: { tone: "violet", label: "AI matched" },
  needs_review: { tone: "amber", label: "Needs review" },
  gap: { tone: "slate", label: "Not in catalogue" },
  rejected: { tone: "red", label: "Rejected" },
};

function ChangeProduct({ item, onPick }: { item: MatchedItem; onPick: (p: ProductBrief) => void }) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const { data: results, isFetching } = useCatalogueSearch(q);
  const options: ProductBrief[] = q.trim().length >= 2
    ? (results ?? []).map((p) => ({ ...p, match_score: null }))
    : item.alternatives ?? [];

  return (
    <div className="mt-2">
      <button onClick={() => setOpen((o) => !o)} className="flex items-center gap-1 text-xs font-medium text-slate-600 hover:text-brand-600">
        Change Product <ChevronDown className={cx("h-3.5 w-3.5 transition", open && "rotate-180")} />
      </button>
      {open && (
        <div className="mt-2 rounded-lg border border-slate-200 bg-white p-2">
          <div className="relative mb-2">
            <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
            <input
              autoFocus
              className="input py-1.5 pl-8 text-xs"
              placeholder="Search the catalogue, e.g. 16A 2way coral"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
          </div>
          {q.trim().length < 2 && <div className="px-1 pb-1 text-[11px] uppercase tracking-wide text-slate-400">Other suggestions</div>}
          {isFetching && <div className="px-1 py-2 text-xs text-slate-400">Searching…</div>}
          {!isFetching && !options.length && <div className="px-1 py-2 text-xs text-slate-400">No products found.</div>}
          <ul className="max-h-56 overflow-y-auto">
            {options.map((p) => (
              <li key={p.product_id}>
                <button
                  onClick={() => {
                    onPick(p);
                    setOpen(false);
                  }}
                  className="flex w-full items-center justify-between gap-3 rounded-md px-2 py-1.5 text-left text-xs hover:bg-brand-50"
                >
                  <span className="min-w-0">
                    <span className="block truncate font-medium text-slate-800">{p.product_name}</span>
                    <span className="block truncate text-slate-500">
                      {p.product_range}{p.product_variant ? ` · ${p.product_variant}` : ""} · <span className="font-mono">{p.sap_code}</span>
                    </span>
                  </span>
                  <span className="shrink-0 text-right text-slate-600">
                    {formatINR(p.unit_price)}
                    {p.match_score != null && <span className="block text-slate-400">{p.match_score}%</span>}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function StockLine({ item }: { item: MatchedItem }) {
  if (item.stock_on_hand == null) return null;
  return (
    <span className="text-xs text-slate-500">
      Stock {formatNumber(item.stock_on_hand)} on hand · {formatNumber(item.monthly_available)}/month
      {item.stock_sufficient === false && <b className="text-rose-600"> · short for qty {item.qty}</b>}
      <span className="text-slate-400"> (demo data)</span>
    </span>
  );
}

function SuggestionCard({
  item,
  eff,
  onDecide,
}: {
  item: MatchedItem;
  eff: Effective;
  onDecide: (d: Decision | undefined) => void;
}) {
  const product = eff.product;
  const isGap = item.status === "gap" && !product;
  return (
    <div className={cx("rounded-xl border border-dashed p-4", isGap ? "border-slate-300 bg-slate-50" : "border-brand-300 bg-brand-50/40")}>
      <div className="text-[11px] font-medium uppercase tracking-wide text-slate-400">Required</div>
      <div className="font-medium text-slate-800">{item.requirement}</div>
      {item.qty != null && <div className="text-xs text-slate-500">Quantity {formatNumber(item.qty)}</div>}
      <ArrowDown className="my-1.5 h-3.5 w-3.5 text-slate-400" />

      {product ? (
        <>
          <div className="text-[11px] font-medium uppercase tracking-wide text-slate-400">Suggested catalogue product</div>
          <div className="font-semibold text-slate-800">{product.product_name}</div>
          <div className="text-xs text-slate-500">
            {product.product_range}{product.product_variant ? ` · ${product.product_variant}` : ""} ·{" "}
            <span className="font-mono">{product.sap_code}</span> · {formatINR(product.unit_price)}
          </div>
          {item.match_score != null && (
            <div className="mt-2">
              <div className="flex items-center gap-2">
                <span className="text-lg font-semibold text-brand-600">{item.match_score}% Match</span>
                {item.confidence && <Badge tone={item.confidence === "low" ? "red" : "amber"}>{item.confidence}</Badge>}
              </div>
              <ProgressBar value={item.match_score} className="mt-1" />
            </div>
          )}
        </>
      ) : (
        <div className="text-sm text-slate-500">No similar product in the catalogue.</div>
      )}

      {item.duplicate_of && (
        <div className="mt-2 rounded-md bg-amber-50 px-2 py-1 text-xs text-amber-800">
          Same product as {item.duplicate_of}: check this isn't quoted twice.
        </div>
      )}
      {item.rationale && !item.duplicate_of && !(isGap && /no similar product/i.test(item.rationale)) && (
        <p className="mt-2 text-xs text-slate-600">{item.rationale}</p>
      )}
      <div className="mt-1"><StockLine item={item} /></div>

      <div className="mt-3 flex flex-wrap gap-2">
        {product && (
          <Button size="sm" icon={<Check className="h-3.5 w-3.5" />} onClick={() => onDecide({ action: "accept" })}>
            Accept
          </Button>
        )}
        <Button size="sm" variant="secondary" icon={<X className="h-3.5 w-3.5" />} onClick={() => onDecide({ action: "reject" })}>
          {product ? "Reject" : "Don't quote"}
        </Button>
      </div>
      <ChangeProduct item={item} onPick={(p) => onDecide({ action: "change", product: p })} />
    </div>
  );
}

export default function MatchingReview({ project }: { project: Project }) {
  const submit = useSubmitReview(project.id);
  const [decisions, setDecisions] = useState<Record<string, Decision>>({});
  const [compliance, setCompliance] = useState<Record<string, "confirm" | "not_met">>({});

  const products = project.matched_items.filter((m) => m.type === "product");
  const complianceItems = project.matched_items.filter((m) => m.type === "compliance");

  const decide = (id: string, d: Decision | undefined) =>
    setDecisions((all) => {
      const next = { ...all };
      if (d) next[id] = d;
      else delete next[id];
      return next;
    });

  const view = products.map((item) => ({ item, d: decisions[item.id], eff: effective(item, decisions[item.id]) }));
  const pending = view.filter(({ item, d }) => (!d && (item.status === "needs_review" || item.status === "gap")) || d?.action === "review");
  const blocking = pending.filter(({ item }) => item.status !== "gap" || decisions[item.id]?.action === "review");
  const selected = view.filter(({ eff, d }) => (eff.status === "approved" || eff.status === "matched") && d?.action !== "review");

  const score = useMemo(() => {
    const total = view.reduce((s, v) => s + v.item.weight, 0);
    const found = view.filter((v) => v.eff.status === "approved" || v.eff.status === "matched" || v.eff.status === "needs_review")
      .filter((v) => v.eff.product)
      .reduce((s, v) => s + v.item.weight, 0);
    return total ? Math.round((found / total) * 1000) / 10 : 0;
  }, [view]);

  function continueToPricing() {
    const out: Record<string, unknown>[] = [];
    for (const [id, d] of Object.entries(decisions)) {
      if (d.action === "accept" || d.action === "reject") out.push({ id, action: d.action });
      if (d.action === "change") out.push({ id, action: "change", product_id: d.product.product_id });
    }
    for (const [id, action] of Object.entries(compliance)) out.push({ id, action });
    submit.mutate({ decisions: out, accept_all: true });
  }

  return (
    <>
      <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="page-title">Product Matching</h1>
          <p className="page-subtitle">Match RFP requirements with products in your catalogue.</p>
        </div>
        <div className="card flex w-72 items-center gap-3 p-4">
          <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-50 text-brand-500"><Sparkles className="h-4 w-4" /></span>
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-medium uppercase tracking-wide text-slate-500" title="Weighted share of product requirements that have a catalogue product">
                AI matching score <Info className="inline h-3 w-3" />
              </span>
              <span className="text-2xl font-semibold text-brand-600">{score}%</span>
            </div>
            <ProgressBar value={score} className="mt-1" />
          </div>
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-3">
        {/* Requirements */}
        <section className="card p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-semibold text-slate-800">RFP Requirements</h2>
            <Badge>{products.length} items</Badge>
          </div>
          <ul className="-mr-2 max-h-[70vh] space-y-2 overflow-y-auto pr-2">
            {view.map(({ item, eff }) => {
              const badge = STATUS_BADGE[eff.status] ?? STATUS_BADGE.matched;
              return (
                <li key={item.id} className="rounded-lg border border-slate-200 p-3">
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-sm font-medium text-slate-800">{item.requirement}</span>
                    <Badge tone={badge.tone}>{badge.label}</Badge>
                  </div>
                  <div className="mt-0.5 truncate text-xs text-slate-500">
                    {eff.product ? `${eff.product.product_name} · ${eff.product.product_range}` : "No catalogue product"}
                  </div>
                  {item.qty != null && <div className="mt-1 text-xs text-slate-600">Quantity <b>{formatNumber(item.qty)}</b></div>}
                </li>
              );
            })}
          </ul>
        </section>

        {/* Suggestions needing a decision */}
        <section className="card p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-semibold text-slate-800">AI Suggested Matches</h2>
            <div className="flex items-center gap-2">
              {pending.length > 0 && <Badge tone="violet"><Sparkles className="h-3 w-3" />{pending.length} to review</Badge>}
              {blocking.some(({ item }) => item.product_id && !item.duplicate_of) && (
                <button
                  className="text-xs font-medium text-brand-600 hover:underline"
                  title="Possible duplicates are left for you to decide"
                  onClick={() =>
                    blocking.forEach(({ item }) => item.product_id && !item.duplicate_of && decide(item.id, { action: "accept" }))
                  }
                >
                  Accept all
                </button>
              )}
            </div>
          </div>
          {pending.length === 0 ? (
            <div className="flex flex-col items-center gap-2 py-10 text-center text-sm text-slate-500">
              <Check className="h-6 w-6 text-emerald-500" /> Every suggestion has been reviewed.
            </div>
          ) : (
            <div className="space-y-3">
              {pending.map(({ item, eff }) => (
                <SuggestionCard key={item.id} item={item} eff={eff} onDecide={(d) => decide(item.id, d)} />
              ))}
            </div>
          )}
        </section>

        {/* Selected products */}
        <section className="card p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-semibold text-slate-800">Selected Products</h2>
            <Badge tone="green">{selected.length} selected</Badge>
          </div>
          <ul className="-mr-2 max-h-[70vh] space-y-2 overflow-y-auto pr-2">
            {selected.map(({ item, eff, d }) => (
              <li key={item.id} className="flex gap-3 rounded-lg border border-slate-200 p-3">
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-emerald-600">
                  <Check className="h-3 w-3" />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[11px] uppercase tracking-wide text-slate-400">
                    {item.requirement}{item.qty != null && ` · qty ${formatNumber(item.qty)}`}
                  </div>
                  <div className="truncate text-sm font-semibold text-slate-800">{eff.product?.product_name}</div>
                  <div className="truncate font-mono text-[11px] text-slate-500">{eff.product?.sap_code} · {eff.product?.product_range}</div>
                  <div className="mt-1 flex flex-wrap items-center gap-2 text-xs">
                    {d ? <Badge tone="green">{eff.manual ? "Chosen by you" : "Approved by you"}</Badge> : <Badge tone="violet">AI match</Badge>}
                    {item.match_score != null && !eff.manual && <span className="text-slate-500">{item.match_score}%</span>}
                  </div>
                </div>
                <button
                  onClick={() => decide(item.id, d ? undefined : { action: "review" })}
                  className="self-start rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
                  title={d ? "Undo" : "Review this match"}
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                </button>
              </li>
            ))}
            {!selected.length && <li className="py-8 text-center text-sm text-slate-400">Nothing selected yet.</li>}
          </ul>
        </section>
      </div>

      {complianceItems.length > 0 && (
        <section className="card mt-5 p-4">
          <div className="mb-1 flex items-center justify-between">
            <h2 className="font-semibold text-slate-800">Compliance & eligibility</h2>
            <span className="text-xs text-slate-500">
              {Object.keys(compliance).length} of {complianceItems.length} confirmed
            </span>
          </div>
          <p className="mb-3 text-xs text-slate-500">These are not catalogue products. Confirm each one against your company documents.</p>
          <ul className="divide-y divide-slate-100">
            {complianceItems.map((c) => (
              <li key={c.id} className="flex items-center justify-between gap-3 py-2">
                <span className="text-sm text-slate-700">{c.requirement}</span>
                <div className="flex shrink-0 gap-1.5">
                  {(["confirm", "not_met"] as const).map((action) => (
                    <button
                      key={action}
                      onClick={() =>
                        setCompliance((all) => {
                          const next = { ...all };
                          if (next[c.id] === action) delete next[c.id];
                          else next[c.id] = action;
                          return next;
                        })
                      }
                      className={cx(
                        "rounded-md border px-2.5 py-1 text-xs font-medium",
                        compliance[c.id] === action
                          ? action === "confirm" ? "border-emerald-300 bg-emerald-50 text-emerald-700" : "border-rose-300 bg-rose-50 text-rose-700"
                          : "border-slate-200 text-slate-500 hover:bg-slate-50",
                      )}
                    >
                      {action === "confirm" ? "Met" : "Not met"}
                    </button>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      {submit.error && <div className="mt-4"><ErrorBox>{(submit.error as Error).message}</ErrorBox></div>}
      <div className="mt-6 flex items-center justify-between gap-4">
        <span className="text-sm text-slate-500">
          {blocking.length ? `${blocking.length} suggestion${blocking.length > 1 ? "s" : ""} still need your review.` : "All suggestions reviewed."}
        </span>
        <Button icon={<ArrowRight className="h-4 w-4" />} disabled={blocking.length > 0} loading={submit.isPending} onClick={continueToPricing}>
          Continue to Pricing
        </Button>
      </div>
    </>
  );
}

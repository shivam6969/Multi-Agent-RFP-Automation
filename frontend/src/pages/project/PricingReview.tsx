import { ArrowRight, Download } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { pricingPreview, usePricingRules, useSubmitReview, type PricingInputs } from "../../api/hooks";
import type { LineItem, PricingSummary, Project } from "../../api/types";
import { Badge, Button, cx, ErrorBox, LinkButton, Notice } from "../../components/ui";
import { formatINR, formatINRPaise, formatNumber } from "../../lib/format";

export default function PricingReview({ project }: { project: Project }) {
  const submit = useSubmitReview(project.id);
  const { data: rules } = usePricingRules();
  const saved = project.pricing_summary;

  const [margin, setMargin] = useState<number>(saved.margin_pct ?? 20);
  const [advance, setAdvance] = useState<boolean>(Boolean(saved.advance_payment));
  const [freight, setFreight] = useState<string>(saved.freight ? String(saved.freight) : "");
  const [qty, setQty] = useState<Record<string, string>>(() =>
    Object.fromEntries(project.line_items.map((li) => [li.requirement_id, String(li.qty)])),
  );
  const [preview, setPreview] = useState<{ line_items: LineItem[]; pricing_summary: PricingSummary } | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  // Only quantities the reviewer actually changed: the server labels sent
  // quantities "reviewer", which must not hide the "default" warnings.
  const originalQty = useMemo(
    () => Object.fromEntries(project.line_items.map((li) => [li.requirement_id, li.qty])),
    [project.line_items],
  );
  const quantities = useMemo(
    () =>
      Object.fromEntries(
        Object.entries(qty)
          .filter(([k, v]) => Number(v) > 0 && Number(v) !== originalQty[k])
          .map(([k, v]) => [k, Number(v)]),
      ),
    [qty, originalQty],
  );
  const qtyInvalid = Object.values(qty).some((v) => !(Number(v) > 0));
  const inputs: PricingInputs = { margin_pct: margin, advance_payment: advance, freight_inr: Number(freight) || 0, quantities };

  // Live, deterministic re-price on the server (debounced). Saves nothing.
  const inputsKey = JSON.stringify(inputs);
  useEffect(() => {
    if (qtyInvalid) return;
    const t = setTimeout(() => {
      pricingPreview(project.id, JSON.parse(inputsKey))
        .then((p) => {
          setPreview(p);
          setPreviewError(null);
        })
        .catch((e: Error) => setPreviewError(e.message));
    }, 250);
    return () => clearTimeout(t);
  }, [inputsKey, project.id, qtyInvalid]);

  const lines = preview?.line_items ?? project.line_items;
  const summary = (preview?.pricing_summary ?? saved) as PricingSummary;
  const editedQty = (li: LineItem) => String(originalQty[li.requirement_id]) !== qty[li.requirement_id];
  const minM = rules?.min_margin_pct ?? 0;
  const maxM = rules?.max_margin_pct ?? 60;

  return (
    <>
      <h1 className="page-title">Pricing</h1>
      <p className="page-subtitle mb-6">Review quantities, margin and payment terms before generating the proposal.</p>

      {(summary.qty_defaulted_count ?? 0) > 0 && (
        <div className="mb-4">
          <Notice>
            {summary.qty_defaulted_count} line{summary.qty_defaulted_count > 1 ? "s use" : " uses"} the default quantity of{" "}
            {rules?.default_qty ?? 100} because the RFP doesn't state one. Please check the quantities marked <b>default</b>.
          </Notice>
        </div>
      )}
      {project.unpriced.length > 0 && (
        <div className="mb-4">
          <Notice tone="blue">
            Not priced: {project.unpriced.map((u) => `${u.requirement}${u.note ? ` (${u.note})` : ""}`).join("; ")}.
          </Notice>
        </div>
      )}

      <div className="card mb-5 flex flex-wrap items-center gap-6 p-5">
        <div className="min-w-44">
          <div className="font-semibold text-slate-800">Proposal margin</div>
          <div className="text-xs text-slate-500">Applied to the catalogue price on every line</div>
        </div>
        <div className="flex items-center gap-1 rounded-lg border border-slate-200 px-3 py-1.5">
          <input
            type="number"
            min={minM}
            max={maxM}
            step={0.5}
            value={margin}
            onChange={(e) => setMargin(Math.min(maxM, Math.max(minM, Number(e.target.value) || 0)))}
            className="w-14 text-right text-sm outline-none"
          />
          <span className="text-sm text-slate-400">%</span>
        </div>
        <input
          type="range"
          min={minM}
          max={maxM}
          step={0.5}
          value={margin}
          onChange={(e) => setMargin(Number(e.target.value))}
          className="min-w-48 flex-1"
        />
      </div>

      <div className="card mb-5 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-100 text-left text-[11px] uppercase tracking-wide text-slate-500">
              <th className="px-4 py-3 font-medium">Product</th>
              <th className="px-4 py-3 font-medium">Quantity</th>
              <th className="px-4 py-3 text-right font-medium">Base price</th>
              <th className="px-4 py-3 text-right font-medium">Margin</th>
              <th className="px-4 py-3 text-right font-medium">Vol. disc.</th>
              <th className="px-4 py-3 text-right font-medium">GST</th>
              <th className="px-4 py-3 text-right font-medium">Final price</th>
              <th className="px-4 py-3 text-right font-medium">Total</th>
            </tr>
          </thead>
          <tbody>
            {lines.map((li) => {
              const value = qty[li.requirement_id] ?? String(li.qty);
              return (
                <tr key={li.requirement_id} className="border-b border-slate-100 last:border-0">
                  <td className="px-4 py-3">
                    <div className="font-medium text-slate-800">{li.product_name}</div>
                    <div className="font-mono text-[11px] text-slate-500">{li.sap_code} · {li.product_range}</div>
                    <div className="text-[11px] text-slate-400">For: {li.requirement}</div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <input
                        className={cx("input w-24 py-1", !(Number(value) > 0) && "border-rose-300")}
                        inputMode="numeric"
                        value={value}
                        onChange={(e) => setQty((q) => ({ ...q, [li.requirement_id]: e.target.value.replace(/[^\d]/g, "") }))}
                      />
                      {li.qty_source === "default" && !editedQty(li) && <Badge tone="amber">default</Badge>}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-right text-slate-700">{formatINRPaise(li.base_unit_price)}</td>
                  <td className="px-4 py-3 text-right text-slate-600">{li.margin_pct}%</td>
                  <td className="px-4 py-3 text-right text-slate-600">{li.volume_discount_pct ? `${li.volume_discount_pct}%` : "—"}</td>
                  <td className="px-4 py-3 text-right text-slate-600">{li.gst_pct}%</td>
                  <td className="px-4 py-3 text-right text-slate-700">{formatINRPaise(li.final_unit_price)}</td>
                  <td className="px-4 py-3 text-right font-semibold text-slate-800">{formatINR(li.line_total)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="grid gap-5 lg:grid-cols-[1fr_22rem]">
        <div className="space-y-4">
          <div className="card space-y-4 p-5">
            <h2 className="font-semibold text-slate-800">Payment & delivery terms</h2>
            <label className="flex items-center gap-3 text-sm text-slate-700">
              <input type="checkbox" className="h-4 w-4 accent-brand-500" checked={advance} onChange={(e) => setAdvance(e.target.checked)} />
              Advance payment discount ({rules?.payment_terms_discount_pct ?? 2}% off the goods value)
            </label>
            <div className="flex items-center gap-3 text-sm text-slate-700">
              <span className="w-44">Freight (₹, GST applies)</span>
              <input
                className="input w-40 py-1.5"
                inputMode="decimal"
                placeholder="0 = buyer pays"
                value={freight}
                onChange={(e) => setFreight(e.target.value.replace(/[^\d.]/g, ""))}
              />
            </div>
            <p className="text-xs text-slate-500">
              Volume discount tiers: {rules?.volume_discount_tiers.map((t) => `${formatNumber(t.min_qty)}+ units ${t.discount_pct}%`).join(" · ")}.
            </p>
          </div>
          {previewError && <ErrorBox title="Could not update the preview">{previewError}</ErrorBox>}
          {submit.error && <ErrorBox>{(submit.error as Error).message}</ErrorBox>}
          <div className="flex flex-wrap gap-3">
            <Button
              icon={<ArrowRight className="h-4 w-4" />}
              loading={submit.isPending}
              disabled={qtyInvalid}
              onClick={() => submit.mutate({ margin_pct: margin, advance_payment: advance, freight_inr: Number(freight) || 0, quantities })}
            >
              Generate Proposal
            </Button>
            <LinkButton
              href={`/api/projects/${project.id}/quote.csv`}
              variant="secondary"
              icon={<Download className="h-4 w-4" />}
              title="The quote as last saved (before your unsaved changes here)"
            >
              Saved quote (CSV)
            </LinkButton>
          </div>
        </div>

        <div className="rounded-xl bg-cocoa-900 p-5 text-sm text-white/80">
          <Row label="Catalogue value" value={formatINR(summary.subtotal_base)} />
          <Row label={`Margin (${summary.margin_pct}%) after volume discounts`} value={formatINR(summary.margin_value)} />
          {summary.payment_discount_value > 0 && (
            <Row label={`Advance-payment discount (${summary.payment_discount_pct}%)`} value={`− ${formatINR(summary.payment_discount_value)}`} />
          )}
          {summary.freight > 0 && <Row label="Freight" value={formatINR(summary.freight)} />}
          <Row label={`GST (${rules?.gst_rate_pct ?? 18}%)`} value={formatINR(summary.total_gst)} />
          <div className="mt-4 flex items-end justify-between border-t border-white/15 pt-4">
            <span>Grand Total</span>
            <span className="text-3xl font-semibold text-white">{formatINR(summary.grand_total)}</span>
          </div>
          <div className="mt-1 text-right text-xs text-white/50">{summary.items_count} lines · incl. GST</div>
        </div>
      </div>
    </>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 py-1">
      <span>{label}</span>
      <span className="font-medium text-white">{value}</span>
    </div>
  );
}

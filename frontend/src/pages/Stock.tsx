import { ArrowDown, ArrowUp, Search, Warehouse } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";

import { useStock } from "../api/hooks";
import type { CatalogueProduct } from "../api/types";
import { Badge, Button, cx, EmptyState, ErrorBox, Notice, PageHeader, Spinner } from "../components/ui";
import { formatINR, formatINRShort, formatNumber } from "../lib/format";

const PAGE = 25;

const onHand = (p: CatalogueProduct) => p.stock_on_hand ?? 0;
const stockValue = (p: CatalogueProduct) => onHand(p) * p.unit_price;

const SORTS = {
  name: (a: CatalogueProduct, b: CatalogueProduct) =>
    a.product_range.localeCompare(b.product_range) || a.product_name.localeCompare(b.product_name),
  on_hand: (a: CatalogueProduct, b: CatalogueProduct) => onHand(a) - onHand(b),
  monthly: (a: CatalogueProduct, b: CatalogueProduct) => (a.monthly_available ?? 0) - (b.monthly_available ?? 0),
  lead: (a: CatalogueProduct, b: CatalogueProduct) => (a.lead_time_days ?? 0) - (b.lead_time_days ?? 0),
  price: (a: CatalogueProduct, b: CatalogueProduct) => a.unit_price - b.unit_price,
  value: (a: CatalogueProduct, b: CatalogueProduct) => stockValue(a) - stockValue(b),
};
type SortKey = keyof typeof SORTS;

function Tile({ label, value, hint, tone }: { label: string; value: string; hint?: string; tone?: "amber" }) {
  return (
    <div className="card p-4">
      <div className="text-xs font-medium text-slate-500">{label}</div>
      <div className={cx("mt-1 text-2xl font-semibold", tone === "amber" ? "text-amber-700" : "text-slate-800")}>{value}</div>
      {hint && <div className="mt-0.5 text-xs text-slate-400">{hint}</div>}
    </div>
  );
}

export default function Stock() {
  const { data, isLoading, error } = useStock();
  const [q, setQ] = useState("");
  const [range, setRange] = useState("all");
  const [status, setStatus] = useState<"all" | "low" | "ok">("all");
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "name", desc: false });
  const [page, setPage] = useState(0);

  const products = data?.products ?? [];
  const low = data?.low_stock_units ?? 0;
  const isLow = (p: CatalogueProduct) => onHand(p) < low;

  const ranges = useMemo(() => [...new Set(products.map((p) => p.product_range))].sort(), [products]);

  const filtered = useMemo(() => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    const rows = products.filter((p) => {
      if (range !== "all" && p.product_range !== range) return false;
      if (status === "low" && !isLow(p)) return false;
      if (status === "ok" && isLow(p)) return false;
      const hay = [p.product_name, p.product_range, p.product_variant, p.sap_code].join(" ").toLowerCase();
      return words.every((w) => hay.includes(w));
    });
    rows.sort(SORTS[sort.key]);
    if (sort.desc) rows.reverse();
    return rows;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [products, q, range, status, sort, low]);

  if (isLoading) return <div className="flex justify-center p-16"><Spinner /></div>;
  if (error || !data) return <ErrorBox title="Could not load stock">{(error as Error)?.message ?? "Try again."}</ErrorBox>;

  const lowCount = products.filter(isLow).length;
  const pages = Math.max(1, Math.ceil(filtered.length / PAGE));
  const shown = filtered.slice(page * PAGE, page * PAGE + PAGE);
  const reset = () => setPage(0);

  function SortHeader({ k, children, right }: { k: SortKey; children: ReactNode; right?: boolean }) {
    const active = sort.key === k;
    return (
      <th className={cx("px-4 py-3 font-medium", right && "text-right")}>
        <button
          onClick={() => {
            setSort(active ? { key: k, desc: !sort.desc } : { key: k, desc: k !== "name" });
            reset();
          }}
          className={cx("inline-flex items-center gap-1 uppercase tracking-wide hover:text-slate-700", active && "text-slate-700")}
        >
          {children}
          {active && (sort.desc ? <ArrowDown className="h-3 w-3" /> : <ArrowUp className="h-3 w-3" />)}
        </button>
      </th>
    );
  }

  return (
    <>
      <PageHeader title="Stock" subtitle="What you have on hand for every catalogue product, and how fast you can get more." />

      {data.is_demo && (
        <div className="mb-6">
          <Notice tone="blue">
            These stock figures are <b>demo data</b>, generated for each product. Real stock would come from your inventory or ERP system.
          </Notice>
        </div>
      )}

      <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Tile label="Products" value={formatNumber(products.length)} hint={`${ranges.length} ranges`} />
        <Tile label="Units on hand" value={formatNumber(products.reduce((s, p) => s + onHand(p), 0))} />
        <Tile label="Stock value" value={formatINRShort(products.reduce((s, p) => s + stockValue(p), 0))} hint="at catalogue price" />
        <Tile label="Low stock" value={formatNumber(lowCount)} hint={`fewer than ${low} units`} tone={lowCount ? "amber" : undefined} />
      </div>

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="relative min-w-[16rem] flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            className="input pl-9"
            placeholder="Search name, range or SAP code"
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              reset();
            }}
          />
        </div>
        <select
          className="input w-auto"
          value={range}
          onChange={(e) => {
            setRange(e.target.value);
            reset();
          }}
        >
          <option value="all">All ranges</option>
          {ranges.map((r) => <option key={r} value={r}>{r}</option>)}
        </select>
        <div className="flex gap-1 rounded-lg bg-slate-100 p-1">
          {([["all", "All"], ["low", "Low stock"], ["ok", "In stock"]] as const).map(([key, label]) => (
            <button
              key={key}
              onClick={() => {
                setStatus(key);
                reset();
              }}
              className={cx(
                "rounded-md px-3 py-1.5 text-sm transition",
                status === key ? "bg-white font-medium text-slate-800 shadow-sm" : "text-slate-500 hover:text-slate-700",
              )}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="card">
        {!shown.length ? (
          <EmptyState icon={<Warehouse className="h-5 w-5" />} title="No products match">
            Try a different search or filter.
          </EmptyState>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100 text-left text-[11px] text-slate-500">
                  <SortHeader k="name">Product</SortHeader>
                  <SortHeader k="on_hand" right>On hand</SortHeader>
                  <SortHeader k="monthly" right>Monthly supply</SortHeader>
                  <SortHeader k="lead" right>Lead time</SortHeader>
                  <SortHeader k="price" right>Unit price</SortHeader>
                  <SortHeader k="value" right>Stock value</SortHeader>
                  <th className="px-4 py-3 font-medium uppercase tracking-wide">Status</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((p) => (
                  <tr key={p.product_id} className="border-b border-slate-100 last:border-0 hover:bg-slate-50">
                    <td className="px-4 py-3">
                      <div className="font-medium text-slate-800">{p.product_name}</div>
                      <div className="text-xs text-slate-500">
                        {p.product_range}
                        {p.product_variant ? ` · ${p.product_variant}` : ""} · SAP {p.sap_code || "—"} · {p.packing}
                      </div>
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-right font-semibold tabular-nums text-slate-800">{formatNumber(p.stock_on_hand)}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-right tabular-nums text-slate-600">{formatNumber(p.monthly_available)}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-right tabular-nums text-slate-600">
                      {p.lead_time_days == null ? "—" : `${p.lead_time_days} days`}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-right tabular-nums text-slate-600">{formatINR(p.unit_price)}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-right tabular-nums text-slate-800">{formatINR(stockValue(p))}</td>
                    <td className="px-4 py-3">
                      {isLow(p) ? <Badge tone="amber">Low stock</Badge> : <Badge tone="green">In stock</Badge>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {filtered.length > PAGE && (
          <div className="flex items-center justify-between border-t border-slate-100 px-4 py-3 text-sm text-slate-500">
            <span>
              Showing {page * PAGE + 1}–{Math.min(filtered.length, (page + 1) * PAGE)} of {filtered.length}
            </span>
            <div className="flex gap-2">
              <Button size="sm" variant="secondary" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</Button>
              <Button size="sm" variant="secondary" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>Next</Button>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

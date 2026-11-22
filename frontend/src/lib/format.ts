// Indian number formats: ₹23,67,729 and "₹24.5 L" / "₹1.8 Cr".

const inr = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });
const inrPaise = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});
const plain = new Intl.NumberFormat("en-IN");

export const formatINR = (n: number | null | undefined) => (n == null ? "—" : inr.format(n));
export const formatINRPaise = (n: number | null | undefined) => (n == null ? "—" : inrPaise.format(n));
export const formatNumber = (n: number | null | undefined) => (n == null ? "—" : plain.format(n));

export function formatINRShort(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n >= 1e7) return `₹${(n / 1e7).toFixed(2).replace(/\.?0+$/, "")} Cr`;
  if (n >= 1e5) return `₹${(n / 1e5).toFixed(1).replace(/\.0$/, "")} L`;
  return inr.format(n);
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  // Backend datetimes are UTC; SQLite drops the zone, so add it back.
  const d = new Date(/[zZ]|[+-]\d\d:\d\d$/.test(iso) ? iso : `${iso}Z`);
  return d.toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
}

export const projectRef = (id: string) => `SYN-${id.slice(0, 6).toUpperCase()}`;

// "p.2 · Table T1, row 5" / "pp.3–4": where a requirement sits in the RFP.
export function sourceLabel(
  source: { page?: number; table?: string; row?: string; pages?: [number, number] } | null | undefined,
): string {
  if (!source) return "";
  if (source.table) return `p.${source.page} · Table ${source.table}, row ${source.row}`;
  if (source.pages) {
    const [a, b] = source.pages;
    return a === b ? `p.${a}` : `pp.${a}–${b}`;
  }
  return source.page ? `p.${source.page}` : "";
}

export const initials = (name: string) =>
  name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0]!.toUpperCase())
    .join("");

// Calendar dates ("2026-10-25", no time zone) from the backend: parse as local days.
export function parseDay(iso: string): Date {
  const [y, m, d] = iso.slice(0, 10).split("-").map(Number);
  return new Date(y!, (m ?? 1) - 1, d ?? 1);
}

export const dayKey = (d: Date) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;

export const formatDay = (iso: string | null | undefined) =>
  iso ? parseDay(iso).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" }) : "—";

/** "today", "tomorrow", "in 5 days", "3 days ago" relative to *today* (both ISO days). */
export function relativeDay(iso: string, today: string): string {
  const days = Math.round((parseDay(iso).getTime() - parseDay(today).getTime()) / 86_400_000);
  if (days === 0) return "today";
  if (days === 1) return "tomorrow";
  if (days === -1) return "yesterday";
  return days > 0 ? `in ${days} days` : `${-days} days ago`;
}

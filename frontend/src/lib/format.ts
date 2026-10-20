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

export const initials = (name: string) =>
  name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0]!.toUpperCase())
    .join("");

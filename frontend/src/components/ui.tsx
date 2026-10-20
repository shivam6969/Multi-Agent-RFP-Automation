import { AlertTriangle, Loader2 } from "lucide-react";
import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from "react";

export function cx(...classes: (string | false | null | undefined)[]) {
  return classes.filter(Boolean).join(" ");
}

type Variant = "primary" | "secondary" | "ghost" | "danger";

const variants: Record<Variant, string> = {
  primary: "bg-brand-500 text-white hover:bg-brand-600 disabled:bg-brand-300",
  secondary: "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:text-slate-400",
  ghost: "text-slate-600 hover:bg-slate-100 disabled:text-slate-300",
  danger: "border border-rose-200 bg-white text-rose-600 hover:bg-rose-50",
};

export function buttonClass(variant: Variant = "primary", size: "sm" | "md" = "md") {
  return cx(
    "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition disabled:cursor-not-allowed",
    size === "sm" ? "px-3 py-1.5 text-xs" : "px-4 py-2.5 text-sm",
    variants[variant],
  );
}

/** A link (e.g. a file download) that looks like a button. */
export function LinkButton({
  variant = "primary",
  icon,
  children,
  className,
  ...rest
}: AnchorHTMLAttributes<HTMLAnchorElement> & { variant?: Variant; icon?: ReactNode }) {
  return (
    <a {...rest} className={cx(buttonClass(variant), className)}>
      {icon}
      {children}
    </a>
  );
}

export function Button({
  variant = "primary",
  size = "md",
  loading,
  icon,
  className,
  children,
  disabled,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: "sm" | "md";
  loading?: boolean;
  icon?: ReactNode;
}) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={cx(buttonClass(variant, size), className)}
    >
      {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : icon}
      {children}
    </button>
  );
}

type Tone = "green" | "amber" | "red" | "blue" | "violet" | "slate";

const tones: Record<Tone, string> = {
  green: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  amber: "bg-amber-50 text-amber-700 ring-amber-200",
  red: "bg-rose-50 text-rose-700 ring-rose-200",
  blue: "bg-sky-50 text-sky-700 ring-sky-200",
  violet: "bg-brand-50 text-brand-700 ring-brand-200",
  slate: "bg-slate-100 text-slate-600 ring-slate-200",
};

export function Badge({ tone = "slate", children, className }: { tone?: Tone; children: ReactNode; className?: string }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function ProgressBar({ value, className }: { value: number; className?: string }) {
  return (
    <div className={cx("h-1.5 w-full overflow-hidden rounded-full bg-slate-100", className)}>
      <div className="h-full rounded-full bg-brand-500 transition-all" style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={cx("h-5 w-5 animate-spin text-brand-500", className)} />;
}

export function ErrorBox({ title = "Something went wrong", children }: { title?: string; children: ReactNode }) {
  return (
    <div className="flex gap-3 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <div>
        <div className="font-medium">{title}</div>
        <div className="mt-0.5 text-rose-700">{children}</div>
      </div>
    </div>
  );
}

export function Notice({ children, tone = "amber" }: { children: ReactNode; tone?: "amber" | "blue" }) {
  return (
    <div
      className={cx(
        "flex gap-2 rounded-lg border p-3 text-sm",
        tone === "amber" ? "border-amber-200 bg-amber-50 text-amber-800" : "border-sky-200 bg-sky-50 text-sky-800",
      )}
    >
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <div>{children}</div>
    </div>
  );
}

export function EmptyState({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-14 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-brand-50 text-brand-500">{icon}</div>
      <div className="font-medium text-slate-700">{title}</div>
      {children && <div className="max-w-sm text-sm text-slate-500">{children}</div>}
    </div>
  );
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="page-title">{title}</h1>
        {subtitle && <p className="page-subtitle">{subtitle}</p>}
      </div>
      {actions}
    </div>
  );
}

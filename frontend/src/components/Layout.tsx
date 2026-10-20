import {
  Bell,
  Boxes,
  ChevronDown,
  FileCheck2,
  FileText,
  History,
  Home,
  IndianRupee,
  LogOut,
  PanelLeftClose,
  PanelLeftOpen,
  PlusCircle,
  Settings as SettingsIcon,
  User as UserIcon,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, Outlet, useLocation, useNavigate } from "react-router-dom";

import { useDashboard, useHealth, useLogout, useMe } from "../api/hooks";
import { initials } from "../lib/format";
import ReminderList, { URGENT } from "./Reminders";
import { cx, Notice } from "./ui";

const NAV = [
  { to: "/", label: "Home", icon: Home, end: true },
  { to: "/new", label: "New Session", icon: PlusCircle },
  { to: "/rfps", label: "RFPs / Documents", icon: FileText },
  { to: "/projects?stage=matching_review", label: "Product Matching", icon: Boxes },
  { to: "/projects?stage=pricing_review", label: "Pricing", icon: IndianRupee },
  { to: "/projects?stage=done", label: "Generated Proposals", icon: FileCheck2 },
  { to: "/history", label: "Completed / History", icon: History },
];

export function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-500 text-white">
        <FileCheck2 className="h-4 w-4" />
      </div>
      {!compact && <span className="font-semibold text-slate-800">Synapse</span>}
    </div>
  );
}

/** Close a dropdown when the user clicks anywhere outside it. */
function useClickOutside(onOutside: () => void) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && onOutside();
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [onOutside]);
  return ref;
}

function NotificationBell() {
  const { data } = useDashboard();
  const [open, setOpen] = useState(false);
  const ref = useClickOutside(useCallback(() => setOpen(false), []));
  const urgent = data?.reminders.filter((r) => URGENT.includes(r.status)).length ?? 0;

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="relative rounded-lg p-2 text-slate-500 hover:bg-slate-50"
        title={urgent ? `${urgent} reminder${urgent === 1 ? "" : "s"} need attention` : "Reminders"}
      >
        <Bell className="h-5 w-5" />
        {urgent > 0 && <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-brand-500 ring-2 ring-white" />}
      </button>
      {open && data && (
        <div className="absolute right-0 z-20 mt-1 w-96 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-lg">
          <div className="border-b border-slate-100 px-3 py-2 text-sm font-semibold text-slate-800">Reminders</div>
          <div className="max-h-96 overflow-y-auto">
            <ReminderList reminders={data.reminders} today={data.today} compact onNavigate={() => setOpen(false)} />
          </div>
        </div>
      )}
    </div>
  );
}

function UserMenu() {
  const { data: user } = useMe();
  const logout = useLogout();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const ref = useClickOutside(useCallback(() => setOpen(false), []));

  if (!user) return null;
  return (
    <div className="relative" ref={ref}>
      <button onClick={() => setOpen((o) => !o)} className="flex items-center gap-2.5 rounded-lg px-2 py-1 hover:bg-slate-50">
        <span className="flex h-8 w-8 items-center justify-center rounded-full bg-slate-800 text-xs font-semibold text-white">
          {initials(user.name)}
        </span>
        <span className="hidden text-left leading-tight sm:block">
          <span className="block text-sm font-semibold text-slate-800">{user.name}</span>
          <span className="block text-xs text-slate-500">{user.email}</span>
        </span>
        <ChevronDown className="h-4 w-4 text-slate-400" />
      </button>
      {open && (
        <div className="absolute right-0 z-20 mt-1 w-44 rounded-lg border border-slate-200 bg-white p-1 shadow-lg">
          <div className="flex items-center gap-2 rounded-md px-3 py-2 text-sm text-slate-500">
            <UserIcon className="h-4 w-4" /> Profile
          </div>
          <button
            onClick={() => logout.mutate(undefined, { onSuccess: () => navigate("/login") })}
            className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-slate-700 hover:bg-slate-50"
          >
            <LogOut className="h-4 w-4" /> Log out
          </button>
        </div>
      )}
    </div>
  );
}

export default function Layout() {
  const { data: user } = useMe();
  const { data: health } = useHealth();
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex h-full">
      <aside className={cx("flex shrink-0 flex-col border-r border-slate-200 bg-white transition-all", collapsed ? "w-16" : "w-60")}>
        <div className="flex h-16 items-center justify-between px-4">
          <Logo compact={collapsed} />
          <button onClick={() => setCollapsed((c) => !c)} className="rounded p-1 text-slate-400 hover:bg-slate-100" title="Collapse sidebar">
            {collapsed ? <PanelLeftOpen className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
          </button>
        </div>
        <nav className="flex-1 space-y-0.5 px-3 py-2">
          {NAV.map(({ to, label, icon: Icon, end }) => {
            // Links with ?stage=… are only active for that exact filter.
            const active = to.includes("?")
              ? location.pathname + location.search === to
              : end
                ? location.pathname === to
                : location.pathname.startsWith(to);
            return (
              <Link
                key={to}
                to={to}
                title={label}
                className={cx(
                  "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition",
                  active ? "bg-brand-50 font-medium text-brand-700" : "text-slate-600 hover:bg-slate-50",
                )}
              >
                <Icon className="h-4 w-4 shrink-0" />
                {!collapsed && label}
              </Link>
            );
          })}
        </nav>
        <div className="border-t border-slate-100 px-3 py-2">
          <Link
            to="/settings"
            title="Settings"
            className={cx(
              "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition",
              location.pathname === "/settings" ? "bg-brand-50 font-medium text-brand-700" : "text-slate-600 hover:bg-slate-50",
            )}
          >
            <SettingsIcon className="h-4 w-4 shrink-0" />
            {!collapsed && "Settings"}
          </Link>
        </div>
        {user && !collapsed && (
          <div className="m-3 flex items-center gap-2.5 rounded-lg bg-slate-50 p-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-slate-800 text-xs font-semibold text-white">
              {initials(user.name)}
            </span>
            <div className="min-w-0 leading-tight">
              <div className="truncate text-sm font-semibold text-slate-800">{user.name}</div>
              <div className="truncate text-xs text-slate-500">{user.email}</div>
            </div>
          </div>
        )}
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-16 shrink-0 items-center justify-end gap-2 border-b border-slate-200 bg-white px-6">
          <NotificationBell />
          <UserMenu />
        </header>
        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-6xl px-6 py-8">
            {health && !health.llm_key_configured && (
              <div className="mb-6">
                <Notice>
                  No OpenRouter API key is configured, so the AI steps will fail. Add <code>OPENROUTER_API_KEY</code> to the{" "}
                  <code>.env</code> file in the project folder and restart the server.
                </Notice>
              </div>
            )}
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}

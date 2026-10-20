import { CheckCircle2, ClipboardList, FileCheck2, Plus, Trophy } from "lucide-react";
import type { ReactNode } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useMe, useProjects } from "../api/hooks";
import ProjectsTable from "../components/ProjectsTable";
import { Button, PageHeader, Spinner } from "../components/ui";

function Stat({ label, value, icon, hint }: { label: string; value: number | string; icon: ReactNode; hint: string }) {
  return (
    <div className="card p-5">
      <div className="flex items-center justify-between text-sm text-slate-500">
        {label}
        <span className="text-brand-500">{icon}</span>
      </div>
      <div className="mt-2 text-3xl font-semibold text-slate-800">{value}</div>
      <div className="mt-1 text-xs text-slate-400">{hint}</div>
    </div>
  );
}

function greeting() {
  const h = new Date().getHours();
  return h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";
}

export default function Home() {
  const { data: user } = useMe();
  const { data: projects, isLoading } = useProjects();
  const navigate = useNavigate();
  const all = projects ?? [];
  const waiting = all.filter((p) => !p.running && p.stage.endsWith("_review")).length;
  const done = all.filter((p) => p.stage === "done").length;
  const won = all.filter((p) => p.outcome === "won").length;

  return (
    <>
      <PageHeader
        title="Dashboard"
        subtitle={`${greeting()}, ${user?.name.split(" ")[0] ?? ""}. Here's what's happening with your RFP workflow.`}
        actions={<Button icon={<Plus className="h-4 w-4" />} onClick={() => navigate("/new")}>Start New Session</Button>}
      />
      <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Projects" value={all.length} icon={<ClipboardList className="h-4 w-4" />} hint="All RFPs you have started" />
        <Stat label="Awaiting your review" value={waiting} icon={<CheckCircle2 className="h-4 w-4" />} hint="Paused for a decision" />
        <Stat label="Proposals generated" value={done} icon={<FileCheck2 className="h-4 w-4" />} hint="Ready to submit" />
        <Stat label="Won" value={won} icon={<Trophy className="h-4 w-4" />} hint="Marked as won in History" />
      </div>
      <div className="card">
        <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
          <h2 className="font-semibold text-slate-800">Recent projects</h2>
          <Link to="/history" className="text-sm font-medium text-brand-600 hover:underline">View all</Link>
        </div>
        {isLoading ? <div className="flex justify-center p-10"><Spinner /></div> : <ProjectsTable projects={all.slice(0, 5)} />}
      </div>
    </>
  );
}

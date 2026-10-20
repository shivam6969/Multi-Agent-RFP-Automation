import { useState } from "react";

import { useProjects } from "../api/hooks";
import ProjectsTable from "../components/ProjectsTable";
import { Button, PageHeader, Spinner } from "../components/ui";

const PAGE = 10;

export default function History() {
  const { data: projects, isLoading } = useProjects();
  const [page, setPage] = useState(0);
  const all = projects ?? [];
  const pages = Math.max(1, Math.ceil(all.length / PAGE));
  const shown = all.slice(page * PAGE, page * PAGE + PAGE);

  return (
    <>
      <PageHeader title="Completed / History" subtitle="Every proposal you have worked on, with its outcome." />
      <div className="card">
        {isLoading ? (
          <div className="flex justify-center p-10"><Spinner /></div>
        ) : (
          <>
            <ProjectsTable projects={shown} showOutcome emptyText="No proposals yet." />
            {all.length > PAGE && (
              <div className="flex items-center justify-between border-t border-slate-100 px-4 py-3 text-sm text-slate-500">
                <span>
                  Showing {page * PAGE + 1}–{Math.min(all.length, (page + 1) * PAGE)} of {all.length}
                </span>
                <div className="flex gap-2">
                  <Button size="sm" variant="secondary" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</Button>
                  <Button size="sm" variant="secondary" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>Next</Button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}

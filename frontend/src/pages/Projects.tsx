import { useSearchParams } from "react-router-dom";

import { useProjects } from "../api/hooks";
import type { Stage } from "../api/types";
import ProjectsTable from "../components/ProjectsTable";
import { PageHeader, Spinner } from "../components/ui";

const TITLES: Partial<Record<Stage, [string, string]>> = {
  matching_review: ["Product Matching", "Projects waiting for you to review their product matches."],
  pricing_review: ["Pricing", "Projects waiting for you to review pricing and margin."],
  done: ["Generated Proposals", "Proposals that are ready to download and submit."],
};

export default function Projects() {
  const [params] = useSearchParams();
  const stage = params.get("stage") as Stage | null;
  const { data: projects, isLoading } = useProjects();
  const [title, subtitle] = (stage && TITLES[stage]) || ["Projects", "All projects."];
  const shown = (projects ?? []).filter((p) => !stage || (p.stage === stage && !p.running));

  return (
    <>
      <PageHeader title={title} subtitle={subtitle} />
      <div className="card">
        {isLoading ? (
          <div className="flex justify-center p-10"><Spinner /></div>
        ) : (
          <ProjectsTable projects={shown} emptyText="No projects at this step right now." />
        )}
      </div>
    </>
  );
}

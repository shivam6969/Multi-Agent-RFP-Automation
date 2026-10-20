import { RefreshCw } from "lucide-react";
import { useParams } from "react-router-dom";

import { useProject, useRetryProject } from "../api/hooks";
import type { Project } from "../api/types";
import Stepper, { stepForStage } from "../components/Stepper";
import { Button, ErrorBox, Spinner } from "../components/ui";
import MatchingReview from "./project/MatchingReview";
import PricingReview from "./project/PricingReview";
import RequirementsReview from "./project/RequirementsReview";
import Success from "./project/Success";

function Working({ project }: { project: Project }) {
  return (
    <div className="card mx-auto max-w-xl p-8 text-center">
      <Spinner className="mx-auto h-8 w-8" />
      <h2 className="mt-4 text-lg font-semibold text-slate-800">{project.progress ?? "Working…"}</h2>
      <p className="mt-1 text-sm text-slate-500">
        The AI is working on <b>{project.rfp.title}</b>. This usually takes one to two minutes; you can leave this page and come back.
      </p>
      <div className="progress-indeterminate mt-6" role="progressbar" aria-label="Working" />
    </div>
  );
}

function Failed({ project }: { project: Project }) {
  const retry = useRetryProject(project.id);
  return (
    <div className="mx-auto max-w-xl space-y-4">
      <ErrorBox title={`A step failed${project.failed_step ? ` (${project.failed_step})` : ""}`}>
        {project.error ?? "Unknown error"}
      </ErrorBox>
      <p className="text-sm text-slate-500">
        Your earlier reviews are saved. Fix the cause (for example, add a missing API key to <code>.env</code> and restart the server), then retry.
      </p>
      <Button icon={<RefreshCw className="h-4 w-4" />} loading={retry.isPending} onClick={() => retry.mutate()}>
        Retry this step
      </Button>
      {retry.error && <ErrorBox>{(retry.error as Error).message}</ErrorBox>}
    </div>
  );
}

export default function ProjectPage() {
  const { projectId } = useParams();
  const { data: project, isLoading, error } = useProject(projectId);

  if (isLoading) return <div className="flex justify-center p-16"><Spinner /></div>;
  if (error || !project) return <ErrorBox title="Could not load this project">{(error as Error)?.message ?? "Not found"}</ErrorBox>;

  let body;
  if (project.running || project.stage === "queued") body = <Working project={project} />;
  else if (project.stage === "failed") body = <Failed project={project} />;
  else if (project.stage === "requirements_review") body = <RequirementsReview key={`${project.id}-req`} project={project} />;
  else if (project.stage === "matching_review") body = <MatchingReview key={`${project.id}-match`} project={project} />;
  else if (project.stage === "pricing_review") body = <PricingReview key={`${project.id}-price`} project={project} />;
  else body = <Success project={project} />;

  return (
    <>
      <Stepper current={stepForStage(project.stage, project.progress)} />
      {body}
    </>
  );
}

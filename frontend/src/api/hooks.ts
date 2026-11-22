// React Query hooks for every backend call the screens use.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, ApiError, uploadPdf } from "./client";
import type {
  CatalogueProduct,
  CompanyProfile,
  Dashboard,
  DiscoverySearch,
  DiscoverySource,
  Health,
  LineItem,
  PricingRules,
  PricingSummary,
  Project,
  ProjectSummary,
  Rfp,
  StockOverview,
  User,
} from "./types";

// ─── Auth ─────────────────────────────────────────────────────────────────────

export function useMe() {
  return useQuery<User | null>({
    queryKey: ["me"],
    queryFn: async () => {
      try {
        return await api<User>("/api/auth/me");
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) return null;
        throw e;
      }
    },
    staleTime: 60_000,
  });
}

export function useLogin() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { email: string; password: string; remember: boolean }) =>
      api<User>("/api/auth/login", { method: "POST", json: body }),
    onSuccess: (user) => qc.setQueryData(["me"], user),
  });
}

export function useRegister() {
  return useMutation({
    mutationFn: (body: { email: string; name: string; password: string }) =>
      api<User>("/api/auth/register", { method: "POST", json: body }),
  });
}

export function useLogout() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api<void>("/api/auth/logout", { method: "POST" }),
    onSuccess: () => {
      qc.clear();
      qc.setQueryData(["me"], null);
    },
  });
}

export function useHealth() {
  return useQuery<Health>({ queryKey: ["health"], queryFn: () => api<Health>("/api/health") });
}

// ─── RFPs ─────────────────────────────────────────────────────────────────────

export function useRfps(q = "") {
  return useQuery<Rfp[]>({
    queryKey: ["rfps", q],
    queryFn: () => api<Rfp[]>(`/api/rfps${q ? `?q=${encodeURIComponent(q)}` : ""}`),
  });
}

export function useUploadRfp() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => uploadPdf<Rfp>("/api/rfps/upload", file),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rfps"] }),
  });
}

// ─── Discovery ────────────────────────────────────────────────────────────────

export function useDiscoverySources() {
  return useQuery<DiscoverySource[]>({
    queryKey: ["discovery-sources"],
    queryFn: () => api<DiscoverySource[]>("/api/discovery/sources"),
    staleTime: Infinity,
  });
}

export type StockKeyword = { keyword: string; products: number; units_on_hand: number; examples: string[] };

/** Fetched on demand (the "Use keywords from my stock" button). */
export function fetchStockKeywords() {
  return api<StockKeyword[]>("/api/discovery/stock-keywords?limit=6");
}

export function useDiscoverySearch() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: DiscoverySearch) => api<Rfp[]>("/api/discovery/search", { method: "POST", json: body }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rfps"] }),
  });
}

// ─── Projects ─────────────────────────────────────────────────────────────────

export function useProjects() {
  return useQuery<ProjectSummary[]>({
    queryKey: ["projects"],
    queryFn: () => api<ProjectSummary[]>("/api/projects?limit=200"),
    // Keep lists fresh while something is working in the background.
    refetchInterval: (query) => (query.state.data?.some((p) => p.running) ? 3000 : false),
  });
}

export function useDashboard() {
  return useQuery<Dashboard>({
    queryKey: ["dashboard"],
    queryFn: () => api<Dashboard>("/api/dashboard"),
    refetchInterval: 30_000,
  });
}

export function useProject(id: string | undefined) {
  return useQuery<Project>({
    queryKey: ["project", id],
    queryFn: () => api<Project>(`/api/projects/${id}`),
    enabled: Boolean(id),
    // Poll while the AI is working, stop once it pauses for review.
    refetchInterval: (query) => (query.state.data?.running ? 2000 : false),
  });
}

export function useStartProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (rfpId: string) => api<Project>("/api/projects", { method: "POST", json: { rfp_id: rfpId } }),
    onSuccess: (project) => {
      qc.setQueryData(["project", project.id], project);
      qc.invalidateQueries({ queryKey: ["projects"] });
    },
  });
}

export function useSubmitReview(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (decision: Record<string, unknown>) =>
      api<Project>(`/api/projects/${projectId}/review`, { method: "POST", json: { decision } }),
    onSuccess: (project) => {
      qc.setQueryData(["project", projectId], project);
      qc.invalidateQueries({ queryKey: ["projects"] });
    },
  });
}

export function useRetryProject(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api<Project>(`/api/projects/${projectId}/retry`, { method: "POST" }),
    onSuccess: (project) => qc.setQueryData(["project", projectId], project),
  });
}

export function useSetOutcome() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, outcome }: { id: string; outcome: string | null }) =>
      api<ProjectSummary>(`/api/projects/${id}/outcome`, { method: "PATCH", json: { outcome } }),
    onSuccess: (_, { id }) => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      qc.invalidateQueries({ queryKey: ["project", id] });
      qc.invalidateQueries({ queryKey: ["dashboard"] });
    },
  });
}

export interface PricingInputs {
  margin_pct?: number;
  advance_payment?: boolean;
  freight_inr?: number;
  quantities?: Record<string, number>;
}

export function pricingPreview(projectId: string, body: PricingInputs) {
  return api<{ line_items: LineItem[]; pricing_summary: PricingSummary }>(
    `/api/projects/${projectId}/pricing/preview`,
    { method: "POST", json: body },
  );
}

export function usePricingRules() {
  return useQuery<PricingRules>({
    queryKey: ["pricing-rules"],
    queryFn: () => api<PricingRules>("/api/pricing/rules"),
    staleTime: Infinity,
  });
}

// ─── Proposal editor ──────────────────────────────────────────────────────────

export type SectionEdit = { key: string; title?: string; body: string };

/** Render the PDF for the editor's current text; resolves to a blob URL. */
export async function proposalPreviewUrl(projectId: string, sections: SectionEdit[], signal?: AbortSignal) {
  const response = await fetch(`/api/projects/${projectId}/proposal/preview`, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sections }),
    signal,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(response.status, (body as { detail?: string })?.detail ?? "Preview failed");
  }
  return URL.createObjectURL(await response.blob());
}

export function useRegenerateSection(projectId: string) {
  return useMutation({
    mutationFn: ({ key, instruction, sections }: { key: string; instruction: string; sections: SectionEdit[] }) =>
      api<{ key: string; body: string; warning: string | null }>(
        `/api/projects/${projectId}/proposal/sections/${key}/regenerate`,
        { method: "POST", json: { instruction, sections } },
      ),
  });
}

export function useSaveProposalDraft(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sections: SectionEdit[]) =>
      api<Project>(`/api/projects/${projectId}/proposal/draft`, { method: "PUT", json: { sections } }),
    onSuccess: (project) => qc.setQueryData(["project", projectId], project),
  });
}

// ─── Settings ─────────────────────────────────────────────────────────────────

export function useCompanyProfile() {
  return useQuery<CompanyProfile>({ queryKey: ["company"], queryFn: () => api<CompanyProfile>("/api/settings/company") });
}

export function useSaveCompanyProfile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (profile: Omit<CompanyProfile, "is_demo">) =>
      api<CompanyProfile>("/api/settings/company", { method: "PUT", json: profile }),
    onSuccess: (profile) => qc.setQueryData(["company"], profile),
  });
}

// ─── Catalogue ────────────────────────────────────────────────────────────────

export function useCatalogueSearch(q: string) {
  return useQuery<CatalogueProduct[]>({
    queryKey: ["catalogue", q],
    queryFn: () => api<CatalogueProduct[]>(`/api/catalogue/products?q=${encodeURIComponent(q)}&limit=15`),
    enabled: q.trim().length >= 2,
    staleTime: 60_000,
  });
}

export function useStock() {
  return useQuery<StockOverview>({
    queryKey: ["stock"],
    queryFn: () => api<StockOverview>("/api/catalogue/stock"),
    staleTime: 60_000,
  });
}

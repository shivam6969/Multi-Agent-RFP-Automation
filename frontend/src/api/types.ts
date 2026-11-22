// Shapes returned by the FastAPI backend (see api/schemas.py and the agents).

export interface User {
  id: number;
  email: string;
  name: string;
}

export interface Rfp {
  id: string;
  title: string;
  source: "upload" | "sample" | "gem" | string;
  filename: string;
  pages: number;
  buyer: string | null;
  deadline: string | null;
  location: string | null;
  category: string | null;
  estimated_value: number | null;
  meta: { reference?: string; published?: string; synthetic?: boolean; total_quantity?: number; [k: string]: unknown };
  created_at: string;
}

export interface DiscoverySource {
  name: string;
  label: string;
  enabled: boolean;
  reason: string | null;
}

export interface DiscoverySearch {
  source: string;
  keywords: string[];
  count: number;
  location?: string;
  date_from?: string;
  date_to?: string;
  category?: string;
  min_value?: number;
  max_value?: number;
}

export type Stage =
  | "queued"
  | "running"
  | "requirements_review"
  | "matching_review"
  | "pricing_review"
  | "proposal_review"
  | "failed"
  | "done";

export interface ProposalSection {
  key: string;
  title: string;
  body: string;
  ai_generated: boolean;
  edited: boolean;
}

export interface ProposalMeta {
  reference?: string;
  generated?: string;
  valid_until?: string;
  buyer?: string | null;
  rfp_title?: string | null;
  rfp_reference?: string | null;
}

export interface CompanyProfile {
  name: string;
  tagline: string;
  about: string;
  brands: string;
  years_in_business: number | null;
  address: string;
  gstin: string;
  email: string;
  phone: string;
  website: string;
  signatory_name: string;
  signatory_title: string;
  is_demo: boolean;
}

export type Criticality = "mandatory" | "high" | "medium" | "low";

export interface Requirement {
  id: string;
  requirement: string;
  criticality: Criticality;
  weight: number;
  quantity: number | null;
}

export interface ProductBrief {
  product_id: string;
  product_name: string;
  product_range: string;
  product_variant: string | null;
  sap_code: string;
  unit_price: number;
  match_score: number | null;
}

export type ItemStatus =
  | "matched"
  | "needs_review"
  | "gap"
  | "approved"
  | "rejected"
  | "met"
  | "not_met";

export interface MatchedItem extends Partial<ProductBrief> {
  id: string;
  parent_id?: string;
  requirement: string;
  type: "product" | "compliance";
  status: ItemStatus;
  matched: boolean;
  weight: number;
  criticality?: Criticality;
  qty?: number;
  confidence?: "high" | "medium" | "low" | "manual" | null;
  rationale?: string;
  alternatives?: ProductBrief[];
  duplicate_of?: string;
  packing?: string;
  stock_on_hand?: number | null;
  monthly_available?: number | null;
  lead_time_days?: number | null;
  stock_is_demo?: boolean;
  stock_sufficient?: boolean | null;
}

export interface MatchSummary {
  fulfillment_score: number;
  product_items: number;
  matched: number;
  approved: number;
  needs_review: number;
  gaps: number;
  rejected: number;
  compliance_pending: number;
  compliance_met: number;
  compliance_not_met: number;
}

export type QtySource = "reviewer" | "rfp" | "requirement" | "query" | "default";

export interface LineItem {
  requirement: string;
  requirement_id: string;
  qty: number;
  qty_source: QtySource;
  currency: string;
  base_unit_price: number;
  margin_pct: number;
  volume_discount_pct: number;
  net_unit_price: number;
  gst_pct: number;
  gst_amount_per_unit: number;
  final_unit_price: number;
  line_total: number;
  product_id: string;
  product_name: string;
  product_range: string;
  sap_code: string;
  packing: string;
  monthly_available: number | null;
  stock_on_hand: number | null;
  stock_is_demo: boolean;
}

export interface PricingSummary {
  currency: string;
  margin_pct: number;
  subtotal_base: number;
  subtotal_ex_gst: number;
  advance_payment: boolean;
  payment_discount_pct: number;
  payment_discount_value: number;
  payment_discount_incl_gst: number;
  freight: number;
  freight_incl_gst: number;
  taxable_value: number;
  total_gst: number;
  grand_total: number;
  margin_value: number;
  items_count: number;
  qty_defaulted_count: number;
}

export interface ReviewLogEntry {
  stage: string;
  at: string;
  reviewer: string | null;
  note: string;
  changes: Record<string, unknown>;
}

export interface ProjectSummary {
  id: string;
  rfp: Rfp;
  stage: Stage;
  running: boolean;
  progress: string | null;
  error: string | null;
  grand_total: number | null;
  fulfillment_score: number | null;
  outcome: "submitted" | "won" | "lost" | null;
  proposal_available: boolean;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface Project extends ProjectSummary {
  failed_step: string | null;
  warnings: string[];
  rfp_summary: string | null;
  requirements: Requirement[];
  matched_items: MatchedItem[];
  match_summary: Partial<MatchSummary>;
  line_items: LineItem[];
  pricing_summary: Partial<PricingSummary>;
  unpriced: { requirement: string; requirement_id?: string; note?: string }[];
  review_log: ReviewLogEntry[];
  proposal_sections: ProposalSection[];
  proposal_meta: ProposalMeta;
  proposal_warnings: Record<string, string>;
}

export interface PricingRules {
  base_margin_pct: number;
  min_margin_pct: number;
  max_margin_pct: number;
  gst_rate_pct: number;
  currency: string;
  volume_discount_tiers: { min_qty: number; max_qty: number; discount_pct: number }[];
  payment_terms_discount_pct: number;
  freight_flat_inr: number;
  default_qty: number;
}

export interface CatalogueProduct {
  product_id: string;
  product_name: string;
  product_range: string;
  product_variant: string | null;
  sap_code: string;
  hsn_code: string;
  packing: string;
  unit_price: number;
  stock_on_hand: number | null;
  monthly_available: number | null;
  lead_time_days: number | null;
  stock_is_demo: boolean;
}

// GET /api/catalogue/stock (api/routes_catalogue.py)
export interface StockOverview {
  products: CatalogueProduct[];
  low_stock_units: number;
  is_demo: boolean;
}

export interface Health {
  status: string;
  llm_key_configured: boolean;
  gemini_key_configured: boolean;
}

// GET /api/dashboard (api/routes_dashboard.py)
export interface MonthCounts {
  value: number;
  this_month: number;
  last_month: number;
}

export type ReminderStatus = "overdue" | "due_today" | "failed" | "pending" | "ready" | "upcoming";

export interface Reminder {
  kind: "start" | "review" | "failed" | "submit" | "outcome";
  title: string;
  status: ReminderStatus;
  deadline: string | null;
  rfp_id: string;
  rfp_title: string;
  project_id: string | null;
}

export interface DeadlineEntry {
  date: string;
  rfp_id: string;
  title: string;
  buyer: string | null;
  project_id: string | null;
  stage: Stage | null;
  outcome: ProjectSummary["outcome"];
}

export interface PendingRfp {
  rfp_id: string;
  title: string;
  buyer: string | null;
  source: string;
  deadline: string | null;
  tag: "closing_soon" | "in_progress" | "new" | "open";
  project_id: string | null;
  stage: Stage | null;
}

export interface Dashboard {
  today: string;
  kpis: {
    proposals_generated: MonthCounts;
    awaiting_review: { value: number; urgent: number };
    rfps_received: MonthCounts;
    win_rate: { value: number | null; won: number; lost: number; awaiting_outcome: number; won_value: number };
  };
  reminders: Reminder[];
  reminders_total: number;
  deadlines: DeadlineEntry[];
  activity: { month: string; generated: number; won: number }[];
  pending_rfps: PendingRfp[];
  pending_total: number;
}

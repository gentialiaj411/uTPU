import type { TierDef, TierKey } from "./tiers";

export interface Claim {
  id: string;
  category: string;
  headline: string;
  value: string | number | boolean | null;
  unit: string;
  tier: TierKey;
  source_artifact: string;
  fence_note: string;
  tags?: string[];
  raw?: unknown;
}

export interface EvidenceBundle {
  meta: {
    generated_at_utc: string;
    artifact_count: number;
    claim_count: number;
    artifacts: string[];
    github_repo: string;
    silicon_tier_note: string;
  };
  claims: Claim[];
  tiers: TierDef[];
}

export interface PipelineNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: {
    label: string;
    stage_id: string;
    op_count: number;
    op_kinds: string[];
    detail: Record<string, unknown>;
    target?: string | null;
  };
}

export interface PipelineEdge {
  id: string;
  source: string;
  target: string;
}

export interface VizNode {
  id: string;
  label: string;
  op: string;
  x: number;
  y: number;
}

export interface VizEdge {
  from: string;
  to: string;
}

export interface WalkthroughFrame {
  id: string;
  stage: string;
  title: string;
  caption: string;
  effect: string;
  graph: { nodes: VizNode[]; edges: VizEdge[] };
  pass_delta?: {
    removed: string[];
    added: string[];
    before_kinds: string[];
    after_kinds: string[];
  };
  backends?: {
    cuda?: BackendCard[];
    utpu?: BackendCard[];
  };
}

export interface BackendCard {
  graph_op: string;
  op: string;
  target: string;
  kernel_name?: string;
  mode?: string;
  program_instruction_words?: number;
  fits_instruction_bram?: boolean;
}

export interface Walkthrough {
  replay_note: string;
  frame_count: number;
  frames: WalkthroughFrame[];
}

export interface Pipeline {
  id: string;
  name: string;
  array_size: number;
  example_input_shape?: number[];
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  walkthrough?: Walkthrough;
  coverage: {
    cuda_backend_ops: number;
    utpu_native_ops: string[];
    asymmetry_note: string;
  };
  summary: Record<string, unknown>;
}

export interface SystolicCase {
  batch_size: number;
  pe_occupancy: number | null;
  rtl_busy_counter: number | null;
  rtl_cycle_counter: number | null;
  busy_fraction: number | null;
  model_per_tile_busy: number | null;
  streaming_ceiling: number | null;
}

export interface SystolicBundle {
  tier: TierKey;
  source_artifact?: string;
  array_size?: number;
  status?: string;
  streaming_ceiling_formula?: string;
  cases: SystolicCase[];
}

export interface VerifyStep {
  id: string;
  label: string;
  tier: TierKey;
  source_artifact: string | null;
  status: string | boolean | null;
  detail: string;
  raw?: unknown;
}

export interface VerifyLadderBundle {
  steps: VerifyStep[];
}

const BASE = import.meta.env.BASE_URL;

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}data/${path}`);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.json() as Promise<T>;
}

export function loadEvidence(): Promise<EvidenceBundle> {
  return fetchJson<EvidenceBundle>("evidence.json");
}

export function loadPipelines(): Promise<{ pipelines: Pipeline[] }> {
  return fetchJson<{ pipelines: Pipeline[] }>("pipelines.json");
}

export function loadSystolic(): Promise<SystolicBundle> {
  return fetchJson<SystolicBundle>("systolic.json");
}

export function loadVerifyLadder(): Promise<VerifyLadderBundle> {
  return fetchJson<VerifyLadderBundle>("verify_ladder.json");
}

export function formatClaimValue(value: Claim["value"], unit: string): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    const rounded = Number.isInteger(value) ? String(value) : value.toFixed(2);
    return unit ? `${rounded} ${unit}` : rounded;
  }
  return unit ? `${value} ${unit}` : String(value);
}

export function groupClaimsByCategory(claims: Claim[]): Map<string, Claim[]> {
  const map = new Map<string, Claim[]>();
  for (const claim of claims) {
    const list = map.get(claim.category) ?? [];
    list.push(claim);
    map.set(claim.category, list);
  }
  return map;
}

/** Public-mode headline panels: one anchor claim per tier ladder. */
export function curatedHeadlineClaims(claims: Claim[]): Claim[] {
  const picks = [
    "megakernel_launch_reduction_pooled",
    "resnet18_eager_parity",
    "scheduler_rtl_reduction_permille",
  ];
  const byId = new Map(claims.map((c) => [c.id, c]));
  const headlines: Claim[] = [];
  for (const id of picks) {
    const c = byId.get(id);
    if (c) headlines.push(c);
  }
  if (headlines.length < 3) {
    for (const c of claims) {
      if (headlines.length >= 3) break;
      if (!headlines.includes(c)) headlines.push(c);
    }
  }
  return headlines.slice(0, 3);
}

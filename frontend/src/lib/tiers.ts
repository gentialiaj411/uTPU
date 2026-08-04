export type TierKey = "sim" | "ci" | "synth" | "silicon";

export interface TierDef {
  key: TierKey;
  label: string;
  color: string;
  meaning: string;
}

export const TIER_STYLES: Record<
  TierKey,
  { bg: string; border: string; text: string; empty?: boolean }
> = {
  sim: { bg: "#fef3c7", border: "#f59e0b", text: "#92400e" },
  ci: { bg: "#dbeafe", border: "#3b82f6", text: "#1e40af" },
  synth: { bg: "#ede9fe", border: "#8b5cf6", text: "#5b21b6" },
  silicon: { bg: "#f3f4f6", border: "#9ca3af", text: "#6b7280", empty: true },
};

export function formatTierLabel(tier: TierDef): string {
  return tier.label;
}

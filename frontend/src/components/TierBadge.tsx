import type { TierKey } from "../lib/tiers";
import { TIER_STYLES } from "../lib/tiers";

interface Props {
  tier: TierKey;
  label?: string;
  compact?: boolean;
}

export function TierBadge({ tier, label, compact }: Props) {
  const style = TIER_STYLES[tier];
  const text = label ?? tier.toUpperCase();
  return (
    <span
      className="tier-badge"
      style={{
        background: style.bg,
        borderColor: style.border,
        color: style.text,
        fontSize: compact ? "0.7rem" : "0.75rem",
        opacity: style.empty ? 0.85 : 1,
      }}
      title={style.empty ? "P0 open — no on-board numbers yet" : undefined}
    >
      {text}
    </span>
  );
}

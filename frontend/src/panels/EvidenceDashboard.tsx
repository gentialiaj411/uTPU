import { useMemo } from "react";
import type { Claim, EvidenceBundle } from "../lib/data";
import { curatedHeadlineClaims, groupClaimsByCategory } from "../lib/data";
import type { AppMode } from "../lib/mode";
import { isDevMode } from "../lib/mode";
import type { TierDef } from "../lib/tiers";
import { ClaimCard } from "../components/ClaimCard";
import { TierBadge } from "../components/TierBadge";

interface Props {
  bundle: EvidenceBundle;
  mode: AppMode;
  onInspectRaw?: (claim: Claim) => void;
}

export function EvidenceDashboard({ bundle, mode, onInspectRaw }: Props) {
  const grouped = useMemo(() => groupClaimsByCategory(bundle.claims), [bundle.claims]);
  const headlines = useMemo(() => curatedHeadlineClaims(bundle.claims), [bundle.claims]);
  const dev = isDevMode(mode);

  return (
    <section className="panel">
      <header className="panel-header">
        <h2>Evidence Dashboard</h2>
        <p className="honesty-statement">
          Every number below is fenced to a committed artifact. Nothing is live CUDA/FPGA execution —
          this UI replays captured bench results only.
        </p>
      </header>

      <TierLegend tiers={bundle.tiers} siliconNote={bundle.meta.silicon_tier_note} />

      {!dev && (
        <div className="headline-row">
          <h3>Headline panels</h3>
          <div className="headline-grid">
            {headlines.map((claim) => (
              <ClaimCard
                key={claim.id}
                claim={claim}
                mode={mode}
                githubRepo={bundle.meta.github_repo}
                onInspectRaw={onInspectRaw}
              />
            ))}
          </div>
        </div>
      )}

      {dev && (
        <p className="dev-banner">
          Dev mode — showing all {bundle.claims.length} claims from {bundle.meta.artifact_count}{" "}
          artifacts.
        </p>
      )}

      {[...grouped.entries()].map(([category, claims]) => (
        <div key={category} className="category-block">
          <h3>{category}</h3>
          <div className="claim-grid">
            {claims.map((claim) => (
              <ClaimCard
                key={claim.id}
                claim={claim}
                mode={mode}
                githubRepo={bundle.meta.github_repo}
                onInspectRaw={onInspectRaw}
              />
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}

function TierLegend({ tiers, siliconNote }: { tiers: TierDef[]; siliconNote: string }) {
  return (
    <div className="tier-legend">
      {tiers.map((t) => (
        <div key={t.key} className="tier-legend-item">
          <TierBadge tier={t.key} label={t.label} />
          <span>{t.meaning}</span>
        </div>
      ))}
      <p className="silicon-gap">{siliconNote}</p>
    </div>
  );
}

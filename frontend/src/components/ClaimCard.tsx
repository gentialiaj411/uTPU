import type { Claim } from "../lib/data";
import { formatClaimValue } from "../lib/data";
import type { AppMode } from "../lib/mode";
import { ArtifactLink } from "./ArtifactLink";
import { TierBadge } from "./TierBadge";

interface Props {
  claim: Claim;
  mode: AppMode;
  githubRepo?: string;
  onInspectRaw?: (claim: Claim) => void;
}

export function ClaimCard({ claim, mode, githubRepo, onInspectRaw }: Props) {
  const unstable = claim.tags?.includes("unstable_latency");
  const synthFailed = claim.tags?.includes("synth_failed");

  return (
    <article className={`claim-card ${synthFailed ? "claim-failed" : ""}`}>
      <header className="claim-header">
        <TierBadge tier={claim.tier} />
        <span className="claim-category">{claim.category}</span>
      </header>
      <h3 className="claim-headline">{claim.headline}</h3>
      <p className={`claim-value ${unstable ? "claim-unstable" : ""}`}>
        {formatClaimValue(claim.value, claim.unit)}
        {unstable && <span className="tag-unstable">unstable latency</span>}
        {synthFailed && <span className="tag-failed">synth failed</span>}
      </p>
      <p className="claim-fence">{claim.fence_note}</p>
      <footer>
        <ArtifactLink
          sourceArtifact={claim.source_artifact}
          mode={mode}
          githubRepo={githubRepo}
          raw={claim.raw}
          onInspectRaw={onInspectRaw ? () => onInspectRaw(claim) : undefined}
        />
      </footer>
    </article>
  );
}

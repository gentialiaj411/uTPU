import type { VerifyLadderBundle } from "../lib/data";
import type { TierKey } from "../lib/tiers";
import { TIER_STYLES } from "../lib/tiers";
import { TierBadge } from "../components/TierBadge";

interface Props {
  ladder: VerifyLadderBundle;
  githubRepo?: string;
}

function statusLabel(status: string | boolean | null): string {
  if (status === true) return "pass";
  if (status === false) return "fail";
  if (status === null || status === undefined) return "—";
  return String(status);
}

export function VerifyLadder({ ladder, githubRepo = "gentialiaj411/uTPU" }: Props) {
  return (
    <section className="panel">
      <header className="panel-header">
        <h2>Verify Ladder</h2>
        <p>Bit-exact ISA-vs-RTL evidence chain toward silicon (P0 open at the top).</p>
      </header>

      <ol className="verify-ladder">
        {ladder.steps.map((step, idx) => {
          const tier = step.tier as TierKey;
          const style = TIER_STYLES[tier];
          const isOpen = step.status === "open";
          const isPass =
            step.status === true ||
            step.status === "pass" ||
            step.status === "PASS" ||
            step.status === "ok" ||
            step.status === "RTL-verified";

          return (
            <li
              key={step.id}
              className={`verify-step ${isOpen ? "verify-open" : isPass ? "verify-pass" : "verify-other"}`}
              style={{ borderLeftColor: style.border }}
            >
              <div className="verify-step-header">
                <span className="verify-index">{idx + 1}</span>
                <TierBadge tier={tier} compact />
                <strong>{step.label}</strong>
                <span className={`verify-status ${isOpen ? "status-open" : ""}`}>
                  {statusLabel(step.status)}
                </span>
              </div>
              <p>{step.detail}</p>
              {step.source_artifact && (
                <a
                  href={`https://github.com/${githubRepo}/blob/main/${step.source_artifact}`}
                  target="_blank"
                  rel="noreferrer"
                  className="artifact-link"
                >
                  {step.source_artifact}
                </a>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import type { Claim, EvidenceBundle } from "./lib/data";
import {
  loadEvidence,
  loadPipelines,
  loadSystolic,
  loadVerifyLadder,
} from "./lib/data";
import type { AppMode } from "./lib/mode";
import { resolveMode } from "./lib/mode";
import { RawArtifactPanel } from "./components/ArtifactLink";
import { EvidenceDashboard } from "./panels/EvidenceDashboard";
import { PipelineExplorer } from "./panels/PipelineExplorer";
import { SystolicArray } from "./panels/SystolicArray";
import { VerifyLadder } from "./panels/VerifyLadder";

function App() {
  const [mode] = useState<AppMode>(() => resolveMode());
  const [evidence, setEvidence] = useState<EvidenceBundle | null>(null);
  const [pipelines, setPipelines] = useState<Awaited<ReturnType<typeof loadPipelines>> | null>(null);
  const [systolic, setSystolic] = useState<Awaited<ReturnType<typeof loadSystolic>> | null>(null);
  const [ladder, setLadder] = useState<Awaited<ReturnType<typeof loadVerifyLadder>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rawInspect, setRawInspect] = useState<Claim | null>(null);
  const [tab, setTab] = useState<"compiler" | "dashboard" | "systolic" | "verify">("compiler");

  useEffect(() => {
    Promise.all([loadEvidence(), loadPipelines(), loadSystolic(), loadVerifyLadder()])
      .then(([e, p, s, l]) => {
        setEvidence(e);
        setPipelines(p);
        setSystolic(s);
        setLadder(l);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  if (error) {
    return (
      <main className="app error">
        <h1>uTPU Evidence Explorer</h1>
        <p>Failed to load data: {error}</p>
        <p>Run <code>python tools/build_frontend_data.py</code> first.</p>
      </main>
    );
  }

  if (!evidence || !pipelines || !systolic || !ladder) {
    return <main className="app loading">Loading evidence bundle…</main>;
  }

  return (
    <main className="app">
      <header className="app-header">
        <div>
          <h1>uTPU Evidence Explorer</h1>
          <p className="subtitle">
            Scoped ML-compiler + FPGA lab — replay only, no fabricated metrics
          </p>
        </div>
        <div className="mode-pill">
          Mode: <strong>{mode}</strong>
          {mode === "public" && (
            <span className="mode-hint-inline"> · add ?mode=dev for introspection</span>
          )}
        </div>
      </header>

      <nav className="tabs">
        {(
          [
            ["compiler", "Compiler"],
            ["dashboard", "Evidence"],
            ["systolic", "Systolic"],
            ["verify", "Verify Ladder"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            className={tab === id ? "tab active" : "tab"}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </nav>

      {tab === "compiler" && (
        <PipelineExplorer pipelines={pipelines.pipelines} mode={mode} />
      )}
      {tab === "dashboard" && (
        <EvidenceDashboard
          bundle={evidence}
          mode={mode}
          onInspectRaw={mode === "dev" ? setRawInspect : undefined}
        />
      )}
      {tab === "systolic" && <SystolicArray data={systolic} />}
      {tab === "verify" && (
        <VerifyLadder ladder={ladder} githubRepo={evidence.meta.github_repo} />
      )}

      {rawInspect && (
        <div className="modal-backdrop" onClick={() => setRawInspect(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <header>
              <h3>{rawInspect.headline}</h3>
              <button type="button" onClick={() => setRawInspect(null)}>Close</button>
            </header>
            <RawArtifactPanel raw={rawInspect.raw} path={rawInspect.source_artifact} />
          </div>
        </div>
      )}

      <footer className="app-footer">
        Generated {evidence.meta.generated_at_utc} · {evidence.meta.claim_count} fenced claims ·{" "}
        {evidence.meta.artifact_count} artifacts
      </footer>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

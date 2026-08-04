# uTPU Evidence Explorer

React + Vite frontend that surfaces **only** metrics from committed `bench/results/*.json` artifacts. No hardcoded performance numbers, no simulated live runs.

## Data flow

```text
bench/results/*.json
        │
        ▼
tools/build_frontend_data.py   ← single source of truth
        │
        ├── evidence.json      (claims[] with tier + source_artifact)
        ├── pipelines.json     (compiler DAG from live compile_model replay)
        ├── systolic.json      (occupancy curve from systolic_characterization)
        └── verify_ladder.json (ISA→RTL→synth chain; silicon step empty)
        │
        ▼
frontend/public/data/
        │
        ▼
npm run dev / vite build → Evidence Explorer UI
```

Regenerate data before dev or build:

```bash
python tools/build_frontend_data.py
cd frontend && npm install && npm run dev
```

Schema lock (CI-gatable):

```bash
python -m pytest firmware/host/test_frontend_data_schema.py -q
```

The build **fails** if any claim lacks `tier ∈ {sim, ci, synth, silicon}` or a `source_artifact` path that exists on disk.

## Tier system

| Tier | Color | Meaning |
|------|-------|---------|
| `sim` | amber | iverilog / ISA-sim / cost-model replay |
| `ci` | blue | Host artifacts exercised in `.github/workflows/ci.yml` |
| `synth` | purple | Vivado P&R, timing-closed (not executed on board) |
| `silicon` | gray (empty) | On-board capture — **P0 open** |

Every metric card shows tier badge + fence note + link to source JSON.

Latency/gap-% claims from `megakernel_payoff.json` / `cublas_baseline.json` are tagged `unstable_latency` — prefer launch-count stats.

`packed_dsp_synth.json` surfaces `synth_failed` honestly; never shown as success.

## Modes (one codebase)

| Mode | How | UI |
|------|-----|-----|
| **public** (default) | no query param | 3 headline claim panels + tabbed explorers; artifact links → GitHub blob |
| **dev** | `?mode=dev` or `VITE_DEFAULT_MODE=dev` | all claims, raw JSON modal, IR node inspector on pipeline graph |

Same components; dev unlocks introspection, never forks layout.

## Panels

1. **Evidence Dashboard** — claim cards grouped by category
2. **Pipeline Explorer** — reactflow DAG (TinyVisualMLP); CUDA vs uTPU coverage asymmetry
3. **Systolic Array** — framer-motion PE grid replaying RTL counter artifact (sim tier)
4. **Verify Ladder** — evidence chain ending in open silicon step

## Rules (non-negotiable)

- UI renders **only** values from the generated JSON bundles
- No fake live CUDA/FPGA execution; any replay is labeled
- Silicon tier stays empty until on-board artifacts land in `bench/results/`

## Deploy

`.github/workflows/pages.yml` runs aggregator → `vite build` (base `/uTPU/`) → GitHub Pages. Workflow is provided but not enabled until you push and configure Pages.

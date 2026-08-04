import { motion } from "framer-motion";
import type { SystolicBundle } from "../lib/data";
import { TierBadge } from "../components/TierBadge";

interface Props {
  data: SystolicBundle;
}

const DEFAULT_ARRAY = 16;

export function SystolicArray({ data }: Props) {
  if (data.status === "missing_artifact" || data.cases.length === 0) {
    return (
      <section className="panel">
        <h2>Systolic Array</h2>
        <p>
          No systolic_characterization.json in bench/results on this branch — occupancy replay unavailable.
        </p>
        <TierBadge tier="sim" label="Simulated / ISA-sim" />
      </section>
    );
  }

  const gridSize = data.array_size ?? DEFAULT_ARRAY;
  const maxBatch = Math.max(...data.cases.map((c) => c.batch_size));
  const selected = data.cases.find((c) => c.batch_size === maxBatch) ?? data.cases[data.cases.length - 1];
  const occupancy = selected.pe_occupancy ?? 0;
  const ceiling = selected.streaming_ceiling ?? 0;
  const busy = selected.rtl_busy_counter ?? 0;
  const cycles = selected.rtl_cycle_counter ?? 1;

  return (
    <section className="panel">
      <header className="panel-header">
        <h2>Systolic Array</h2>
        <TierBadge tier="sim" label="Simulated — RTL perf counters" />
        <p>{data.streaming_ceiling_formula}</p>
        <p className="artifact-ref">Source: {data.source_artifact}</p>
      </header>

      <div className="systolic-layout">
        <svg viewBox={`0 0 ${gridSize * 44} ${gridSize * 44}`} className="systolic-grid">
          {Array.from({ length: gridSize * gridSize }, (_, i) => {
            const row = Math.floor(i / gridSize);
            const col = i % gridSize;
            const active = i / (gridSize * gridSize) < occupancy;
            return (
              <motion.rect
                key={i}
                x={col * 44 + 4}
                y={row * 44 + 4}
                width={36}
                height={36}
                rx={4}
                fill={active ? "#f59e0b" : "#e5e7eb"}
                initial={{ opacity: 0.3 }}
                animate={{ opacity: active ? 1 : 0.35 }}
                transition={{
                  duration: 0.4,
                  delay: (row + col) * 0.02,
                  repeat: Infinity,
                  repeatType: "reverse",
                  repeatDelay: 0.5,
                }}
              />
            );
          })}
        </svg>

        <div className="systolic-stats">
          <h3>
            {gridSize}×{gridSize} flagship · B={selected.batch_size}
          </h3>
          <dl>
            <dt>PE occupancy</dt>
            <dd>{(occupancy * 100).toFixed(2)}%</dd>
            <dt>RTL busy cycles</dt>
            <dd>{busy}</dd>
            <dt>RTL total cycles</dt>
            <dd>{cycles}</dd>
            <dt>Streaming ceiling (model)</dt>
            <dd>{(ceiling * 100).toFixed(2)}%</dd>
            <dt>Model per-tile busy</dt>
            <dd>{selected.model_per_tile_busy ?? "—"}</dd>
          </dl>

          <div className="batch-sweep">
            <h4>Batch sweep</h4>
            <table>
              <thead>
                <tr>
                  <th>B</th>
                  <th>Occupancy</th>
                  <th>Busy</th>
                </tr>
              </thead>
              <tbody>
                {data.cases.map((c) => (
                  <tr key={c.batch_size}>
                    <td>{c.batch_size}</td>
                    <td>{c.pe_occupancy != null ? (c.pe_occupancy * 100).toFixed(1) + "%" : "—"}</td>
                    <td>{c.rtl_busy_counter ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <p className="replay-note">
        Replay of committed RTL-sim counters — not live FPGA execution.
      </p>
    </section>
  );
}

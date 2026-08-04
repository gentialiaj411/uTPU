import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useMemo, useState, type CSSProperties } from "react";
import type { Pipeline, VizEdge, VizNode, WalkthroughFrame } from "../lib/data";

const STAGE_COLORS: Record<string, string> = {
  pytorch: "#64748b",
  fx: "#0ea5e9",
  ir: "#6366f1",
  pass: "#f59e0b",
  cuda: "#3b82f6",
  utpu: "#8b5cf6",
};

const OP_COLORS: Record<string, string> = {
  input: "#94a3b8",
  output: "#94a3b8",
  linear_relu: "#7c3aed",
  linear: "#2563eb",
  relu: "#ea580c",
  Linear: "#2563eb",
  ReLU: "#ea580c",
};

function nodeColor(op: string, stage: string): string {
  if (stage === "cuda") return "#dbeafe";
  if (stage === "utpu") return "#ede9fe";
  for (const [key, color] of Object.entries(OP_COLORS)) {
    if (op.includes(key)) return color;
  }
  return "#e2e8f0";
}

function edgePath(from: VizNode, to: VizNode): string {
  const x1 = from.x + 70;
  const y1 = from.y + 28;
  const x2 = to.x;
  const y2 = to.y + 28;
  const mx = (x1 + x2) / 2;
  return `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
}

interface Props {
  pipeline: Pipeline;
}

export function CompilerWalkthrough({ pipeline }: Props) {
  const frames = pipeline.walkthrough?.frames ?? [];
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);

  const frame = frames[idx] ?? null;
  const stageColor = frame ? STAGE_COLORS[frame.stage] ?? "#64748b" : "#64748b";

  const next = useCallback(() => {
    setIdx((i) => Math.min(i + 1, frames.length - 1));
  }, [frames.length]);

  const prev = useCallback(() => {
    setIdx((i) => Math.max(i - 1, 0));
  }, []);

  useEffect(() => {
    if (!playing) return;
    if (idx >= frames.length - 1) {
      setPlaying(false);
      return;
    }
    const t = window.setTimeout(next, 2200);
    return () => window.clearTimeout(t);
  }, [playing, idx, frames.length, next]);

  const nodeMap = useMemo(() => {
    const m = new Map<string, VizNode>();
    for (const n of frame?.graph.nodes ?? []) m.set(n.id, n);
    return m;
  }, [frame]);

  if (!frames.length) {
    return (
      <p className="walkthrough-empty">
        Walkthrough unavailable — rebuild with PyTorch: <code>python tools/build_frontend_data.py</code>
      </p>
    );
  }

  return (
    <div className="compiler-walkthrough">
      <div className="walkthrough-toolbar">
        <div className="walkthrough-controls">
          <button type="button" onClick={() => setIdx(0)} disabled={idx === 0} title="Reset">
            ⏮
          </button>
          <button type="button" onClick={prev} disabled={idx === 0} title="Previous">
            ◀
          </button>
          <button
            type="button"
            className={playing ? "play active" : "play"}
            onClick={() => setPlaying((p) => !p)}
            title={playing ? "Pause" : "Play"}
          >
            {playing ? "⏸" : "▶"}
          </button>
          <button type="button" onClick={next} disabled={idx >= frames.length - 1} title="Next">
            ▶
          </button>
        </div>
        <div className="walkthrough-progress">
          {frames.map((f, i) => (
            <button
              key={f.id}
              type="button"
              className={`progress-dot ${i === idx ? "active" : ""} ${i < idx ? "done" : ""}`}
              style={{ "--stage-color": STAGE_COLORS[f.stage] } as CSSProperties}
              onClick={() => setIdx(i)}
              title={f.title}
            />
          ))}
        </div>
        <span className="frame-counter">
          {idx + 1} / {frames.length}
        </span>
      </div>

      <p className="replay-banner">{pipeline.walkthrough?.replay_note}</p>

      <AnimatePresence mode="wait">
        {frame && (
          <motion.div
            key={frame.id}
            className="walkthrough-stage"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.35 }}
          >
            <header className="stage-header" style={{ borderColor: stageColor }}>
              <span className="stage-pill" style={{ background: stageColor }}>
                {frame.stage}
              </span>
              <h3>{frame.title}</h3>
            </header>
            <p className="stage-caption">{frame.caption}</p>

            {frame.pass_delta && frame.effect === "fuse" && (
              <motion.div
                className="fusion-banner"
                initial={{ scale: 0.95, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
              >
                <span className="fuse-from">linear + relu</span>
                <span className="fuse-arrow">→</span>
                <span className="fuse-to">linear_relu</span>
                <span className="fuse-note">producer-consumer fusion</span>
              </motion.div>
            )}

            <div className="graph-canvas-wrap">
              <GraphCanvas
                nodes={frame.graph.nodes}
                edges={frame.graph.edges}
                nodeMap={nodeMap}
                stage={frame.stage}
                effect={frame.effect}
                playing={playing}
              />
            </div>

            {frame.backends && (
              <div className="backend-split">
                {frame.backends.cuda && (
                  <BackendPanel title="CUDA / NVRTC" tint="#3b82f6" cards={frame.backends.cuda} />
                )}
                {frame.backends.utpu && (
                  <BackendPanel title="uTPU ISA" tint="#8b5cf6" cards={frame.backends.utpu} />
                )}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function GraphCanvas({
  nodes,
  edges,
  nodeMap,
  stage,
  effect,
  playing,
}: {
  nodes: VizNode[];
  edges: VizEdge[];
  nodeMap: Map<string, VizNode>;
  stage: string;
  effect: string;
  playing: boolean;
}) {
  const width = Math.max(720, ...nodes.map((n) => n.x + 160));
  const height = Math.max(220, ...nodes.map((n) => n.y + 80));

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="graph-canvas">
      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8" />
        </marker>
        <filter id="glow">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {edges.map((e, i) => {
        const from = nodeMap.get(e.from);
        const to = nodeMap.get(e.to);
        if (!from || !to) return null;
        return (
          <g key={`${e.from}-${e.to}`}>
            <path
              d={edgePath(from, to)}
              fill="none"
              stroke="#cbd5e1"
              strokeWidth={2}
              markerEnd="url(#arrow)"
            />
            {playing && (
              <motion.circle
                r={5}
                fill={STAGE_COLORS[stage] ?? "#6366f1"}
                initial={{ offsetDistance: "0%" }}
                animate={{ offsetDistance: "100%" }}
                transition={{
                  duration: 1.2,
                  delay: i * 0.15,
                  repeat: Infinity,
                  ease: "linear",
                }}
                style={{
                  offsetPath: `path('${edgePath(from, to)}')`,
                }}
              />
            )}
          </g>
        );
      })}

      {nodes.map((n, i) => (
        <motion.g
          key={n.id}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: i * 0.08, type: "spring", stiffness: 260, damping: 20 }}
        >
          <rect
            x={n.x}
            y={n.y}
            width={140}
            height={56}
            rx={10}
            fill={nodeColor(n.op, stage)}
            stroke={effect === "fuse" && n.op.includes("linear_relu") ? "#7c3aed" : "#94a3b8"}
            strokeWidth={effect === "fuse" && n.op.includes("linear_relu") ? 3 : 1.5}
            filter={stage === "cuda" || stage === "utpu" ? "url(#glow)" : undefined}
          />
          <text x={n.x + 12} y={n.y + 22} className="node-label">
            {n.label}
          </text>
          <text x={n.x + 12} y={n.y + 42} className="node-op">
            {n.op}
          </text>
        </motion.g>
      ))}
    </svg>
  );
}

function BackendPanel({
  title,
  tint,
  cards,
}: {
  title: string;
  tint: string;
  cards: NonNullable<WalkthroughFrame["backends"]>["cuda"];
}) {
  if (!cards?.length) return null;
  return (
    <div className="backend-panel" style={{ borderColor: tint }}>
      <h4 style={{ color: tint }}>{title}</h4>
      {cards.map((c) => (
        <div key={c.graph_op} className="backend-card">
          <strong>{c.graph_op}</strong>
          <span className="backend-op">{c.op}</span>
          {c.kernel_name && <span>kernel: {c.kernel_name}</span>}
          {c.program_instruction_words != null && (
            <span>{c.program_instruction_words} ISA words</span>
          )}
          {c.fits_instruction_bram != null && (
            <span>fits BRAM: {String(c.fits_instruction_bram)}</span>
          )}
        </div>
      ))}
    </div>
  );
}

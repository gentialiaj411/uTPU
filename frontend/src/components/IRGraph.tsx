import { useMemo } from "react";
import type { PipelineNode } from "../lib/data";

interface Props {
  node: PipelineNode | null;
}

export function IRGraph({ node }: Props) {
  const detail = node?.data.detail;
  const json = useMemo(() => JSON.stringify(detail ?? {}, null, 2), [detail]);

  if (!node) {
    return <p className="ir-empty">Select a pipeline stage to inspect IR lowering details.</p>;
  }

  return (
    <div className="ir-inspector">
      <h3>{node.data.label}</h3>
      <p className="ir-meta">
        {node.data.op_count} ops · kinds: {node.data.op_kinds.join(", ") || "—"}
        {node.data.target ? ` · target=${node.data.target}` : ""}
      </p>
      <pre className="raw-panel">
        <code>{json}</code>
      </pre>
    </div>
  );
}

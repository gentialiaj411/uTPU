import { useCallback, useMemo, useState } from "react";

import {

  Background,

  Controls,

  MiniMap,

  ReactFlow,

  type Node,

  type Edge,

  type NodeMouseHandler,

} from "@xyflow/react";

import "@xyflow/react/dist/style.css";

import type { Pipeline } from "../lib/data";

import type { AppMode } from "../lib/mode";

import { isDevMode } from "../lib/mode";

import { IRGraph } from "../components/IRGraph";



interface Props {

  pipeline: Pipeline;

  mode: AppMode;

}



function stageNodeLabel(data: Pipeline["nodes"][0]["data"]): string {

  return `${data.label}\n(${data.op_count} ops)`;

}



export function PipelineDagView({ pipeline, mode }: Props) {

  const [selectedId, setSelectedId] = useState<string | null>(null);

  const dev = isDevMode(mode);



  const nodes: Node[] = useMemo(

    () =>

      pipeline.nodes.map((n) => ({

        id: n.id,

        position: n.position,

        data: { label: stageNodeLabel(n.data) },

        style: {

          background:

            n.data.stage_id === "utpu"

              ? "#ede9fe"

              : n.data.stage_id === "cuda"

                ? "#dbeafe"

                : "#f8fafc",

          border: "1px solid #cbd5e1",

          borderRadius: 8,

          padding: 12,

          fontSize: 12,

          whiteSpace: "pre-wrap",

          width: 200,

        },

      })),

    [pipeline],

  );



  const edges: Edge[] = useMemo(

    () =>

      pipeline.edges.map((e) => ({

        id: e.id,

        source: e.source,

        target: e.target,

        animated: true,

      })),

    [pipeline],

  );



  const selectedNode = pipeline.nodes.find((n) => n.id === selectedId) ?? null;



  const onNodeClick: NodeMouseHandler = useCallback(

    (_evt, node) => {

      if (dev) setSelectedId(node.id);

    },

    [dev],

  );



  return (

    <div className="pipeline-layout">

      <div className="flow-wrap flow-wrap-compact">

        <ReactFlow

          nodes={nodes}

          edges={edges}

          fitView

          onNodeClick={onNodeClick}

          nodesDraggable={dev}

          proOptions={{ hideAttribution: true }}

        >

          <Background />

          <Controls />

          <MiniMap />

        </ReactFlow>

      </div>

      {dev && <IRGraph node={selectedNode} />}

    </div>

  );

}


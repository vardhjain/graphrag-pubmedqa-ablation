"use client";

import { useMemo } from "react";
import ReactFlow, {
  Background,
  Controls,
  type Edge,
  type Node,
  Position,
} from "reactflow";
import "reactflow/dist/style.css";
import type { ReasoningStep } from "@/lib/types";

const KIND_STYLE: Record<ReasoningStep["kind"], { bg: string; border: string }> = {
  seed_chunk: { bg: "#eff6ff", border: "#3b82f6" }, // retrieved chunk
  parent_paper: { bg: "#f0fdf4", border: "#22c55e" }, // expanded via HAS_CONTEXT
  concept_neighbour: { bg: "#fdf4ff", border: "#a855f7" }, // MeSH concept hop
};

const KIND_LABEL: Record<ReasoningStep["kind"], string> = {
  seed_chunk: "Retrieved chunk",
  parent_paper: "Parent paper (HAS_CONTEXT)",
  concept_neighbour: "Concept neighbour (MENTIONS)",
};

function buildGraph(steps: ReasoningStep[]): { nodes: Node[]; edges: Edge[] } {
  const seedChunks = steps.filter((s) => s.kind === "seed_chunk");
  const parentPapers = steps.filter((s) => s.kind === "parent_paper");
  const conceptNeighbours = steps.filter((s) => s.kind === "concept_neighbour");

  const nodes: Node[] = [];
  const edges: Edge[] = [];
  const seen = new Set<string>();

  const addNode = (step: ReasoningStep, x: number, y: number) => {
    if (seen.has(step.node_id)) return;
    seen.add(step.node_id);
    const style = KIND_STYLE[step.kind];
    nodes.push({
      id: step.node_id,
      position: { x, y },
      data: { label: step.label },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
      style: {
        background: style.bg,
        border: `1.5px solid ${style.border}`,
        borderRadius: 8,
        padding: 8,
        fontSize: 12,
        width: 180,
      },
    });
  };

  seedChunks.forEach((s, i) => addNode(s, i * 220, 0));
  parentPapers.forEach((s, i) => addNode(s, i * 220, 160));
  conceptNeighbours.forEach((s, i) => addNode(s, i * 220 + 60, 320));

  for (const step of parentPapers) {
    if (step.from_node) {
      edges.push({
        id: `${step.from_node}->${step.node_id}`,
        source: step.from_node,
        target: step.node_id,
        label: step.edge ?? undefined,
        animated: true,
        style: { stroke: KIND_STYLE.parent_paper.border },
      });
    }
  }

  // Concept-hop origin isn't tracked per-neighbour by the backend (the graph
  // query ranks across all seed papers together) -- draw from every parent
  // paper so the fan-out is visible without overclaiming a single source.
  for (const neighbour of conceptNeighbours) {
    for (const parent of parentPapers) {
      edges.push({
        id: `${parent.node_id}->${neighbour.node_id}`,
        source: parent.node_id,
        target: neighbour.node_id,
        label: neighbour.edge ?? undefined,
        style: { stroke: KIND_STYLE.concept_neighbour.border, strokeDasharray: "4 3" },
      });
    }
  }

  return { nodes, edges };
}

export default function ReasoningGraph({ steps }: { steps: ReasoningStep[] }) {
  const { nodes, edges } = useMemo(() => buildGraph(steps), [steps]);

  if (steps.length === 0) {
    return (
      <p className="text-sm text-gray-500 italic p-4">
        No reasoning path to show for this answer.
      </p>
    );
  }

  return (
    <div className="w-full">
      <div className="h-72 w-full rounded-lg border border-gray-200">
        <ReactFlow nodes={nodes} edges={edges} fitView proOptions={{ hideAttribution: true }}>
          <Background />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
      <div className="flex gap-4 mt-2 text-xs text-gray-600">
        {(Object.keys(KIND_STYLE) as ReasoningStep["kind"][]).map((kind) => (
          <span key={kind} className="flex items-center gap-1">
            <span
              className="inline-block w-2.5 h-2.5 rounded-sm"
              style={{ background: KIND_STYLE[kind].bg, border: `1.5px solid ${KIND_STYLE[kind].border}` }}
            />
            {KIND_LABEL[kind]}
          </span>
        ))}
      </div>
    </div>
  );
}

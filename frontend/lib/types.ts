export type ReasoningStepKind = "seed_chunk" | "parent_paper" | "concept_neighbour";

export interface ReasoningStep {
  kind: ReasoningStepKind;
  node_id: string;
  label: string;
  from_node: string | null;
  edge: string | null;
}

export interface QueryResponse {
  answer: string;
  reasoning_path: ReasoningStep[];
  sources: string[];
}

export interface QueryRequest {
  question: string;
  graph_id?: string;
  use_concepts?: boolean;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  reasoningPath?: ReasoningStep[];
  error?: boolean;
  latencyMs?: number;
}

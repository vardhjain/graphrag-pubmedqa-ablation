import fs from "fs";
import path from "path";

export interface ArmResult {
  arm: string;
  accuracy: number;
  // [lo, hi] as percentages, e.g. [52.6, 66.1] -- Wilson 95% CI, the
  // sampling uncertainty a bare accuracy percentage never discloses.
  // Optional: absent in summary.json files generated before this existed.
  accuracy_ci95?: [number, number];
  macro_f1: number;
  // Fraction of (non-failed) questions where the gold paper was among the
  // ones _select() retrieved, pre any graph expansion -- isolates retrieval
  // quality from LLM synthesis. null/absent for arms/files predating this.
  recall_at_k?: number | null;
  avg_latency: number;
  samples: number;
  // Count of questions where the LLM call exhausted retries (recorded as a
  // placeholder "maybe", not a genuine judgment). Absent in older files.
  n_failed?: number;
  adds: string;
}

export interface Contrast {
  from: string;
  to: string;
  effect: string;
  delta_acc: number;
  gains: number;
  losses: number;
  p_value: number;
  // Pre-formatted by scripts/compare.py's format_p() (e.g. "<0.0001") --
  // use this instead of re-deriving a display string from p_value in two
  // places that could drift. Optional for pre-existing summary.json files.
  p_display?: string;
  // Holm-Bonferroni-adjusted p-value/display for the same contrast, correcting
  // for testing multiple contrasts from one benchmark run. "significant"
  // below already reflects this adjustment, not the raw p_value's threshold.
  p_holm?: number;
  p_holm_display?: string;
  significant: boolean;
}

export interface BenchmarkSummary {
  n: number;
  seed: number;
  model: string;
  dataset: string;
  arms: ArmResult[];
  contrasts: Contrast[];
}

// Reads the repo's canonical results/summary.json directly -- no duplicated
// numbers to drift out of sync with RESULTS.md. Runs server-side only
// (fs is unavailable in the browser); this page is statically generated at
// build time, so the file is read once per build, not per request.
//
// Vercel note: if this project's Root Directory is set to `frontend`, enable
// "Include source files outside of the Root Directory in the Build Step" in
// the Vercel project settings, or this import will fail to find the file.
export function loadBenchmarkSummary(): BenchmarkSummary {
  const resultsPath = path.join(process.cwd(), "..", "results", "summary.json");
  const raw = fs.readFileSync(resultsPath, "utf-8");
  return JSON.parse(raw) as BenchmarkSummary;
}

// ---- Derived display helpers (shared by /, /benchmark, /case-study) ----
// Every headline number on the site is derived from summary.json here, so no
// page carries a hand-typed literal that can drift from the committed results.

export function fmtP(p: number): string {
  return p < 0.0001 ? "<0.0001" : p.toFixed(4);
}

// "p<0.0001" / "p=0.6889" -- the operator depends on whether the display
// string already starts with a comparison.
export function pWithOperator(c: Contrast): string {
  const s = c.p_display ?? fmtP(c.p_value);
  return /^[<>=]/.test(s) ? `p${s}` : `p=${s}`;
}

export function fmtDelta(delta: number): string {
  return `${delta > 0 ? "+" : delta < 0 ? "−" : ""}${Math.abs(delta).toFixed(1)}pp`;
}

// Wilson score 95% interval, as percentages. Derivable from accuracy (%) and
// sample count alone, so it is computed client-side when summary.json predates
// compare.py emitting accuracy_ci95 -- same method, same label.
export function wilsonCI95(accuracyPct: number, n: number): [number, number] | undefined {
  if (!n || n <= 0) return undefined;
  const z = 1.959963984540054;
  const p = accuracyPct / 100;
  const denom = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
  return [Math.max(0, centre - half) * 100, Math.min(1, centre + half) * 100];
}

export function armCI(arm: ArmResult): [number, number] | undefined {
  return arm.accuracy_ci95 ?? wilsonCI95(arm.accuracy, arm.samples);
}

export function findContrast(s: BenchmarkSummary, from: string, to: string): Contrast | undefined {
  return s.contrasts.find((c) => c.from === from && c.to === to);
}

export interface Headline {
  armCount: number;
  parentExpansion?: Contrast;
  conceptHop?: Contrast;
  // graph_concepts avg latency / graph avg latency, e.g. 5.4
  conceptLatencyRatio?: number;
}

export function deriveHeadline(s: BenchmarkSummary): Headline {
  const graph = s.arms.find((a) => a.arm === "graph");
  const concepts = s.arms.find((a) => a.arm === "graph_concepts");
  return {
    armCount: s.arms.length,
    parentExpansion: findContrast(s, "plain_rr", "graph"),
    conceptHop: findContrast(s, "graph", "graph_concepts"),
    conceptLatencyRatio:
      graph && concepts && graph.avg_latency > 0
        ? concepts.avg_latency / graph.avg_latency
        : undefined,
  };
}

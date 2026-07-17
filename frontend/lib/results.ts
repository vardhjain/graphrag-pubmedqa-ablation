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

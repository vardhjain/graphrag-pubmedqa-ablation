import fs from "fs";
import path from "path";

export interface ArmResult {
  arm: string;
  accuracy: number;
  macro_f1: number;
  avg_latency: number;
  samples: number;
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

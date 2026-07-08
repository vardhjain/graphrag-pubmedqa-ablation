import Link from "next/link";
import { loadBenchmarkSummary } from "@/lib/results";

export const metadata = {
  title: "Benchmark -- PubMed GraphRAG assistant",
};

function fmtPct(n: number) {
  return `${n.toFixed(1)}%`;
}

function fmtP(p: number) {
  return p < 0.0001 ? "<0.0001" : p.toFixed(4);
}

export default function BenchmarkPage() {
  const summary = loadBenchmarkSummary();
  const winner = summary.arms.length
    ? summary.arms.reduce((a, b) => (b.accuracy > a.accuracy ? b : a))
    : null;

  return (
    <div className="min-h-screen bg-zinc-50">
      <header className="border-b border-gray-200 bg-white px-6 py-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-between">
            <h1 className="text-lg font-semibold text-zinc-900">Benchmark</h1>
            <Link href="/" className="text-sm text-blue-600 hover:underline">
              &larr; Back to chat
            </Link>
          </div>
          <p className="text-sm text-zinc-500 mt-1">
            4-arm ablation on {summary.dataset}, n={summary.n}, seed={summary.seed}, model{" "}
            <code className="text-xs bg-gray-100 px-1 rounded">{summary.model}</code>. Numbers
            pulled directly from{" "}
            <a
              href="https://github.com/vardhjain/graphrag-pubmedqa-ablation/blob/main/RESULTS.md"
              target="_blank"
              rel="noreferrer"
              className="text-blue-600 hover:underline"
            >
              RESULTS.md
            </a>
            .
          </p>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8 space-y-10">
        <section>
          <h2 className="text-sm font-semibold text-zinc-700 uppercase tracking-wide mb-3">
            Per-arm results
          </h2>
          <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 text-left text-zinc-500">
                  <th className="px-4 py-2 font-medium">Arm</th>
                  <th className="px-4 py-2 font-medium">Accuracy</th>
                  <th className="px-4 py-2 font-medium">Macro F1</th>
                  <th className="px-4 py-2 font-medium">Avg latency</th>
                  <th className="px-4 py-2 font-medium">Adds</th>
                </tr>
              </thead>
              <tbody>
                {summary.arms.map((arm) => (
                  <tr
                    key={arm.arm}
                    className={`border-b border-gray-100 last:border-0 ${
                      arm.arm === winner?.arm ? "bg-green-50" : ""
                    }`}
                  >
                    <td className="px-4 py-2 font-mono text-xs">
                      {arm.arm}
                      {arm.arm === winner?.arm && (
                        <span className="ml-2 inline-block rounded-full bg-green-600 text-white text-[10px] px-2 py-0.5 align-middle">
                          winner
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-2 font-medium">{fmtPct(arm.accuracy)}</td>
                    <td className="px-4 py-2">{fmtPct(arm.macro_f1)}</td>
                    <td className="px-4 py-2">{arm.avg_latency.toFixed(1)}s</td>
                    <td className="px-4 py-2 text-zinc-600">{arm.adds}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section>
          <h2 className="text-sm font-semibold text-zinc-700 uppercase tracking-wide mb-3">
            Significance (paired McNemar)
          </h2>
          <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 text-left text-zinc-500">
                  <th className="px-4 py-2 font-medium">Contrast</th>
                  <th className="px-4 py-2 font-medium">&Delta;acc</th>
                  <th className="px-4 py-2 font-medium">Gains / losses</th>
                  <th className="px-4 py-2 font-medium">p-value</th>
                  <th className="px-4 py-2 font-medium">Significant?</th>
                </tr>
              </thead>
              <tbody>
                {summary.contrasts.map((c) => (
                  <tr key={`${c.from}-${c.to}`} className="border-b border-gray-100 last:border-0">
                    <td className="px-4 py-2">
                      <span className="font-mono text-xs">{c.from}</span> &rarr;{" "}
                      <span className="font-mono text-xs">{c.to}</span>
                      <span className="text-zinc-500"> ({c.effect})</span>
                    </td>
                    <td className={`px-4 py-2 font-medium ${c.delta_acc > 0 ? "text-green-700" : "text-red-700"}`}>
                      {c.delta_acc > 0 ? "+" : ""}
                      {c.delta_acc.toFixed(1)}pp
                    </td>
                    <td className="px-4 py-2 text-zinc-600">
                      {c.gains} / {c.losses}
                    </td>
                    <td className="px-4 py-2 font-mono text-xs">{fmtP(c.p_value)}</td>
                    <td className="px-4 py-2">
                      {c.significant ? (
                        <span className="text-green-700 font-medium">yes</span>
                      ) : (
                        <span className="text-zinc-400">no</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="text-sm text-zinc-600 leading-relaxed">
          <h2 className="text-sm font-semibold text-zinc-700 uppercase tracking-wide mb-3">
            Honest summary
          </h2>
          <p>
            The graph arm&apos;s win is one specific mechanism, not &quot;a knowledge graph&quot; in
            the abstract: <strong>parent-document expansion</strong>{" "}
            lifts accuracy +22.5 percentage points over the reranked baseline, confirmed by a
            paired McNemar test (p&lt;0.0001) -- not noise. The further MeSH concept-hop arm does not clear
            significance against the plain graph arm (p=0.69) and costs ~5x the latency, so it is
            not the recommended arm to ship. See{" "}
            <a
              href="https://github.com/vardhjain/graphrag-pubmedqa-ablation/blob/main/RESULTS.md"
              target="_blank"
              rel="noreferrer"
              className="text-blue-600 hover:underline"
            >
              RESULTS.md
            </a>{" "}
            for the full write-up.
          </p>
        </section>
      </main>
    </div>
  );
}

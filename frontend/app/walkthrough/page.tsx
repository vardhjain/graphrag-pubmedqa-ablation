import Link from "next/link";
import type { Metadata } from "next";
import RetrievalWalkthrough from "@/components/RetrievalWalkthrough";
import { deriveHeadline, fmtDelta, loadBenchmarkSummary, pWithOperator } from "@/lib/results";

export const metadata: Metadata = {
  title: "Project walkthrough",
  description:
    "A three-minute, no-backend walkthrough of GraphRAG retrieval, the PubMedQA ablation, code, and study limitations.",
};

const REPO = "https://github.com/vardhjain/graphrag-pubmedqa-ablation";

const codePaths = [
  {
    title: "Shared retrieval pipeline",
    path: "src/kgqa/retrieval/base.py",
    description: "Embeddings, reranking, and candidate selection are reused across arms.",
  },
  {
    title: "ArangoDB graph expansion",
    path: "src/kgqa/retrieval/graph.py",
    description: "Actual AQL traversal from a retrieved chunk to its parent abstract.",
  },
  {
    title: "Benchmark runner",
    path: "scripts/run_benchmark.py",
    description: "Sampling, checkpoints, per-question evaluation, and run metadata.",
  },
  {
    title: "Statistics and comparisons",
    path: "scripts/compare.py",
    description: "Paired McNemar tests, confidence intervals, and generated result tables.",
  },
];

export default function WalkthroughPage() {
  const summary = loadBenchmarkSummary();
  const { parentExpansion, conceptHop } = deriveHeadline(summary);

  return (
    <div className="min-h-screen bg-zinc-50">
      <header className="border-b border-zinc-200 bg-white px-5 py-4">
        <div className="mx-auto max-w-4xl flex flex-wrap items-center justify-between gap-3">
          <Link href="/" className="text-sm text-blue-700 hover:underline">
            &larr; Live chat
          </Link>
          <nav className="flex flex-wrap gap-4 text-sm" aria-label="Project pages">
            <Link href="/case-study" className="text-blue-700 hover:underline">
              Case study
            </Link>
            <Link href="/benchmark" className="text-blue-700 hover:underline">
              Benchmark
            </Link>
            <a href={REPO} target="_blank" rel="noreferrer" className="text-blue-700 hover:underline">
              GitHub
            </a>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-4xl px-5 py-10 space-y-12">
        <section className="space-y-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-blue-700">
            Three-minute technical walkthrough · No backend required
          </p>
          <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-zinc-900">
            What does the graph actually add to RAG?
          </h1>
          <p className="text-base leading-relaxed text-zinc-700 max-w-3xl">
            This project studies one precise question: when a retrieval pipeline already
            finds a relevant fragment of a biomedical paper, does following a graph
            link to the <strong>complete abstract</strong> improve the answer?
            Follow the example below, then inspect the measured evidence and source.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link href="#interactive-example" className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500">
              Explore the retrieval path
            </Link>
            <Link href="/benchmark" className="rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-medium text-zinc-800 hover:border-zinc-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500">
              See actual results
            </Link>
          </div>
        </section>

        <section id="interactive-example" className="scroll-mt-6 space-y-4">
          <h2 className="text-xl font-semibold text-zinc-900">1. Watch parent expansion work</h2>
          <p className="text-sm leading-relaxed text-zinc-700">
            Click through a synthetic retrieval example. It runs entirely in your browser,
            even when the free-tier chat backend is sleeping.
          </p>
          <RetrievalWalkthrough />
        </section>

        <section className="space-y-4">
          <h2 className="text-xl font-semibold text-zinc-900">2. Inspect the historical experiment</h2>
          <p className="text-sm leading-relaxed text-zinc-700">
            The published study compared {summary.arms.length} retrieval setups on {summary.n} seeded
            PubMedQA questions. Each step in the main ladder adds a component:
            plain RAG → reranking → full parent abstract → related-paper concept hop.
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-xl border border-zinc-200 bg-white p-5">
              <p className="text-xs uppercase tracking-wide text-zinc-500">Parent expansion</p>
              <p className="mt-2 text-3xl font-semibold text-zinc-900">
                {parentExpansion ? fmtDelta(parentExpansion.delta_acc) : "Not reported"}
              </p>
              <p className="mt-2 text-sm text-zinc-600">
                Historical accuracy change vs reranked chunk RAG
                {parentExpansion ? ` · McNemar ${pWithOperator(parentExpansion)}` : ""}
              </p>
            </div>
            <div className="rounded-xl border border-zinc-200 bg-white p-5">
              <p className="text-xs uppercase tracking-wide text-zinc-500">Related-paper expansion</p>
              <p className="mt-2 text-3xl font-semibold text-zinc-900">
                {conceptHop ? fmtDelta(conceptHop.delta_acc) : "Not reported"}
              </p>
              <p className="mt-2 text-sm text-zinc-600">
                Historical accuracy change vs parent expansion alone
                {conceptHop ? ` · McNemar ${pWithOperator(conceptHop)}` : ""}
              </p>
            </div>
          </div>
          <p className="text-sm text-zinc-700">
            This is a comparison of retrieval configurations, <strong>not proof that
            graph edges themselves are better than a non-graph parent-document lookup</strong>.
            An equally budgeted flat parent-document baseline is a planned follow-up.
          </p>
          <Link href="/benchmark" className="inline-block text-sm font-medium text-blue-700 hover:underline">
            Full metrics, sample size, and tests &rarr;
          </Link>
        </section>

        <section className="space-y-4">
          <h2 className="text-xl font-semibold text-zinc-900">3. Review the engineering decisions</h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {codePaths.map((item) => (
              <a
                key={item.path}
                href={`${REPO}/blob/main/${item.path}`}
                target="_blank"
                rel="noreferrer"
                className="rounded-xl border border-zinc-200 bg-white p-5 hover:border-blue-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
              >
                <h3 className="font-semibold text-zinc-900">{item.title}</h3>
                <p className="mt-1 font-mono text-xs text-blue-700 break-all">{item.path}</p>
                <p className="mt-3 text-sm leading-relaxed text-zinc-600">{item.description}</p>
              </a>
            ))}
          </div>
          <p className="text-sm text-zinc-700">
            The live product also includes a FastAPI REST/MCP backend, a graph-path
            visualization, graceful fallbacks, and a memory-optimized ONNX encoder.
            The <Link href="/case-study" className="text-blue-700 hover:underline">case study</Link>{" "}
            explains the deployment tradeoffs.
          </p>
        </section>

        <section className="space-y-4 rounded-2xl border border-zinc-200 bg-white p-6">
          <h2 className="text-xl font-semibold text-zinc-900">4. Understand the limits before the demo</h2>
          <ul className="list-disc pl-5 space-y-2 text-sm leading-relaxed text-zinc-700">
            <li>
              <strong>Different runtime:</strong> the historical benchmark used a large
              ArangoDB corpus and local Ollama model. The hosted demo uses only the
              1,000-paper labeled split in Neo4j, a cloud LLM, and no reranker.
              Its output is <strong>not</strong> a reproduction of the numbers above.
            </li>
            <li>
              <strong>Preliminary evidence:</strong> the historical graph may have
              contained duplicate context edges. The benchmark needs a clean rerun,
              and raw per-question outputs were not committed.
            </li>
            <li>
              <strong>Modest absolute accuracy:</strong> the historical majority-yes
              baseline scored 56.5%; the best reported graph score is only narrowly above it.
            </li>
            <li>
              <strong>Hosted availability:</strong> the free-tier backend can sleep
              and take minutes to wake up. This walkthrough and the benchmark page
              still work without it.
            </li>
          </ul>
          <div className="flex flex-wrap gap-4 text-sm">
            <a href={`${REPO}/blob/main/RESULTS.md`} target="_blank" rel="noreferrer" className="text-blue-700 hover:underline">
              Research limitations &rarr;
            </a>
            <a href={`${REPO}/blob/main/GAPS.md`} target="_blank" rel="noreferrer" className="text-blue-700 hover:underline">
              Open issues &rarr;
            </a>
          </div>
        </section>

        <section className="space-y-3 pb-8">
          <h2 className="text-xl font-semibold text-zinc-900">Try the real system</h2>
          <p className="text-sm text-zinc-700">
            The <Link href="/" className="text-blue-700 hover:underline">chat demo</Link> retrieves
            from the hosted knowledge graph and links to source PubMed papers. For
            a concise explanation of the architecture, evidence, and remaining
            work, see the{" "}
            <a href={`${REPO}/blob/main/REVIEWER_GUIDE.md`} target="_blank" rel="noreferrer" className="text-blue-700 hover:underline">
              reviewer guide
            </a>.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link href="/" className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700">
              Open live chat &rarr;
            </Link>
            <Link href="/case-study" className="rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-medium text-zinc-800 hover:border-zinc-400">
              Read engineering case study
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
}

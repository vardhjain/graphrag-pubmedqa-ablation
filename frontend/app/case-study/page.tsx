import Link from "next/link";
import { loadBenchmarkSummary } from "@/lib/results";

export const metadata = {
  title: "Case study -- PubMed GraphRAG assistant",
  description:
    "How a confounded GraphRAG-vs-RAG comparison became a fair 4-arm ablation, plus the engineering behind hosting it as a live demo.",
};

const REPO = "https://github.com/vardhjain/graphrag-pubmedqa-ablation";

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <h2 className="text-sm font-semibold text-zinc-700 uppercase tracking-wide mb-3">
        {title}
      </h2>
      <div className="text-sm text-zinc-700 leading-relaxed space-y-3">{children}</div>
    </section>
  );
}

export default function CaseStudyPage() {
  const summary = loadBenchmarkSummary();
  const graph = summary.arms.find((a) => a.arm === "graph");
  const plainRr = summary.arms.find((a) => a.arm === "plain_rr");
  const parentExpansion = summary.contrasts.find(
    (c) => c.from === "plain_rr" && c.to === "graph"
  );

  return (
    <div className="min-h-screen bg-zinc-50">
      <header className="border-b border-gray-200 bg-white px-6 py-4">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-center justify-between">
            <h1 className="text-lg font-semibold text-zinc-900">Case study</h1>
            <Link href="/" className="text-sm text-blue-600 hover:underline">
              &larr; Back to chat
            </Link>
          </div>
          <p className="text-sm text-zinc-500 mt-1">
            Does a knowledge graph actually help retrieval-augmented QA, once you stop
            cheating? A fair ablation, and the engineering behind turning it into a
            live agent.
          </p>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-6 py-8 space-y-10">
        <Section title="The question">
          <p>
            An earlier version of this project claimed GraphRAG beat plain RAG on
            PubMedQA. It didn&apos;t hold up: the GraphRAG arm secretly also had a
            reranker the baseline lacked, pulled from a different corpus, and worst of
            all, leaked the paper title (and with it, most of the answer) directly into
            the LLM&apos;s prompt. The &quot;win&quot; was three confounds stacked on
            top of each other, not a knowledge graph doing anything.
          </p>
          <p>
            This repo is the fixed version, built to answer the question honestly: with
            everything else held constant, what does a knowledge graph actually add?
          </p>
        </Section>

        <Section title="The method">
          <p>
            A controlled 4-arm ablation on{" "}
            <a
              href="https://pubmedqa.github.io/"
              target="_blank"
              rel="noreferrer"
              className="text-blue-600 hover:underline"
            >
              PubMedQA
            </a>
            , each arm adding exactly one component on top of the last:
          </p>
          <div className="rounded-lg border border-gray-200 bg-white px-4 py-3 font-mono text-xs text-zinc-600 overflow-x-auto">
            plain &rarr; plain_rr &rarr; graph &rarr; graph_concepts
            <div className="text-zinc-400 mt-1">
              (vector top-k) &rarr; (+ cross-encoder reranker) &rarr; (+ parent-document
              expansion) &rarr; (+ MeSH concept hop)
            </div>
          </div>
          <p>
            Corpus, chunking, embedder, reranker, prompt, LLM, random seed, sample size,
            and top-k are identical across all four arms &mdash; centralized in one
            config module every arm imports, so nothing can silently diverge. Graph
            context uses generic <code className="text-xs bg-gray-100 px-1 rounded">=== STUDY n ===</code>{" "}
            labels with no titles and no gold label anywhere near the prompt; a test
            asserts the question string never appears in retrieved context, closing off
            the exact leak that broke the original comparison. Significance between
            adjacent arms is a paired McNemar test on the same questions, not a raw
            accuracy comparison.
          </p>
        </Section>

        <Section title="The result">
          <p>
            The win is one specific, isolated mechanism &mdash;{" "}
            <strong>parent-document expansion</strong>: once a chunk is retrieved,
            walk back to its parent paper and hand the LLM the full reconstructed
            abstract instead of an isolated fragment.
          </p>
          <div className="rounded-lg border border-gray-200 bg-white px-4 py-4 space-y-1">
            {graph && plainRr && (
              <p className="text-2xl font-semibold text-zinc-900">
                +{((graph.accuracy - plainRr.accuracy)).toFixed(1)}pp accuracy
              </p>
            )}
            <p className="text-zinc-500">
              over the reranked baseline
              {parentExpansion && (
                <>
                  , McNemar{" "}
                  <span className="font-mono">
                    p{parentExpansion.p_value < 0.0001 ? "<0.0001" : `=${parentExpansion.p_value}`}
                  </span>
                </>
              )}
            </p>
          </div>
          <p>
            The further MeSH concept-hop arm doesn&apos;t clear significance against
            the plain graph arm and costs roughly 5x the latency &mdash; so it&apos;s
            not the arm this project ships. See{" "}
            <a
              href={`${REPO}/blob/main/RESULTS.md`}
              target="_blank"
              rel="noreferrer"
              className="text-blue-600 hover:underline"
            >
              RESULTS.md
            </a>{" "}
            for the full breakdown, or the{" "}
            <Link href="/benchmark" className="text-blue-600 hover:underline">
              live benchmark page
            </Link>{" "}
            for the numbers behind this page.
          </p>
        </Section>

        <Section title="Turning it into a live agent">
          <p>
            The benchmark pipeline runs on ArangoDB and a local Ollama model, which
            produced every number above but can&apos;t be hosted for free. Rather than
            rewrite that pipeline (which would require re-running the GPU benchmark to
            re-validate the published numbers), the winning <code className="text-xs bg-gray-100 px-1 rounded">graph</code>{" "}
            retrieval logic was extracted behind a stable{" "}
            <code className="text-xs bg-gray-100 px-1 rounded">answer(question, graph_id)</code>{" "}
            service boundary and given a second, parallel backend: a small Neo4j
            AuraDB graph (the 1,000-paper labeled split, not the full 62k-paper
            benchmark corpus) plus a cloud LLM provider chain (Gemini, falling back to
            local Ollama) so the demo needs no GPU. FastAPI on Render serves it; this
            Next.js app and a reasoning-path graph visualization are the frontend.
          </p>
          <p>
            Three real bugs surfaced only once this was live, not in tests:
          </p>
          <ul className="list-disc pl-5 space-y-2">
            <li>
              <strong>Free-tier OOM.</strong> Loading both the sentence encoder and the
              cross-encoder reranker (two full torch models) killed the 512MB Render
              instance. Fix: skip the reranker on this tier &mdash; every retriever
              already degrades to raw top-k without one by design, so this was a config
              flag, not a code change.
            </li>
            <li>
              <strong>A lazy-loader race condition.</strong> Concurrent first requests
              could each start building the shared encoder/graph connection
              independently. Fixed with a lock around the check-then-create path.
            </li>
            <li>
              <strong>A wedged service from a corrupted cache.</strong> An instance
              restart mid-write left a truncated vector cache on disk; every subsequent
              query then hit the same <code className="text-xs bg-gray-100 px-1 rounded">EOFError</code>{" "}
              until the disk was reset. Fixed by treating an unreadable cache as a miss
              (rebuild, don&apos;t crash) and writing the cache atomically (temp file +
              rename) so a mid-write kill can never leave a partial file again &mdash;
              caught and fixed by firing a real test question at the deployed backend,
              not by unit tests alone.
            </li>
          </ul>
        </Section>

        <Section title="Stack">
          <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white">
            <table className="w-full text-sm">
              <tbody>
                {[
                  ["Benchmark graph DB", "ArangoDB -- every RESULTS.md number comes from here"],
                  ["Benchmark LLM", "deepseek-r1:8b via local Ollama, temperature=0"],
                  ["Hosted-demo graph DB", "Neo4j AuraDB Free -- labeled-split-only demo graph"],
                  ["Hosted-demo LLM", "Gemini Flash, falling back to Ollama"],
                  ["Backend", "FastAPI on Render (free tier)"],
                  ["Frontend", "Next.js 16 + Tailwind + reactflow, on Vercel"],
                  ["Dashboard", "Streamlit, reading committed results/ JSON directly"],
                ].map(([k, v]) => (
                  <tr key={k} className="border-b border-gray-100 last:border-0">
                    <td className="px-4 py-2 font-medium text-zinc-900 whitespace-nowrap">{k}</td>
                    <td className="px-4 py-2 text-zinc-600">{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <Section title="Links">
          <ul className="list-disc pl-5 space-y-1">
            <li>
              <Link href="/" className="text-blue-600 hover:underline">
                Live chat demo
              </Link>
            </li>
            <li>
              <Link href="/benchmark" className="text-blue-600 hover:underline">
                Interactive benchmark dashboard
              </Link>
            </li>
            <li>
              <a href={REPO} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">
                Source on GitHub
              </a>
            </li>
            <li>
              <a
                href={`${REPO}/blob/main/RESULTS.md`}
                target="_blank"
                rel="noreferrer"
                className="text-blue-600 hover:underline"
              >
                Full results write-up
              </a>
            </li>
          </ul>
        </Section>
      </main>
    </div>
  );
}

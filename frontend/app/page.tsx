import Link from "next/link";
import ChatPanel from "@/components/ChatPanel";
import { deriveHeadline, fmtDelta, loadBenchmarkSummary, pWithOperator } from "@/lib/results";

export default function Home() {
  const { parentExpansion } = deriveHeadline(loadBenchmarkSummary());
  return (
    <div className="flex flex-col h-dvh bg-zinc-50">
      <header className="border-b border-gray-200 bg-white px-5 sm:px-6 py-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h1 className="text-lg font-semibold text-zinc-900">
            PubMed GraphRAG assistant
          </h1>
          <nav aria-label="Project navigation" className="flex flex-wrap items-center gap-4">
            <Link href="/walkthrough" className="text-sm text-blue-600 hover:underline">
              3-minute walkthrough
            </Link>
            <Link href="/case-study" className="text-sm text-blue-600 hover:underline">
              Case study
            </Link>
            <Link href="/benchmark" className="text-sm text-blue-600 hover:underline">
              View benchmark &rarr;
            </Link>
          </nav>
        </div>
        <p className="text-sm text-zinc-600 mt-2">
          Graph-augmented retrieval over PubMedQA.
          {parentExpansion && (
            <>
              {" "}
              Parent-document expansion changes accuracy by{" "}
              {fmtDelta(parentExpansion.delta_acc)} over a reranked baseline (McNemar{" "}
              {pWithOperator(parentExpansion)}) &mdash;
            </>
          )}{" "}
          see{" "}
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
        <p className="mt-2 text-xs text-zinc-600">
          This live chat uses a smaller, unbenchmarked Neo4j/LLM setup. The free
          backend may take a few minutes to wake.{" "}
          <Link href="/walkthrough" className="text-blue-700 hover:underline">
            Explore the retrieval mechanism without waiting or calling the API.
          </Link>
        </p>
      </header>
      <main className="flex-1 overflow-hidden">
        <ChatPanel />
      </main>
    </div>
  );
}

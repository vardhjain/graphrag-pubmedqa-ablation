import Link from "next/link";
import ChatPanel from "@/components/ChatPanel";

export default function Home() {
  return (
    <div className="flex flex-col h-screen bg-zinc-50">
      <header className="border-b border-gray-200 bg-white px-6 py-4">
        <div className="flex items-center justify-between">
          <h1 className="text-lg font-semibold text-zinc-900">
            PubMed GraphRAG assistant
          </h1>
          <div className="flex items-center gap-4">
            <Link href="/case-study" className="text-sm text-blue-600 hover:underline">
              Case study
            </Link>
            <Link href="/benchmark" className="text-sm text-blue-600 hover:underline">
              View benchmark &rarr;
            </Link>
          </div>
        </div>
        <p className="text-sm text-zinc-500">
          Graph-augmented retrieval over PubMedQA. Parent-document expansion lifts
          accuracy +22.5pp over a reranked baseline (McNemar p&lt;0.0001) --
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
      </header>
      <main className="flex-1 overflow-hidden">
        <ChatPanel />
      </main>
    </div>
  );
}

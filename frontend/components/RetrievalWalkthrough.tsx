"use client";

import { useState } from "react";

const steps = [
  {
    title: "1. Find a matching section",
    subtitle: "Shared vector search returns short chunks, not full papers.",
    highlight: "chunk",
    detail:
      "Plain RAG would pass just this matching section to the language model. The graph-enabled arm begins with the same retrieval pool as its reranked baseline.",
  },
  {
    title: "2. Follow the parent link",
    subtitle: "Chunk → Paper → all sections of the same abstract.",
    highlight: "paper",
    detail:
      "The graph stores an explicit HAS_CONTEXT relationship from each paper to its sections. Parent expansion retrieves the sibling sections, without using the gold label or question as stored metadata.",
  },
  {
    title: "3. Assemble the complete context",
    subtitle: "Give the answer generator all sections from that paper.",
    highlight: "context",
    detail:
      "The model receives a generic STUDY label followed by the reconstructed abstract. The optional MeSH hop can add other papers, but it did not improve measured accuracy in the historical experiment.",
  },
] as const;

const sections = [
  { id: "0", label: "Background", text: "The study asks whether intervention A changes outcome B." },
  { id: "1", label: "Methods", text: "Two groups are compared using the same outcome measure." },
  { id: "2", label: "Results", text: "The observed difference is reported with its uncertainty." },
  { id: "3", label: "Conclusion", text: "The authors describe what the results support and what remains unclear." },
];

export default function RetrievalWalkthrough() {
  const [active, setActive] = useState(0);
  const step = steps[active];

  return (
    <div className="rounded-2xl border border-zinc-200 bg-white p-5 sm:p-7 shadow-sm">
      <div className="flex flex-wrap gap-2 mb-6" role="group" aria-label="Retrieval walkthrough steps">
        {steps.map((item, index) => (
          <button
            key={item.title}
            type="button"
            onClick={() => setActive(index)}
            aria-pressed={index === active}
            className={`rounded-full px-4 py-2 text-sm font-medium border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${index === active ? "bg-blue-600 text-white border-blue-600" : "bg-white border-zinc-200 text-zinc-700 hover:border-blue-400"}`}
          >
            {index + 1}. {["Retrieve", "Expand", "Assemble"][index]}
          </button>
        ))}
      </div>

      <div aria-live="polite" className="space-y-4">
        <div>
          <h3 className="text-lg font-semibold text-zinc-900">{step.title}</h3>
          <p className="mt-1 text-sm text-zinc-600">{step.subtitle}</p>
        </div>

        <div className="flex flex-wrap items-center gap-2 font-mono text-xs">
          <span className={`rounded-lg border px-3 py-2 ${step.highlight === "chunk" ? "border-blue-600 bg-blue-50 text-blue-800" : "border-zinc-200 text-zinc-600"}`}>
            matching Chunk
          </span>
          <span aria-hidden="true" className="text-zinc-400">→</span>
          <span className={`rounded-lg border px-3 py-2 ${step.highlight === "paper" ? "border-blue-600 bg-blue-50 text-blue-800" : "border-zinc-200 text-zinc-600"}`}>
            parent Paper
          </span>
          <span aria-hidden="true" className="text-zinc-400">→</span>
          <span className={`rounded-lg border px-3 py-2 ${step.highlight === "context" ? "border-blue-600 bg-blue-50 text-blue-800" : "border-zinc-200 text-zinc-600"}`}>
            LLM context
          </span>
        </div>

        <div className="rounded-xl bg-zinc-50 border border-zinc-200 p-4">
          <p className="font-medium text-sm text-zinc-900 mb-3">
            {active === 0 ? "Retrieved context" : active === 1 ? "Paper sections found via HAS_CONTEXT" : "Example context assembled for the model"}
          </p>
          {active === 2 && (
            <p className="font-mono text-xs text-zinc-500 mb-3">=== STUDY 1 ===</p>
          )}
          <div className="grid gap-2">
            {sections.filter((part) => active !== 0 || part.id === "2").map((part) => (
              <div
                key={part.id}
                className={`rounded-lg border p-3 ${part.id === "2" ? "border-blue-300 bg-blue-50" : "border-zinc-200 bg-white"}`}
              >
                <p className="text-xs font-semibold text-zinc-700">{part.label} · section {part.id}</p>
                <p className="mt-1 text-sm text-zinc-700">{part.text}</p>
              </div>
            ))}
          </div>
        </div>

        <p className="text-sm leading-relaxed text-zinc-700">{step.detail}</p>
      </div>

      <div className="mt-6 flex items-center justify-between gap-3">
        <button
          type="button"
          onClick={() => setActive((value) => Math.max(0, value - 1))}
          disabled={active === 0}
          className="rounded-lg border border-zinc-300 px-4 py-2 text-sm text-zinc-700 disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
        >
          Previous
        </button>
        <span className="text-xs text-zinc-500" aria-label={`Step ${active + 1} of ${steps.length}`}>
          {active + 1} of {steps.length}
        </span>
        <button
          type="button"
          onClick={() => setActive((value) => Math.min(steps.length - 1, value + 1))}
          disabled={active === steps.length - 1}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm text-white disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
        >
          Next step
        </button>
      </div>
      <p className="mt-4 text-xs leading-relaxed text-zinc-500">
        Synthetic illustration only: these sentences are invented, not retrieved from PubMed,
        not a saved benchmark prediction, and not a live LLM response. This walkthrough makes
        no API calls.
      </p>
    </div>
  );
}

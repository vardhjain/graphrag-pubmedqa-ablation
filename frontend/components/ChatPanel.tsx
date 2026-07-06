"use client";

import { useState } from "react";
import { askQuestion, ApiError } from "@/lib/api";
import type { ChatMessage } from "@/lib/types";
import ReasoningGraph from "./ReasoningGraph";

const EXAMPLE_QUESTIONS = [
  "Do preoperative statins reduce postoperative atrial fibrillation?",
  "Is vitamin D deficiency associated with increased mortality?",
  "Does laparoscopic surgery reduce hospital stay versus open surgery?",
];

export default function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [openReasoningFor, setOpenReasoningFor] = useState<number | null>(null);

  async function send(question: string) {
    if (!question.trim() || loading) return;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setLoading(true);

    try {
      const result = await askQuestion({ question });
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: result.answer,
          sources: result.sources,
          reasoningPath: result.reasoning_path,
        },
      ]);
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Something went wrong.";
      setMessages((prev) => [...prev, { role: "assistant", content: message, error: true }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-full w-full max-w-3xl mx-auto">
      <div className="flex-1 overflow-y-auto space-y-4 p-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-12">
            <p className="mb-4">
              Ask a biomedical question. Answers are grounded in PubMed abstracts
              retrieved via a knowledge graph -- click a sample question to try it.
            </p>
            <div className="flex flex-col gap-2 items-center">
              {EXAMPLE_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => send(q)}
                  className="text-sm px-3 py-2 rounded-full border border-gray-300 hover:border-blue-400 hover:text-blue-600 transition"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm whitespace-pre-wrap ${
                m.role === "user"
                  ? "bg-blue-600 text-white"
                  : m.error
                    ? "bg-red-50 text-red-700 border border-red-200"
                    : "bg-gray-100 text-gray-900"
              }`}
            >
              {m.content}

              {m.role === "assistant" && !m.error && m.sources && m.sources.length > 0 && (
                <div className="mt-3 pt-2 border-t border-gray-200 text-xs">
                  <span className="font-medium">Sources: </span>
                  {m.sources.map((pmid, si) => (
                    <a
                      key={pmid}
                      href={`https://pubmed.ncbi.nlm.nih.gov/${pmid}/`}
                      target="_blank"
                      rel="noreferrer"
                      className="text-blue-600 hover:underline"
                    >
                      PMID {pmid}
                      {si < m.sources!.length - 1 ? ", " : ""}
                    </a>
                  ))}
                </div>
              )}

              {m.role === "assistant" && !m.error && m.reasoningPath && (
                <div className="mt-2">
                  <button
                    onClick={() => setOpenReasoningFor(openReasoningFor === i ? null : i)}
                    className="text-xs font-medium text-blue-600 hover:underline"
                  >
                    {openReasoningFor === i ? "Hide reasoning path" : "Show reasoning path"}
                  </button>
                  {openReasoningFor === i && (
                    <div className="mt-2 bg-white rounded-lg">
                      <ReasoningGraph steps={m.reasoningPath} />
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-2xl px-4 py-3 text-sm text-gray-500 italic">
              Retrieving and reasoning over the graph...
            </div>
          </div>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="flex gap-2 p-4 border-t border-gray-200"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a biomedical question..."
          disabled={loading}
          className="flex-1 rounded-full border border-gray-300 px-4 py-2 text-sm focus:outline-none focus:border-blue-400"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="rounded-full bg-blue-600 text-white px-5 py-2 text-sm font-medium disabled:opacity-40"
        >
          Send
        </button>
      </form>
    </div>
  );
}

"use client";

import { Component, type ReactNode, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { askQuestion, warmUpBackend, ApiError } from "@/lib/api";
import { stripThink } from "@/lib/text";
import type { ChatMessage } from "@/lib/types";
import AnswerText from "./AnswerText";

// React Flow is heavy and browser-only -- load it on demand, only when a
// user actually opens a reasoning path.
const ReasoningGraph = dynamic(() => import("./ReasoningGraph"), {
  ssr: false,
  loading: () => <p className="text-xs text-gray-600 italic p-4">Loading graph...</p>,
});

// A rendering failure in the graph (bad reasoning_path shape, React Flow
// error) must not take the whole chat down with it.
class GraphErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (this.state.failed) {
      return (
        <p className="text-xs text-gray-600 italic p-4">
          The reasoning-path graph couldn&apos;t be displayed for this answer.
        </p>
      );
    }
    return this.props.children;
  }
}

const STORAGE_KEY = "kgqa-chat-messages";

function loadStoredMessages(): ChatMessage[] {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? (parsed as ChatMessage[]) : [];
  } catch {
    return [];
  }
}

const EXAMPLE_QUESTIONS = [
  "Do preoperative statins reduce postoperative atrial fibrillation?",
  "Is vitamin D deficiency associated with increased mortality?",
  "Does laparoscopic surgery reduce hospital stay versus open surgery?",
];

// Escalating loading copy: a cold free-tier backend can take 1-2 minutes on
// the first question, and a silent spinner that long reads as "broken".
function LoadingBubble() {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  const message =
    elapsed < 8
      ? "Retrieving and reasoning over the graph..."
      : elapsed < 45
        ? "Still working -- the free-tier backend is likely waking from a cold start. The first question can take a minute or two; follow-ups take ~15-20s."
        : "Almost there -- the backend is loading its models and vector cache. This only happens on the first question after an idle period.";

  return (
    <div className="flex justify-start">
      <div className="bg-gray-100 rounded-2xl px-4 py-3 text-sm text-gray-600 italic">
        {message}
      </div>
    </div>
  );
}

export default function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [openReasoningFor, setOpenReasoningFor] = useState<number | null>(null);
  const [useConcepts, setUseConcepts] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  // State updates are async, so `loading` alone can't stop a fast double
  // click/Enter from firing two requests at the small backend.
  const inFlightRef = useRef(false);

  useEffect(() => {
    warmUpBackend();
    // Restore after mount (not in the initializer) so server and first client
    // render match; keeps the chat when navigating to /benchmark and back.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMessages(loadStoredMessages());
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch {
      // Storage full/blocked (private mode) -- persistence is best-effort.
    }
  }, [messages, hydrated]);

  // Without this, the viewport stays pinned at its scroll position from
  // before the request -- so the first long answer (and every one after it)
  // lands below the fold and looks like nothing happened.
  useEffect(() => {
    // "smooth" silently no-ops in some browser contexts when the animation's
    // rAF loop never gets a tick (backgrounded/unfocused tab) -- "auto" is an
    // instant jump with no animation to get interrupted or skipped.
    bottomRef.current?.scrollIntoView({ behavior: "auto" });
  }, [messages.length, loading]);

  async function send(question: string) {
    if (!question.trim() || inFlightRef.current) return;
    inFlightRef.current = true;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setLoading(true);
    // send() only ever runs from an event handler (onClick/onSubmit below),
    // never during render -- the lint rule can't see that from here.
    // eslint-disable-next-line react-hooks/purity
    const start = performance.now();

    try {
      const result = await askQuestion({ question, use_concepts: useConcepts });
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: stripThink(result.answer ?? ""),
          sources: result.sources,
          reasoningPath: result.reasoning_path,
          latencyMs: performance.now() - start,
        },
      ]);
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Something went wrong.";
      setMessages((prev) => [...prev, { role: "assistant", content: message, error: true }]);
    } finally {
      inFlightRef.current = false;
      setLoading(false);
      // The input was disabled while loading; focus it once it re-enables.
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }

  return (
    <div className="flex flex-col h-full w-full max-w-3xl mx-auto">
      <div
        className="flex-1 overflow-y-auto space-y-4 p-4"
        role="log"
        aria-live="polite"
        aria-label="Conversation"
      >
        {messages.length === 0 && (
          <div className="text-center text-gray-600 mt-12">
            <p className="mb-4">
              Ask a biomedical question. Answers are grounded in PubMed abstracts
              retrieved via a knowledge graph -- click a sample question to try it.
            </p>
            <div className="flex flex-col gap-2 items-center">
              {EXAMPLE_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => send(q)}
                  className="text-sm px-3 py-2 rounded-full border border-gray-300 hover:border-blue-400 hover:text-blue-600 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
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
              {m.role === "assistant" && !m.error ? <AnswerText text={m.content} /> : m.content}

              {m.role === "assistant" && !m.error && ((m.sources && m.sources.length > 0) || m.latencyMs != null) && (
                <div className="mt-3 pt-2 border-t border-gray-200 text-xs">
                  {m.sources && m.sources.length > 0 && (
                    <>
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
                    </>
                  )}
                  {m.latencyMs != null && (
                    <span className="text-gray-600">
                      {m.sources && m.sources.length > 0 ? " · " : ""}
                      answered in {(m.latencyMs / 1000).toFixed(1)}s
                    </span>
                  )}
                </div>
              )}

              {m.role === "assistant" && !m.error && m.reasoningPath && (
                <div className="mt-2">
                  <button
                    onClick={() => setOpenReasoningFor(openReasoningFor === i ? null : i)}
                    className="text-xs font-medium text-blue-600 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
                  >
                    {openReasoningFor === i ? "Hide reasoning path" : "Show reasoning path"}
                  </button>
                  {openReasoningFor === i && (
                    <div className="mt-2 bg-white rounded-lg">
                      <GraphErrorBoundary>
                        <ReasoningGraph steps={m.reasoningPath} />
                      </GraphErrorBoundary>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && <LoadingBubble />}
        <div ref={bottomRef} />
      </div>

      <div className="px-4 pt-3 border-t border-gray-200">
        <label className="flex items-center gap-2 text-xs text-gray-600">
          <input
            type="checkbox"
            checked={useConcepts}
            onChange={(e) => setUseConcepts(e.target.checked)}
            className="rounded"
          />
          Explore MeSH concept hops (slower;{" "}
          <a
            href="https://github.com/vardhjain/graphrag-pubmedqa-ablation/blob/main/RESULTS.md"
            target="_blank"
            rel="noreferrer"
            className="underline hover:text-blue-600"
          >
            not shown to improve accuracy
          </a>
          )
        </label>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="flex gap-2 p-4 pt-2"
      >
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a biomedical question..."
          aria-label="Ask a biomedical question"
          maxLength={2000}
          disabled={loading}
          className="flex-1 rounded-full border border-gray-300 px-4 py-2 text-base sm:text-sm focus:outline-none focus:border-blue-400 focus-visible:ring-2 focus-visible:ring-blue-500"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="rounded-full bg-blue-600 text-white px-5 py-2 text-sm font-medium disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
        >
          Send
        </button>
      </form>
    </div>
  );
}

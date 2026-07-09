import type { QueryRequest, QueryResponse } from "./types";

// Render free tier cold-starts in ~30-50s after idling, and the backend's
// sync FastAPI route keeps running to completion server-side even after the
// client gives up waiting -- so a too-short timeout here doesn't just show
// a premature error, it encourages retrying while the abandoned first
// request is still consuming memory, which can crash the free-tier instance
// via two requests' worth of retrieval+LLM work running at once. Observed
// worst case (cold start, no reranker, Gemini synthesis) is ~190s; give it
// real headroom.
const REQUEST_TIMEOUT_MS = 200_000;

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Fire-and-forget /health ping on page load. Render's free tier spins down
// after ~15 min idle; starting the wake-up as soon as the page renders means
// the instance is often warm by the time the visitor submits a question.
export function warmUpBackend(): void {
  fetch(`${API_URL}/health`).catch(() => {});
}

export class ApiError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

export async function askQuestion(req: QueryRequest): Promise<QueryResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(`${API_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ graph_id: "demo", use_concepts: false, ...req }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? res.statusText);
      throw new ApiError(detail, res.status);
    }

    return await res.json();
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (err instanceof Error && err.name === "AbortError") {
      throw new ApiError(
        "The backend is waking up from a cold start and is taking longer than expected. Please try again in a moment."
      );
    }
    throw new ApiError("Could not reach the backend. Is it running / deployed?");
  } finally {
    clearTimeout(timeout);
  }
}

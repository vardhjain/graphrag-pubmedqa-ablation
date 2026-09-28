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
      throw new ApiError(await errorMessageFor(res), res.status);
    }

    return await res.json();
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (err instanceof Error && err.name === "AbortError") {
      throw new ApiError(
        "No answer after over 3 minutes. The backend is likely still processing your question " +
          "(cold starts are slow) -- please wait about a minute before asking again, so two " +
          "requests don't compete for the same small instance."
      );
    }
    // fetch() rejects with a TypeError on network failure/CORS/DNS -- never
    // surface the raw "Failed to fetch" to users.
    throw new ApiError(
      "Couldn't reach the demo backend. Check your connection, or retry shortly -- it may be restarting."
    );
  } finally {
    clearTimeout(timeout);
  }
}

async function errorMessageFor(res: Response): Promise<string> {
  if (res.status === 422) return "Question too long or invalid.";
  if (res.status === 502 || res.status === 503 || res.status === 504) {
    return "The demo backend is unavailable right now — retry shortly.";
  }
  const body: unknown = await res.json().catch(() => null);
  const detail =
    body && typeof body === "object" && "detail" in body
      ? (body as { detail: unknown }).detail
      : undefined;
  // Only a plain-string detail is meant for humans; FastAPI validation
  // errors are arrays of objects and would render as raw JSON.
  if (typeof detail === "string" && detail.trim()) return detail;
  return `The backend returned an error (HTTP ${res.status}). Please try again.`;
}

import type { QueryRequest, QueryResponse } from "./types";

// Render free tier cold-starts in ~30-50s after idling; give it real headroom
// instead of failing a legitimate cold start.
const REQUEST_TIMEOUT_MS = 60_000;

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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

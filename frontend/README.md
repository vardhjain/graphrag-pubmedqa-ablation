## GraphRAG PubMedQA — frontend

Next.js chat UI for the hosted agent (see [`../backend/`](../backend/) for
the FastAPI service it talks to). Deployed at
[graphrag-pubmedqa-ablation.vercel.app](https://graphrag-pubmedqa-ablation.vercel.app).

### Pages

| Route | What it is |
| --- | --- |
| `/` | The chat itself (`components/ChatPanel.tsx`) — ask a biomedical question, get an answer with cited PubMed IDs and a reasoning-path graph of the retrieval behind it. |
| `/benchmark` | Reads `../results/summary.json` directly at build time (`lib/results.ts`) and renders the ablation table + McNemar significance tests. Every headline number on `/`, `/benchmark` and `/case-study` is derived from that file (`lib/results.ts`), never typed by hand; a missing Wilson 95% CI is computed from accuracy + sample count. No API calls, no live backend needed. |
| `/case-study` | The project's own "confounded demo → fair ablation" narrative, plus the engineering story behind hosting it. |

### Local dev

```bash
npm install
npm run dev          # -> http://localhost:3000
```

Requires one env var, in `.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000    # or the deployed Render URL
```

Points the chat page (`lib/api.ts`) at the FastAPI backend's `/query` and
`/health` endpoints. Without it, `askQuestion`/`warmUpBackend` fall back to
`http://localhost:8000`.

### Vercel deploy note

If this project's Root Directory is set to `frontend`, enable "Include
source files outside of the Root Directory in the Build Step" in the Vercel
project settings — `/benchmark` reads `../results/summary.json` from outside
this directory at build time, and the build fails to find it otherwise.

---

Built with [Next.js](https://nextjs.org) (`create-next-app`) + Tailwind +
[React Flow](https://reactflow.dev) v12 (the `@xyflow/react` package, which
replaced the legacy `reactflow` v11 package) for the reasoning-path
visualization. `components/ReasoningGraph.tsx` is lazy-loaded
(`next/dynamic`, `ssr: false`) only when a user opens a reasoning path, and is
wrapped in an error boundary so a graph failure can't break the chat.

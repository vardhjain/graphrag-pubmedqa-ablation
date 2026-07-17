## GraphRAG PubMedQA — frontend

Next.js chat UI for the hosted agent (see [`../backend/`](../backend/) for
the FastAPI service it talks to). Deployed at
[graphrag-pubmedqa-ablation.vercel.app](https://graphrag-pubmedqa-ablation.vercel.app).

### Pages

| Route | What it is |
| --- | --- |
| `/` | The chat itself (`components/ChatPanel.tsx`) — ask a biomedical question, get an answer with cited PubMed IDs and a reasoning-path graph of the retrieval behind it. |
| `/benchmark` | Reads `../results/summary.json` directly at build time (`lib/results.ts`) and renders the 4-arm ablation table + McNemar significance tests. No API calls, no live backend needed. |
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
[React Flow](https://reactflow.dev) for the reasoning-path visualization.

import type { MetadataRoute } from "next";

const BASE = "https://graphrag-pubmedqa-ablation.vercel.app";

export default function sitemap(): MetadataRoute.Sitemap {
  return ["", "/benchmark", "/case-study"].map((path) => ({ url: `${BASE}${path}` }));
}

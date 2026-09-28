import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const SITE_NAME = "PubMed GraphRAG assistant";
const DESCRIPTION =
  "Ask biomedical questions answered from PubMed abstracts via graph-augmented retrieval, with a reasoning-path visualization and a fair 4-arm GraphRAG ablation behind it.";

export const metadata: Metadata = {
  metadataBase: new URL("https://graphrag-pubmedqa-ablation.vercel.app"),
  title: { default: SITE_NAME, template: `%s | ${SITE_NAME}` },
  description: DESCRIPTION,
  openGraph: {
    type: "website",
    siteName: SITE_NAME,
    title: SITE_NAME,
    description: DESCRIPTION,
    url: "/",
  },
  twitter: { card: "summary", title: SITE_NAME, description: DESCRIPTION },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}

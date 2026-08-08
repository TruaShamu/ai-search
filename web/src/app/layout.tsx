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

const SITE_URL = "https://black-grass-0df1c7a0f.7.azurestaticapps.net";
const TITLE = "BookSearch — Hybrid Semantic Search + RAG over 84,801 books";
const DESCRIPTION =
  "Hybrid semantic search + RAG over 84,801 books: TF-IDF sparse retrieval fused with dense vectors (nomic-embed-text-v1.5) via Reciprocal Rank Fusion, a cross-encoder reranker, and grounded RAG answers — self-hosted on Qdrant.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: TITLE,
  description: DESCRIPTION,
  applicationName: "BookSearch",
  keywords: [
    "semantic search",
    "vector search",
    "hybrid search",
    "RAG",
    "reciprocal rank fusion",
    "cross-encoder reranker",
    "Qdrant",
    "information retrieval",
    "nomic-embed-text",
  ],
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    url: SITE_URL,
    siteName: "BookSearch",
    title: TITLE,
    description: DESCRIPTION,
  },
  twitter: {
    card: "summary_large_image",
    title: TITLE,
    description: DESCRIPTION,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased dark`}
    >
      <body className="min-h-full flex flex-col bg-background text-foreground">
        {children}
      </body>
    </html>
  );
}

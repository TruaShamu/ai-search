import { ImageResponse } from "next/og";

// Static 1200x630 social card generated at build time (no binary asset in the
// repo). Rendered by Satori, so styles use flexbox + hex colors that mirror the
// app's dark monochrome theme (globals.css).
export const alt =
  "BookSearch — Hybrid semantic search + RAG over 84,801 books";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpengraphImage() {
  const chips = [
    "TF-IDF × Dense Vectors",
    "Reciprocal Rank Fusion",
    "Cross-Encoder Reranker",
    "Grounded RAG",
  ];
  const stack = "Kubernetes · Kafka · Terraform · AKS · OpenTelemetry";

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          background: "#0a0a0a",
          color: "#fafafa",
          padding: "72px",
          fontFamily: "sans-serif",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <div
            style={{
              display: "flex",
              width: "44px",
              height: "44px",
              borderRadius: "10px",
              background: "#fafafa",
              color: "#0a0a0a",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "26px",
              fontWeight: 700,
            }}
          >
            B
          </div>
          <div
            style={{
              display: "flex",
              fontSize: "26px",
              fontWeight: 600,
              letterSpacing: "-0.5px",
            }}
          >
            BookSearch
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          <div
            style={{
              display: "flex",
              fontSize: "62px",
              fontWeight: 700,
              lineHeight: 1.08,
              letterSpacing: "-1.5px",
              maxWidth: "980px",
            }}
          >
            Hybrid semantic search + RAG over 84,801 books
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "12px" }}>
            {chips.map((c) => (
              <div
                key={c}
                style={{
                  display: "flex",
                  padding: "8px 16px",
                  borderRadius: "999px",
                  border: "1px solid #2a2a2a",
                  background: "#141414",
                  color: "#d4d4d4",
                  fontSize: "22px",
                }}
              >
                {c}
              </div>
            ))}
          </div>
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            borderTop: "1px solid #232323",
            paddingTop: "28px",
          }}
        >
          <div style={{ display: "flex", fontSize: "22px", color: "#8a8a8a" }}>
            {stack}
          </div>
          <div style={{ display: "flex", fontSize: "22px", color: "#8a8a8a" }}>
            nomic-embed-text · Qdrant
          </div>
        </div>
      </div>
    ),
    { ...size }
  );
}

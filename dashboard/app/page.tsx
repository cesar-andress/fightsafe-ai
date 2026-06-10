"use client";

import { useEffect, useMemo, useState } from "react";

/** Mirrors GET/WebSocket JSON from ``fightsafe_ai.api.serialization.safety_event_to_json``. */
export type SafetyEventJson = {
  event_type: string;
  category: string;
  start_time: number;
  end_time: number;
  duration: number;
  level: "INFO" | "WARNING" | "HIGH" | "CRITICAL";
  score: number;
  title: string;
  description: string;
  explanation: string;
  source: string;
  timestamp_seconds: number;
};

function levelColor(level: string): string {
  switch (level) {
    case "INFO":
      return "#64748b";
    case "WARNING":
      return "#ca8a04";
    case "HIGH":
      return "#ea580c";
    case "CRITICAL":
      return "#b91c1c";
    default:
      return "#475569";
  }
}

function formatTs(ev: SafetyEventJson): string {
  const t =
    typeof ev.timestamp_seconds === "number"
      ? ev.timestamp_seconds
      : ev.end_time;
  return `${t.toFixed(2)}s`;
}

export default function LiveDashboardPage() {
  const [events, setEvents] = useState<SafetyEventJson[]>([]);
  const [wsStatus, setWsStatus] = useState<
    "idle" | "connecting" | "live" | "error"
  >("idle");

  const wsUrl = useMemo(() => {
    const explicit = process.env.NEXT_PUBLIC_WS_URL;
    if (explicit) return explicit;
    const base =
      process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
    const u = new URL(base);
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    u.pathname = "/ws/events";
    u.search = "";
    u.hash = "";
    return u.toString();
  }, []);

  useEffect(() => {
    setWsStatus("connecting");
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => setWsStatus("live");
    ws.onerror = () => setWsStatus("error");
    ws.onclose = () => setWsStatus((s) => (s === "live" ? "error" : s));

    ws.onmessage = (msg: MessageEvent<string>) => {
      try {
        const data = JSON.parse(msg.data) as SafetyEventJson;
        setEvents((prev) => [data, ...prev].slice(0, 200));
      } catch {
        /* ignore malformed frames */
      }
    };

    return () => {
      ws.close();
    };
  }, [wsUrl]);

  const videoSrc = process.env.NEXT_PUBLIC_VIDEO_URL;

  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        fontFamily:
          'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
      }}
    >
      <section
        style={{
          flex: "1 1 55%",
          minWidth: 0,
          borderRight: "1px solid #e2e8f0",
          padding: 16,
          display: "flex",
          flexDirection: "column",
          gap: 12,
        }}
      >
        <h1 style={{ margin: 0, fontSize: "1.15rem" }}>Video</h1>
        {videoSrc ? (
          <video
            controls
            playsInline
            src={videoSrc}
            style={{
              width: "100%",
              maxHeight: "calc(100vh - 100px)",
              background: "#000",
            }}
          />
        ) : (
          <p style={{ color: "#64748b", margin: 0, lineHeight: 1.5 }}>
            Opcional: define <code>NEXT_PUBLIC_VIDEO_URL</code> para reproducir
            un archivo (p. ej. el mismo vídeo que procesa la API con{" "}
            <code>FIGHTSAFE_LIVE_SOURCE</code>). El pipeline corre en el backend.
          </p>
        )}
      </section>

      <aside
        style={{
          flex: "0 0 380px",
          maxWidth: "100%",
          padding: 16,
          overflow: "auto",
          background: "#f8fafc",
        }}
      >
        <h1 style={{ margin: "0 0 8px", fontSize: "1.15rem" }}>
          Eventos en vivo
        </h1>
        <p style={{ margin: "0 0 12px", fontSize: 12, color: "#64748b" }}>
          WebSocket:{" "}
          <strong
            style={{
              color:
                wsStatus === "live"
                  ? "#15803d"
                  : wsStatus === "connecting"
                    ? "#ca8a04"
                    : "#b91c1c",
            }}
          >
            {wsStatus === "live"
              ? "conectado"
              : wsStatus === "connecting"
                ? "conectando…"
                : wsStatus === "error"
                  ? "error / desconectado"
                  : "—"}
          </strong>
        </p>
        <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
          {events.map((ev, i) => (
            <li
              key={`${ev.timestamp_seconds}-${ev.event_type}-${i}`}
              style={{
                borderBottom: "1px solid #e2e8f0",
                padding: "10px 0",
              }}
            >
              <div
                style={{
                  fontWeight: 700,
                  fontSize: 12,
                  letterSpacing: "0.04em",
                  color: levelColor(ev.level),
                  marginBottom: 4,
                }}
              >
                {ev.level}
              </div>
              <div style={{ fontSize: 14, color: "#0f172a", marginBottom: 4 }}>
                {ev.title}
              </div>
              <div style={{ fontSize: 11, color: "#64748b" }}>
                <span>{formatTs(ev)}</span>
                <span style={{ margin: "0 6px" }}>·</span>
                <span>{ev.category}</span>
              </div>
            </li>
          ))}
        </ul>
        {events.length === 0 && wsStatus === "live" && (
          <p style={{ fontSize: 13, color: "#94a3b8", marginTop: 12 }}>
            Esperando eventos…
          </p>
        )}
      </aside>
    </div>
  );
}

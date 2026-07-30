/* global fetch */

const $ = (id) => document.getElementById(id);

let ws;
let cachedEvents = [];
let lastStatus = {};
let lastMeta = {};
/** @type {Set<string>} */
const reviewedEventIds = new Set();

/** @type {string | null} */
let timelineHighlightedEventId = null;

function levelClass(level) {
  const l = (level || "").toUpperCase();
  if (l === "WARNING") return "level-WARNING";
  if (l === "HIGH") return "level-HIGH";
  if (l === "CRITICAL") return "level-CRITICAL";
  return "level-INFO";
}

function pickReadableDetail(ev) {
  const d = (ev.description || "").trim();
  const x = (ev.explanation || "").trim();
  if (d.length > 0) return d;
  if (x.length > 0) return x;
  return "";
}

function isTapkoEvent(ev) {
  const et = String(ev.event_type || "");
  return (
    et.startsWith("submission_signal.") ||
    et.startsWith("extreme_vulnerability.") ||
    !!(ev.metadata && ev.metadata.tapko_family)
  );
}

function tapkoFamilySubtype(ev) {
  const meta = ev.metadata || {};
  const et = String(ev.event_type || "");
  if (meta.tapko_family && meta.tapko_subtype) {
    return { family: meta.tapko_family, subtype: meta.tapko_subtype };
  }
  const dot = et.indexOf(".");
  if (dot === -1) return { family: "", subtype: et };
  return { family: et.slice(0, dot), subtype: et.slice(dot + 1) };
}

function formatEvidenceJson(evidence) {
  if (evidence == null) return "";
  try {
    return JSON.stringify(evidence, null, 2);
  } catch {
    return String(evidence);
  }
}

function scoreOrConfidenceLabel(ev) {
  const s = ev.score;
  if (s == null || Number.isNaN(Number(s))) return "—";
  return Number(s).toFixed(3);
}

function tapkoExtraLines(ev) {
  const meta = ev.metadata || {};
  const lines = [];
  if (meta.repetition_count != null) lines.push(`Repetitions: ${meta.repetition_count}`);
  if (meta.level != null) lines.push(`Candidate level: ${meta.level}`);
  return lines;
}

function formatClock(seconds) {
  if (seconds == null || Number.isNaN(seconds)) return "—";
  const s = Math.max(0, Number(seconds));
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  const sec = r < 10 ? `0${r.toFixed(2)}` : r.toFixed(2);
  return `${m}:${sec}`;
}

function renderEvents(events) {
  cachedEvents = events;
  const list = $("eventList");
  list.innerHTML = "";
  const sorted = [...events].reverse();
  for (const ev of sorted) {
    const eid = ev.event_id || "";
    const card = document.createElement("div");
    const reviewed = eid && reviewedEventIds.has(eid);
    const tapko = isTapkoEvent(ev);
    card.className = `event-card ${levelClass(ev.level)}${
      reviewed ? " event-reviewed" : ""
    }${tapko ? " event-card-tapko" : ""}`;
    card.dataset.eventId = eid;
    const metaRaw = ev.metadata || {};
    const detail = pickReadableDetail(ev);
    const expl = String(ev.explanation || "").trim();
    const detailBlock =
      tapko
        ? ""
        : detail.length > 0
          ? `<p class="event-desc">${escapeHtml(detail)}</p>`
          : "";
    const explBlock =
      expl.length > 0 && expl !== detail
        ? `<p class="event-desc event-explanation">${escapeHtml(expl)}</p>`
        : "";
    const headRow = `
      <div class="event-head">
        <span class="severity-pill ${levelClass(ev.level)}">${escapeHtml(ev.level)}</span>
        <div class="title">${escapeHtml(ev.title || ev.event_type)}</div>
      </div>`;

    let tapkoBlock = "";
    if (tapko) {
      const fs = tapkoFamilySubtype(ev);
      const meta = metaRaw;
      const evidence = meta.evidence || null;
      const evSummary = String(meta.evidence_summary || detail || "").trim();
      const extra = tapkoExtraLines(ev);
      const extraHtml =
        extra.length > 0
          ? `<div class="tapko-extra">${extra.map((x) => escapeHtml(x)).join("<br/>")}</div>`
          : "";
      const conf =
        ev.requires_human_confirmation === true
          ? `<div class="confirm-pill">Human confirmation required</div>`
          : "";
      tapkoBlock = `
      <div class="tapko-fields">
        <dl>
          <dt>Family</dt><dd>${escapeHtml(fs.family || "—")}</dd>
          <dt>Subtype</dt><dd>${escapeHtml(fs.subtype || "—")}</dd>
          <dt>Score</dt><dd>${escapeHtml(scoreOrConfidenceLabel(ev))}</dd>
          <dt>Evidence summary</dt><dd>${escapeHtml(evSummary || "—")}</dd>
          <dt>Event type</dt><dd><code>${escapeHtml(String(ev.event_type || ""))}</code></dd>
        </dl>
        ${extraHtml}
        ${conf}
      </div>
      ${
        evidence
          ? `<div class="tapko-evidence-wrap"><span class="tapko-evidence-label">Structured evidence (detail)</span><pre class="tapko-evidence" aria-label="Structured evidence">${escapeHtml(formatEvidenceJson(evidence))}</pre></div>`
          : ""
      }`;
    }

    const feedbackSection = `
      <div class="feedback-section">
        <label class="feedback-note-label">Note (optional)
          <textarea class="feedback-note" rows="2" maxlength="8000" placeholder="Optional note"></textarea>
        </label>
        <div class="feedback-grid">
          <div class="feedback-row">
            <button type="button" data-fb="correct">Correct</button>
            <button type="button" data-fb="false_positive">False positive</button>
            <button type="button" data-fb="missed_event">Missed event</button>
          </div>
          <div class="feedback-row">
            <button type="button" data-fb="wrong_subtype">Wrong subtype</button>
            <button type="button" data-fb="wrong_severity">Wrong severity</button>
            <button type="button" data-fb="needs_expert_review">Needs expert review</button>
          </div>
        </div>
      </div>`;

    if (reviewed) {
      card.innerHTML = `
      ${headRow}
      <div class="meta">${formatClock(ev.timestamp_seconds)} · ${(ev.duration ?? 0).toFixed(2)}s</div>
      ${tapkoBlock}
      ${detailBlock}
      ${explBlock}
      <div class="feedback-reviewed"><strong>Reviewed</strong> · feedback saved</div>`;
    } else if (!eid) {
      card.innerHTML = `
      ${headRow}
      <div class="meta">${formatClock(ev.timestamp_seconds)} · ${(ev.duration ?? 0).toFixed(2)}s</div>
      ${tapkoBlock}
      ${detailBlock}
      ${explBlock}`;
    } else {
      card.innerHTML = `
      ${headRow}
      <div class="meta">${formatClock(ev.timestamp_seconds)} · ${(ev.duration ?? 0).toFixed(2)}s</div>
      ${tapkoBlock}
      ${detailBlock}
      ${explBlock}
      ${feedbackSection}`;
      card.querySelectorAll("[data-fb]").forEach((btn) => {
        btn.addEventListener("click", () => sendFeedback(eid, btn.dataset.fb, card));
      });
    }
    list.appendChild(card);
  }
  renderTimeline(cachedEvents, lastStatus, lastMeta);
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

async function sendFeedback(eventId, feedbackType, cardEl) {
  if (!eventId || !cardEl) return;
  const noteEl = cardEl.querySelector(".feedback-note");
  let note = null;
  if (noteEl && noteEl.value.trim()) {
    note = noteEl.value.trim();
  }
  const btns = cardEl.querySelectorAll("[data-fb]");
  btns.forEach((b) => {
    b.disabled = true;
  });
  try {
    const r = await fetch(
      `/events/${encodeURIComponent(eventId)}/feedback`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          feedback_type: feedbackType,
          note,
        }),
      },
    );
    if (!r.ok) {
      btns.forEach((b) => {
        b.disabled = false;
      });
      return;
    }
    reviewedEventIds.add(eventId);
    cardEl.classList.add("event-reviewed");
    const section = cardEl.querySelector(".feedback-section");
    if (section) {
      section.replaceWith(
        (() => {
          const d = document.createElement("div");
          d.className = "feedback-reviewed";
          d.innerHTML =
            "<strong>Reviewed</strong> · feedback saved";
          return d;
        })(),
      );
    }
  } catch {
    btns.forEach((b) => {
      b.disabled = false;
    });
  }
}

function highlightEventFromTimeline(eventId) {
  if (!eventId) return;
  timelineHighlightedEventId = eventId;
  const list = $("eventList");
  if (!list) return;
  list.querySelectorAll(".event-card").forEach((c) => {
    c.classList.remove("event-highlight");
  });
  const cards = list.querySelectorAll(".event-card");
  for (const card of cards) {
    if (card.dataset.eventId === eventId) {
      card.classList.add("event-highlight");
      card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      break;
    }
  }
}

function renderTimeline(events, s, meta) {
  const track = $("timelineTrack");
  const fill = $("timelineFill");
  const markers = $("timelineMarkers");
  const playhead = $("timelinePlayhead");
  const rangeEl = $("timelineRange");
  const scaleEnd = $("timelineScaleEnd");

  const media = s.media_timestamp_seconds;
  const maxMedia = media != null ? Number(media) : 0;
  const evTimes = events.map((e) => e.timestamp_seconds ?? 0);
  const maxEv = evTimes.length ? Math.max(...evTimes, 0) : 0;

  const durMeta =
    meta?.duration_seconds != null && Number(meta.duration_seconds) > 0
      ? Number(meta.duration_seconds)
      : null;

  let fullDuration = durMeta;
  if (fullDuration == null || fullDuration <= 0) {
    fullDuration = Math.max(maxEv, maxMedia, 1e-3);
  }

  const pctAlong = (tSec) =>
    Math.min(100, Math.max(0, (Number(tSec) / fullDuration) * 100));

  const head = media != null ? pctAlong(media) : 0;
  playhead.style.left = `${head}%`;
  fill.style.width = `${head}%`;

  markers.innerHTML = "";
  for (const ev of events) {
    const t = ev.timestamp_seconds ?? 0;
    const eid = ev.event_id || "";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `timeline-marker ${levelClass(ev.level)}`;
    if (eid && timelineHighlightedEventId && eid === timelineHighlightedEventId) {
      btn.classList.add("timeline-marker-active");
    }
    btn.style.left = `${pctAlong(t)}%`;
    const tsLabel = formatClock(t);
    let titleShort = (ev.title || ev.event_type || "").slice(0, 80);
    if (ev.metadata && ev.metadata.tapko_family) {
      titleShort = `${ev.metadata.tapko_family} · ${titleShort}`.slice(0, 96);
    }
    btn.title = `${tsLabel} · ${ev.level} · ${titleShort}`;
    btn.setAttribute(
      "aria-label",
      `Event at ${tsLabel}, ${ev.level}. Click to highlight in list.`,
    );
    if (eid) {
      btn.dataset.eventId = eid;
    }
    btn.addEventListener("click", (evClick) => {
      evClick.preventDefault();
      evClick.stopPropagation();
      if (!eid) return;
      highlightEventFromTimeline(eid);
      renderTimeline(cachedEvents, lastStatus, lastMeta);
    });
    markers.appendChild(btn);
  }

  const curLabel = formatClock(media);
  const endLabel = formatClock(fullDuration);
  rangeEl.textContent = `${curLabel} / ${endLabel}`;
  if (scaleEnd) {
    scaleEnd.textContent = endLabel;
  }
  track.title = `Playhead ${curLabel} of ${endLabel}. Click colored dots to jump to events.`;
}

function updateEventsEmptyHint(s) {
  const el = $("eventsEmptyHint");
  if (!el) return;
  const st = (s.status || "").toLowerCase();
  const running = st === "running";
  const n = Number(s.event_count ?? 0);
  const demo = !!s.demo_events;
  const startedWall = s.session_started_wall;
  const started =
    startedWall != null && Number.isFinite(Number(startedWall));
  let show = false;
  if (running && !demo && n === 0 && started) {
    const elapsedSec = Date.now() / 1000 - Number(startedWall);
    if (elapsedSec > 10) show = true;
  }
  el.hidden = !show;
}

async function refreshStatus() {
  const [rStatus, rMeta] = await Promise.all([
    fetch("/session/status"),
    fetch("/session/metadata"),
  ]);
  const s = await rStatus.json();
  const meta = await rMeta.json();
  lastStatus = s;
  lastMeta = meta;

  $("videoDuration").textContent =
    meta.duration_seconds != null ? formatClock(meta.duration_seconds) : "—";
  $("progressPct").textContent =
    meta.progress_percent != null ? `${Number(meta.progress_percent).toFixed(1)}%` : "—";
  const pf = meta.processed_frames;
  const tf = meta.total_frames;
  $("frameProgress").textContent =
    pf != null && tf != null ? `${pf} / ${tf}` : pf != null ? `${pf} / —` : "—";

  $("sessionBadge").textContent = s.status || "—";
  $("sessionBadge").classList.toggle("completed", !!s.completed);

  $("eventCount").textContent = String(s.event_count ?? 0);
  const rl = (s.risk_level ?? "—").trim();
  const hero = $("riskHero");
  const rawBand = $("riskRawBand");
  if (hero) {
    hero.textContent = rl || "—";
    hero.className =
      rl === "—" || rl === ""
        ? "risk-hero-value risk-hero-neutral"
        : `risk-hero-value ${levelClass(rl)}`;
  }
  if (rawBand) {
    const raw = String(s.raw_risk_level ?? "").trim();
    rawBand.textContent =
      raw && raw !== "—" ? `Raw band: ${raw}` : "";
  }

  $("fpsVal").textContent =
    s.fps != null ? Number(s.fps).toFixed(1) : "—";
  $("latVal").textContent =
    s.latency_ms != null ? `${Number(s.latency_ms).toFixed(1)} ms` : "—";

  const mt = s.media_timestamp_seconds;
  $("mediaTime").textContent = mt != null ? formatClock(mt) : "—";

  $("completedBanner").hidden = !s.completed;

  updateEventsEmptyHint(s);

  renderTimeline(cachedEvents, s, meta);
}

async function refreshEvents() {
  const r = await fetch("/events?limit=500");
  const events = await r.json();
  renderEvents(events);
}

function connectWs() {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${window.location.host}/ws/events`);
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === "event" && msg.event) {
        refreshEvents();
      } else if (msg.type === "events_snapshot" && Array.isArray(msg.events)) {
        renderEvents(msg.events);
      } else if (msg.type === "session") {
        refreshStatus();
      }
    } catch {
      refreshEvents();
    }
  };
  ws.onclose = () => setTimeout(connectWs, 2000);
}

$("btnStart").addEventListener("click", async () => {
  await fetch("/session/start", { method: "POST" });
  $("mjpeg").src = "/video/stream?t=" + Date.now();
  refreshStatus();
  refreshEvents();
});

$("btnPause").addEventListener("click", async () => {
  await fetch("/session/pause", { method: "POST" });
  refreshStatus();
});

$("btnResume").addEventListener("click", async () => {
  await fetch("/session/resume", { method: "POST" });
  refreshStatus();
});

$("btnStop").addEventListener("click", async () => {
  await fetch("/session/stop", { method: "POST" });
  refreshStatus();
});

$("btnExport").addEventListener("click", async () => {
  await fetch("/session/export", { method: "POST" });
  alert("Exported to outputs/live/events.json and outputs/live/events.csv");
});

$("btnClear").addEventListener("click", async () => {
  if (!window.confirm("Clear all in-memory events?")) return;
  await fetch("/session/clear", { method: "POST" });
  reviewedEventIds.clear();
  timelineHighlightedEventId = null;
  renderEvents([]);
  refreshStatus();
});

async function refreshGpu() {
  try {
    const r = await fetch("/system/gpu");
    const g = await r.json();
    const note = $("gpuStatusNote");
    if (!g.nvidia_nvml_available) {
      note.hidden = false;
      note.textContent =
        g.message ||
        (g.status === "nvml_unavailable"
          ? "GPU monitoring unavailable (install nvidia-ml-py on NVIDIA hosts)."
          : "GPU monitoring unavailable.");
      $("gpuName").textContent = "—";
      $("gpuUtil").textContent = "—";
      $("gpuVram").textContent = "—";
      $("gpuTemp").textContent = "—";
      $("gpuPower").textContent = "—";
    } else {
      note.hidden = true;
      $("gpuName").textContent = g.gpu_name || "—";
      $("gpuUtil").textContent =
        g.gpu_utilization_percent != null
          ? `${Number(g.gpu_utilization_percent).toFixed(0)}%`
          : "—";
      const u = g.memory_used_mib;
      const t = g.memory_total_mib;
      const pct = g.memory_percent;
      $("gpuVram").textContent =
        u != null && t != null
          ? `${u.toFixed(0)} / ${t.toFixed(0)} MiB${
              pct != null ? ` (${pct.toFixed(1)}%)` : ""
            }`
          : "—";
      $("gpuTemp").textContent =
        g.temperature_c != null ? `${Number(g.temperature_c).toFixed(0)} °C` : "—";
      $("gpuPower").textContent =
        g.power_available && g.power_draw_watts != null
          ? `${Number(g.power_draw_watts).toFixed(1)} W`
          : "—";
    }
    $("cudaAvail").textContent = g.cuda_available ? "yes" : "no";
    $("poseBackend").textContent = g.pose_backend ?? "—";
    $("poseDevice").textContent = g.pose_device ?? "—";
  } catch {
    $("gpuStatusNote").hidden = false;
    $("gpuStatusNote").textContent = "Could not load GPU stats.";
  }
}

setInterval(refreshStatus, 800);
setInterval(refreshGpu, 1000);
connectWs();
refreshStatus();
refreshEvents();
refreshGpu();

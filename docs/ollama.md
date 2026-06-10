# FightSafe AI

**Authors:**
David Martin Moncunill (david.martinm@ucjc.edu)
César Andrés Sánchez (cesar.andress@ucjc.edu)

**Affiliation:**
Camilo José Cela University (UCJC)
Madrid, Spain

---

# Ollama integration (optional)

FightSafe AI can call a **local** [Ollama](https://ollama.com) server to **narrate** structured risk data in Markdown. **Ollama is fully optional** and does not participate in **pose estimation**, **feature tables**, or **numeric risk scores**.

For secrets and data handling, see [security-and-data-policy.md](security-and-data-policy.md). For a short overview, see the [README](../README.md) (section **Using Ollama**).

---

## What Ollama is used for

- **Explanations and reports only** — Post-hoc text for **event-level** or **clip-level** summaries (e.g. `fightsafe explain-event`, `fightsafe run-pipeline --explain-events`), and any path that uses `fightsafe_ai.llm.reporting` with `use_ollama=True`.
- Ollama is **not** a second opinion on safety: it **rephrases and contextualizes** fields already produced by **deterministic** pipelines (pose + rules + interpretable scoring when applicable).

## What Ollama must not do

- **Ollama must not replace deterministic risk scoring.** Frame-wise and rule-based outputs from `fightsafe_ai.risk` and related modules are **authoritative** for numbers, flags, and event boundaries in this design. The LLM layer is **narration**, not a replacement engine.

## Operating without Ollama

- **The system must work without Ollama.** Pipelines, CLI commands, and library APIs for **video → pose → features → risk → overlay** do **not** require a running Ollama process or any HTTP call to it.
- **If Ollama is disabled or unavailable**, the code uses **rule-based, template fallbacks** (`fightsafe_ai.llm.risk_explainer.fallback_risk_explanation`), built from the same **structured** event fields (time range, level, max score, optional triggered rules). The same path is used when the LLM is enabled but a request fails (`LLMError`, network errors, etc.): the implementation **logs a warning** and **falls back** to the deterministic string.

In short: **no Ollama** (or a failed Ollama call) still yields a **useful, consistent** non-LLM explanation text, not a crash of the whole pipeline (individual features may log errors per their own contracts).

## Defaults in this repository

The canonical settings live in [`configs/llm.yaml`](../configs/llm.yaml):

| Setting | Default | Notes |
|---------|---------|--------|
| **HTTP base URL** | `http://localhost:11434` | Ollama’s default; client uses `{base_url}/api/generate` |
| **Model** | `llama3.1` | Must be pulled locally (see below) |
| `ollama.enabled` | `true` | If `false`, code paths that respect config skip **generation** and use the template fallback only |

You can point `llm_config` to another YAML or override the model on the CLI.

---

## How to install Ollama

Install the **Ollama application** for your platform from the official site:

- **https://ollama.com** — follow **Download** for **Linux**, **macOS**, or **Windows**.

Typical one-line install (Linux, check Ollama docs for the current script):

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

On macOS/Windows, use the **installer** from the website. After installation, ensure the Ollama **daemon** is running so that `http://localhost:11434` responds (default).

---

## How to pull a model

Pull a model **before** FightSafe will successfully generate with that model. The repository defaults to **`llama3.1`** (see `configs/llm.yaml`).

```bash
ollama pull llama3.1
```

To use another model, pull it and set `ollama.model` in `configs/llm.yaml` or pass `--model` / `--ollama-explain-model` on the CLI where supported.

---

## How to run local explanations

**1. Single event from a JSON file → Markdown** (`fightsafe explain-event`):

```bash
# With Ollama (default) using configs/llm.yaml
fightsafe explain-event --event-json ./path/to/event.json -o ./out/explanation.md

# Override model
fightsafe explain-event --event-json ./event.json -o ./out/explanation.md --model llama3.1
```

**2. Full pipeline with per-event explanations** (`fightsafe run-pipeline`):

```bash
fightsafe run-pipeline --video ./data/clips/clip.mp4 -o ./runs/clip_001/ --explain-events
```

With `--explain-events`, the run writes `explanations/event_*.md` for relevant segments, using Ollama when allowed by config and flags. Risk CSV and demo video do **not** depend on Ollama succeeding for text.

---

## How to disable Ollama

You can turn off **LLM generation** at **three** levels (combine as needed).

**1. Configuration file** — In `configs/llm.yaml`, set:

```yaml
ollama:
  enabled: false
```

When `enabled` is `false`, the explainer does **not** call the Ollama API and always uses the **template / rule-based** explanation (same as “unavailable Ollama” for text output).

**2. CLI: single event, no Ollama**

```bash
fightsafe explain-event --event-json ./event.json -o ./out.md --no-use-ollama
```

**3. CLI: pipeline explanations without Ollama**

```bash
fightsafe run-pipeline --video ./v.mp4 -o ./out/ --explain-events --explanations-no-ollama
```

With `--explanations-no-ollama`, the pipeline still writes **Markdown** files, but the body of each file is the **fallback** text, not a model completion.

**4. Library** — When calling `write_explanation_markdown` or `generate_pipeline_event_explanations`, pass `use_ollama=False` (or rely on `ollama.enabled: false` in the resolved config).

---

## See also

- [`configs/llm.yaml`](../configs/llm.yaml) — temperature, timeout, disclaimer and review-threshold text options under `explanations:`
- `src/fightsafe_ai/llm/risk_explainer.py` — `explain_risk_event` and `fallback_risk_explanation`
- [`SECURITY.md`](../SECURITY.md) — if Ollama is bound to non-local hosts

# FightSafe AI

**Authors:**
David Martin Moncunill (david.martinm@ucjc.edu)
César Andrés Sánchez (cesar.andress@ucjc.edu)

**Affiliation:**
Camilo José Cela University (UCJC)
Madrid, Spain

---

# Security and data policy

This document states **strict rules** for what must **not** enter version control, how to handle **local data**, and what we expect when **datasets** or **models** are discussed in issues or pull requests. It complements the repository [`.gitignore`](../.gitignore) and [`SECURITY.md`](../SECURITY.md) (vulnerability reporting).

**Scope:** everyone with commit access, contributors, and anyone publishing forks or copies of the software.

**Product scope:** FightSafe AI is **research software for decision support**—not a certified medical device or officiating system. Documentation and issues must **not** imply certified safety, referee replacement, or validated knockout detection unless backed by an explicit, peer-reviewed study cited in context.

---

## Secrets and credentials

- **Do not commit secrets** — No passwords, private tokens, HMAC keys, or session material.
- **Do not commit `.env` files** — Real configuration with secrets must stay on your machine. Only **templates** that contain **placeholders** may be committed (e.g. `.env.example`, `.env.sample`).
- **Do not commit API keys** — Including cloud keys, YouTube/cookie strings, or LLM service keys. Use environment variables and local-only files.
- **Use `.env.example` for documentation only** — It must list **variable names** and **dummy values** or clear placeholders, never real credentials. The same rule applies to any `*.example` env file.

If you accidentally commit a secret, **revoke the credential immediately**, then remove it from git history (e.g. `git filter-repo` or support from maintainers) before relying on the branch again.

---

## Media and large binaries

- **Do not commit raw videos** — Match footage, sparring, or any full-resolution recordings belong outside the repo (local `data/`, object storage, or a licensed dataset host).
- **Do not commit downloaded YouTube (or other platform) videos** — Caching or mirroring platform content in git violates typical terms of use and bloats history. Use URLs or your own **documented** acquisition process outside the tree.
- **Do not commit generated frames** — Extracted JPEG/PNG frame sequences, thumbnails at scale, or rendered report videos are **build artifacts**, not source.
- **Do not commit trained model checkpoints** (`.pt`, `.pth`, `.onnx`, large `.bin`, **unless explicitly approved** by maintainers in writing for a specific, justified release) — The default is **no** large weight files in the mainline tree.

---

## Personal and sensitive data

- **Do not commit athlete personal data** — No names, IDs, birthdates, medical notes, or identifying imagery tied to real individuals without **lawful basis** and **consent** appropriate to your jurisdiction and use case. The research toolkit is not a consent-management system: **assume public repos are public**.

Favor **synthetic** or **anonymized** material in examples, tests, and issues.

---

## Local layout: `data/` and folder structure

- **Use `data/` only for local experimentation** — Paths under `data/` (raw, frames, clips, processed) are **gitignored** by design. Nothing under those trees should be committed.
- **Use `.gitkeep`** in otherwise empty directories when the **project layout** must be documented (e.g. `data/raw/.gitkeep`). Only tiny placeholder files; never use `.gitkeep` to sneak in data.

---

## Datasets and licensing

- **Any dataset** (whether you link it in a paper, issue, or PR) must have **clear licensing documentation** — Cite the license, redistribution terms, and any **attribution** or **non-commercial** constraints. If you cannot state the license, do not point contributors at it as a required dependency of the project without maintainer review.

- Pull requests that add **sample data** in-repo must be **small**, **synthetic** or **explicitly licensed**, and **documented** in `docs/` or the PR description (source, license, and purpose).

---

## Relationship to code

- The application code must **not** `print` or log **secrets** at default log levels. Follow [`engineering-standards.md`](engineering-standards.md) and [`.cursor/rules/fightsafe-ai.md`](../.cursor/rules/fightsafe-ai.md) for logging and input validation.
- For **vulnerability** handling (e.g. dependency CVEs, unsafe defaults), see [`SECURITY.md`](../SECURITY.md).

---

## Summary checklist (before `git commit`)

- [ ] No `.env` (only `.env.example` / `.env.sample` with placeholders if needed)
- [ ] No keys, tokens, cookies, or `secrets/` content
- [ ] No video files, no bulk frames, no downloads from YouTube in-tree
- [ ] No PII of athletes or subjects without a documented, lawful basis
- [ ] No large checkpoints unless a maintainer explicitly approved the exception
- [ ] New external data: license + `docs` note (or link to a policy-compliant archive)

If any box is uncertain, **do not commit**—ask in an issue or draft PR first.

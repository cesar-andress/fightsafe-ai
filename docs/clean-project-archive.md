# Clean project archive (no VCS, caches, or run artifacts)

Use this when you need a **small, shareable ZIP** of the FightSafe AI source and configuration: reviewers, lab partners, or offline backup—**without** Git history, virtualenvs, Python caches, pipeline outputs, or local media.

## Recommended: `git archive`

`git archive` only packs **tracked files** (what Git has in the index for the ref you name). It **does not** include untracked items (typical for `data/`, `runs/`, etc. when ignored) and it **omits** `.git/` by design.

From the repository root, on the default branch **`main`**, run:

```bash
git archive --format=zip --output fightsafe-ai-clean.zip main
```

If you use another primary branch, replace `main` with that ref (e.g. `master`).

- To archive the **current HEAD** regardless of branch name: `git archive --format=zip --output fightsafe-ai-clean.zip HEAD`
- The ZIP contains paths **relative to the repo root** (no `.git/` directory).

**Note:** Anything **never committed** and ignored (e.g. `runs/`, local `videos/`) is **not** in a `git archive` because Git does not track it.

## When you build a ZIP manually (Finder, Explorer, `zip`, cloud export)

File managers often bundle **entire** folders, including **ignored** or **untracked** content. Exclude at least the following (paths are relative to the project root when applicable):

| Path / pattern     | Why exclude                                      |
| ----------------- | ------------------------------------------------ |
| `.git/`           | Version control metadata (optional for readers)  |
| `.venv/`          | Local Python virtual environment                 |
| `__pycache__/`    | Bytecode (any depth)                            |
| `.pytest_cache/`  | Pytest cache                                    |
| `.ruff_cache/`   | Ruff cache                                      |
| `.mypy_cache/`   | Mypy cache                                      |
| `runs/`          | Pipeline run outputs (overlays, CSVs, reports)  |
| `outputs/`       | Generic output root if used                    |
| `reports/`      | Generated reports if present locally             |
| `data/clips/`   | Local video clips (often large)                 |
| `data/raw/`     | Raw or proprietary inputs                       |
| `videos`        | Local video directories or single files         |
| `frames`        | Extracted frame sequences                       |

**Tip:** If you have **only** a manual ZIP workflow, `git clean -fdX -n` (dry run) and `git clean -fdX` (after review) can help remove some ignored build/cache artifacts in a working tree; still verify your exclude list, and do **not** use clean options that remove untracked *tracked* work you need.

## Summary

| Method           | Excludes `.git` | Excludes typical caches / runs / data           |
| ---------------- | -------------- | ------------------------------------------------ |
| `git archive`    | Yes (built-in) | Tracked source only; ignores stay out by default |
| Manual ZIP      | You must      | You must; use the table above as a checklist   |

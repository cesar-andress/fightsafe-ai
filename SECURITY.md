# Security policy

This document applies to the [FightSafe AI](https://github.com/cesar-andress/fightsafe-ai) repository. For **what not to put in git** (secrets, videos, PII, checkpoints), read [`docs/security-and-data-policy.md`](docs/security-and-data-policy.md).

---

## How to report vulnerabilities

**Please do not** open a **public** GitHub issue for **undisclosed** security vulnerabilities (it exposes other users to risk before a fix exists).

1. **Preferred — GitHub** — If enabled for this repository, use **Report a vulnerability** in the [Security](https://github.com/cesar-andress/fightsafe-ai/security) tab and open a **private** security advisory, or follow **GitHub’s** guided disclosure flow.
2. **Content to include** — Affected component (e.g. `fightsafe_ai.video`, CLI), version or commit, **steps to reproduce**, impact, and, if you have one, a **suggested** fix (patch or design).
3. **Response** — Maintainers will acknowledge in a reasonable time and work toward a **coordinated** disclosure when appropriate. This is a **volunteer research** project: timelines may vary.

For **low-risk** hardening (e.g. “dependency X should be capped”) a normal issue or pull request is acceptable if it does not reveal a live, exploitable chain before maintainers can assess it.

---

## Supported versions

Security fixes are **prioritized** on the **default branch** (`main`) and released through normal development practices.

| Line | Status | Notes |
|------|--------|--------|
| **Python 3.12** | Supported | As declared in `pyproject.toml` (`>=3.12,<3.13`) |
| **fightsafe-ai (main / latest tag)** | Supported for active development | **0.x**; breaking changes are possible; pin versions in production experiments |
| **Older commits / forks** | Best effort only | No guarantee of backports; upgrade when possible |

There is no separate **LTS** branch at this time. If you need a long-lived deployment, **pin** a specific commit or tag and monitor **advisories** and **changelogs**.

---

## Security expectations

**What the project is**

- A **local-first research toolkit** with optional **local** LLM calls (Ollama). It is **not** a hosted SaaS; attack surface is mainly **the machine running the code** and **any network features you enable** (e.g. download, Ollama HTTP to localhost).
- **Dependencies** are declared in `pyproject.toml`. CI runs tests and static checks; it does not replace your own **update** and **vulnerability** review of third-party packages.

**What we expect from contributors and operators**

- **No secrets in the repo** — See [`docs/security-and-data-policy.md`](docs/security-and-data-policy.md).
- **Validate input** — Paths, URLs, and file formats should be **checked**; avoid arbitrary code execution from user-controlled strings.
- **Principle of least privilege** — Run with minimal OS permissions; do not run as root for analysis jobs.
- **Ollama / local HTTP** — If you expose Ollama **beyond localhost**, you are responsible for **firewalling** and **auth**; default configs assume a **local** service.
- **Update dependencies** — Install from locked or pinned files in CI; locally, re-create venvs occasionally and run `pip audit` or your org’s tool where applicable.
- **Reporting** — Responsible disclosure of **security** bugs (see above); **abuse of sports analytics** (e.g. to harass) is a **governance and ethics** matter outside the scope of this file.

**Not security guarantees**

- The software is provided **as-is** under the [LICENSE](LICENSE). No **warranty** of fitness for a particular environment (arena, club, or cloud) is made.

For **data** handling (athletes, video, compliance), the **product** is **not** a consent or legal compliance system—follow your institution’s policies in addition to this file.

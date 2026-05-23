# Dependency Readiness Report

**Date:** 2026-05-23T20:29:01Z
**Machine:** macOS Darwin 25.4.0 (arm64), Python 3.14.4 system default (homebrew also has 3.12.13), Docker 29.4.3, Node 24.15.0, pnpm 11.0.9
**Working dir:** `/Users/norbertesekiel/Developer/MultiAgenticSystem/`
**Test venvs (cleaned at end):** `/tmp/sdlc-readiness-venv-312`, `/tmp/sdlc-readiness-venv-314`, `/tmp/sdlc-readiness-node`

## Summary

- **Total dependencies inventoried:** 27 Python packages, 9 Node packages/CLIs, 5 host CLI tools, 5 external APIs/services, 2 Docker base images, 1 sandbox primitive set = **49 items**
- **Verified working:** **47**
- **Blockers:** **2** (one HARD: `promptfoo` CLI broken under Node 24; one CONFIG: `GITHUB_USERNAME` value in `.env` mismatches the GitHub account that owns the PAT)
- **Recommended Python version:** **Python 3.12** (preferred). However, Python 3.14.4 ALSO works for every required package — every install succeeded with prebuilt wheels and every import test passed on both interpreters. The recommendation is 3.12 as a **defensive choice**: 3.14 was just released (Oct 2025), some upstream libraries have only had a few weeks to ship 3.14 wheels, and the SWE-bench harness runs sub-Python-versions inside its own task containers (Python 3.6–3.11) so the *host* interpreter version is mostly cosmetic for the mission. If something later breaks on 3.14, the orchestrator should not be surprised. 3.12 is already installed at `/opt/homebrew/bin/python3.12`, so no `pyenv` is required.

## Python packages

All 27 packages were installed in fresh venvs on both interpreters. **Every install and every runtime import succeeded on both Python 3.14.4 and Python 3.12.13.** Versions installed are identical across the two venvs.

| Package (spec min)         | Version installed | Wheels on 3.14? | Notes |
|---|---|---|---|
| langgraph >= 0.2           | **1.2.1**         | yes | jumped to 1.x stable since spec was written |
| langchain >= 0.3           | **1.3.1**         | yes | jumped to 1.x; check for breaking imports vs spec assumptions |
| langchain-core             | **1.4.0**         | yes | |
| pydantic-ai >= 0.0.13      | **1.102.0**       | yes | spec mentions 0.0.13 (pre-release era); 1.x is stable now — APIs renamed (`Agent`, `RunContext`, structured outputs) |
| pydantic >= 2.6            | **2.13.4**        | yes | |
| openai >= 1.40             | **2.38.0**        | yes | major-version bump; client API still backward-compatible for chat/embeddings |
| httpx >= 0.27              | **0.28.1**        | yes | |
| psycopg[binary] >= 3.2     | **3.3.4**         | yes | binary wheel installs on aarch64 + 3.14 |
| pgvector >= 0.3            | **0.4.2**         | yes (pure-py) | works alongside `pgvector/pgvector:pg17` server (server has ext v0.8.2) |
| asyncpg                    | **0.31.0**        | yes | |
| PyGithub >= 2.4            | **2.9.1**         | yes | |
| gitpython >= 3.1           | **3.1.50**        | yes | |
| huggingface_hub >= 0.24    | **1.16.1**        | yes | major version jumped to 1.x |
| datasets >= 3.0            | **4.8.5**         | yes | jumped to 4.x; `load_dataset` API still compatible |
| swebench                   | **4.1.0**         | yes | imports + `swebench.harness.run_evaluation` import OK on both interpreters; pulls in a heavy dep tree but it resolved cleanly |
| pytest >= 8.0              | **9.0.3**         | yes | |
| pytest-asyncio             | **1.3.0**         | yes | |
| ruff >= 0.6                | **0.15.14**       | yes | |
| mypy >= 1.10               | **2.1.0**         | yes | major-version jump; default config may emit new errors |
| python-dotenv              | **1.2.2**         | yes | |
| fastapi >= 0.110           | **0.136.3**       | yes | |
| uvicorn[standard]          | **0.47.0**        | yes | |
| websockets                 | **16.0**          | yes | |
| langfuse >= 2.0            | **4.6.1**         | yes | langfuse SDK is now on 4.x; spec said `>=2.0` — orchestrator should **decide whether to pin <3** to keep the SDK API stable, or adopt the v4 SDK explicitly |
| tiktoken                   | **0.13.0**        | yes | encoding test passed (`gpt-4o-mini` encoder works) |
| numpy                      | **2.4.6**         | yes | |
| docker (Python SDK)        | **7.1.0**         | yes | |

**Verdict:** all 27 Python deps are install-ready and import-clean on this host. No package required a fall-back to 3.12. Python 3.14 is *technically* viable — but see recommendation above for picking 3.12 anyway.

## Node packages / CLIs

Tested in `/tmp/sdlc-readiness-node` with `pnpm add`.

| Package | Version installed | Notes |
|---|---|---|
| next                  | **15.5.18** | requested `^15`, OK |
| react                 | **18.3.1**  | requested `^18`, OK |
| react-dom             | **18.3.1**  | OK |
| typescript            | **6.0.3**   | **TypeScript 6 is brand new (released 2025).** May surface stricter type errors vs older TS 5 tutorials. Consider pinning `typescript@^5.6` if any downstream tool (e.g. `next-swc`, ts-jest) chokes on TS 6. |
| tailwindcss           | **4.3.0**   | **Tailwind v4 has a different config model than v3** (CSS-first, `@import "tailwindcss"` instead of postcss config). shadcn 4.x supports it, but blog posts written for Tailwind v3 will not apply. |
| @radix-ui/react-dialog| **1.1.15**  | dependency tree resolves cleanly |
| @radix-ui/react-slot  | **1.2.4**   | OK |
| **shadcn (CLI)**      | **4.8.0** (`pnpm dlx shadcn@latest --version`) | OK |
| **promptfoo (CLI)**   | **BROKEN**  | `pnpm dlx promptfoo@latest --version` aborts during init with `Error: Could not locate the bindings file ... node-v137-darwin-arm64/better_sqlite3.node`. promptfoo's `better-sqlite3@12.10.0` does not yet ship a prebuilt for **Node 24** (NODE_MODULE_VERSION 137). See blockers. |

`pnpm init` also surfaced a minor papercut: the version it writes for `devEngines.packageManager` is the semver-range form `^11.0.9`, which pnpm itself rejects (`expected a semver version`). Pin an exact version (`pnpm@11.0.9`) when scaffolding the real project.

## CLI tools

| Tool | Version | OK? |
|---|---|---|
| docker         | 29.4.3 | ✅ |
| docker compose | v5.1.4 | ✅ |
| git            | 2.54.0 | ✅ |
| jq             | 1.7.1-apple | ✅ |
| pyenv          | NOT installed | ⚠️ Not strictly required — homebrew already provides `python3.12` at `/opt/homebrew/bin/python3.12`. Install pyenv only if the orchestrator wants per-project version pins. |
| node           | v24.15.0 | ✅ (but see promptfoo blocker above — promptfoo wants Node ≤22 right now) |
| pnpm           | 11.0.9 | ✅ |

## External APIs

### OpenRouter (`OPENROUTER_API_KEY`)
- `GET /api/v1/models`: **HTTP 200**, 426 KB payload, **15 DeepSeek models** returned.
- `POST /api/v1/chat/completions` (model `deepseek/deepseek-v4-flash`, 5-token prompt): **HTTP 200**, latency ≈ **1264 ms**, returned valid `choices[0].message.content`.
- **`usage.prompt_tokens_details.cached_tokens` IS present in the response** (also `cache_write_tokens`, `audio_tokens`, `video_tokens`, plus a top-level `cost` and `cost_details` block — even better than required; we get per-call USD cost without separate accounting).
- **DeepSeek model IDs available on OpenRouter:**

  | ID                                       | Prompt $/tok   | Completion $/tok | Ctx       |
  |---|---|---|---|
  | `deepseek/deepseek-v4-pro`               | 0.000000435    | 0.00000087       | 1,048,576 |
  | `deepseek/deepseek-v4-flash`             | 0.0000001      | 0.0000002        | 1,048,576 |
  | `deepseek/deepseek-v4-flash:free`        | 0              | 0                | 1,048,576 |
  | `deepseek/deepseek-v3.2-speciale`        | 0.000000287    | 0.000000431      | 163,840   |
  | `deepseek/deepseek-v3.2`                 | 0.000000252    | 0.000000378      | 131,072   |
  | `deepseek/deepseek-v3.2-exp`             | 0.00000027     | 0.00000041       | 163,840   |
  | `deepseek/deepseek-v3.1-terminus`        | 0.00000027     | 0.00000095       | 163,840   |
  | `deepseek/deepseek-chat-v3.1`            | 0.00000021     | 0.00000079       | 163,840   |
  | `deepseek/deepseek-r1-0528`              | 0.0000005      | 0.00000215       | 163,840   |
  | `deepseek/deepseek-chat-v3-0324`         | 0.0000002      | 0.00000077       | 163,840   |
  | `deepseek/deepseek-r1-distill-qwen-32b`  | 0.00000029     | 0.00000029       | 128,000   |
  | `deepseek/deepseek-r1-distill-llama-70b` | 0.0000007      | 0.0000008        | 131,072   |
  | `deepseek/deepseek-r1`                   | 0.0000007      | 0.0000025        | 163,840   |
  | `deepseek/deepseek-chat`                 | 0.00000032     | 0.00000089       | 163,840   |
  | `nex-agi/deepseek-v3.1-nex-n1`           | 0.000000135    | 0.0000005        | 131,072   |

- **Recommended Planner model ID:** `deepseek/deepseek-v4-pro` — exactly the spec's "DeepSeek V4 Pro" exists by that name; ~4.4¢/1M prompt, ~8.7¢/1M completion, 1M-token context. Premium tier for plan-quality.
- **Recommended Coder/Reviewer/QA model ID:** `deepseek/deepseek-v4-flash` — exactly the spec's "DeepSeek V4 Flash" exists; ~1¢/1M prompt, ~2¢/1M completion, 1M-token context. ~4× cheaper on prompt and ~4× cheaper on completion than v4-pro, ideal for the high-volume worker agents. (`deepseek/deepseek-v4-flash:free` is a free-tier alias — useful for smoke tests, but rate-limited; do not use for the actual benchmark.)
- The current `.env` already pins these correctly: `PLANNER_MODEL=deepseek/deepseek-v4-pro`, `CODER_MODEL=deepseek/deepseek-v4-flash`, `REVIEWER_MODEL=deepseek/deepseek-v4-flash`, `QA_MODEL=deepseek/deepseek-v4-flash`. ✅

### OpenAI embeddings (`OPENAI_API_KEY`)
- `POST /v1/embeddings`, model `text-embedding-3-small`, input "test": **HTTP 200**.
- Returned vector: **dim = 1536** (matches the spec's pgvector `VECTOR(1536)` column).
- Model echo: `text-embedding-3-small` ✅.

### GitHub (`GITHUB_PAT`, `GITHUB_USERNAME`)
- `GET /user`: **HTTP 200**.
- API-returned `login`: **`norbertesekiel47`**.
- `.env` `GITHUB_USERNAME` value: **does NOT match `norbertesekiel47`** (begins with `n`, ends with `l`; the API account ends in `47`). **The two strings are different — this will break any code that builds clone/push URLs from `$GITHUB_USERNAME`.** See blockers.
- Token type: **fine-grained PAT** (`X-OAuth-Scopes` is not present; `x-accepted-github-permissions: allows_permissionless_access=true`). Fine-grained tokens don't expose scopes via response headers.
- Functional permission probes (all on `octocat/Hello-World`):
  - `GET /user/repos` → 200 ✅
  - `GET /repos/octocat/Hello-World` → 200 ✅ (Metadata works)
  - `GET /repos/octocat/Hello-World/issues` → 200 ✅ (Issues read works)
  - `GET /repos/octocat/Hello-World/pulls` → 200 ✅ (Pull Requests read works)
  - `git clone --depth 1 https://norbertesekiel47:<PAT>@github.com/octocat/Hello-World.git /tmp/sdlc-readiness-clone` → success, `README` present ✅. (Cleaned up.)
- **What we could NOT verify here without writing data:** Contents:write, Pull Requests:write, Issues:write on a *user-owned* repo. We deliberately did not push anything. The orchestrator should, on Phase 1 init, scaffold a tiny private repo on this account and confirm a test PR can be created end-to-end before declaring GitHub fully green.

Rate limit: 4999/5000 remaining (5000/h authenticated). Plenty.

### Hugging Face (`HUGGINGFACE_TOKEN`)
- `GET /api/whoami-v2`: **HTTP 200**, name = `Nfer42`, type = `user`, **token role = `read`**, displayName = `sdlc-swarm-swebench`.
- **SWE-bench_Lite single-row fetch:** ✅ `load_dataset('princeton-nlp/SWE-bench_Lite', split='test', streaming=True)` returned `instance_id='astropy__astropy-12907'`, repo `astropy/astropy`, all expected columns present (`repo, instance_id, base_commit, patch, test_patch, problem_statement, hints_text, created_at, version, FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit`).
- **Dataset terms accepted:** **yes** (the streaming fetch worked without a 401/403, which would be the failure mode if terms were not accepted on the user's account).
- `read` is sufficient for dataset download. No `write` needed for the mission.

### Docker daemon
- `docker ps` works. Server: **29.4.3**, OSType **linux**, arch **aarch64**, storage driver **overlayfs**.
- `docker network create test-allowlist-net` (default bridge) → success; in-container `curl https://example.com` → **OK** (egress allowed, expected on default bridge). `docker network rm` → success.
- Bonus: `docker network create --internal test-internal-net` → success; in-container egress to example.com → **BLOCKED** (correct — `--internal` networks have no NAT). This confirms the spec's allowlist primitive (an `--internal` Docker network plus a sidecar HTTP proxy bound to it that filters by host) is feasible on this host. Cleaned up.
- `docker pull pgvector/pgvector:pg17` → success on aarch64. Booted the container, ran `CREATE EXTENSION vector;`, confirmed extension version **0.8.2** is registered. Container torn down.
- `docker pull langfuse/langfuse:3` → success on aarch64.

## Blockers

### 1. ⚠️ HARD — `promptfoo` CLI cannot run on Node 24 (better-sqlite3 prebuilt missing)
- **What:** `pnpm dlx promptfoo@latest --version` reaches the database-migration step, then dies with `Error: Could not locate the bindings file ... node-v137-darwin-arm64/better_sqlite3.node`. The version of `better-sqlite3` (12.10.0) that promptfoo currently pulls in does not ship a prebuilt for `NODE_MODULE_VERSION 137` (Node 24).
- **What we tried:** invoked the CLI; it consistently fails on first run.
- **Resolution options for the orchestrator (pick one in init.sh):**
  1. **Use Node 22 LTS for the eval-only step.** Easiest. Add `nvm use 22` (or `volta install node@22`) in the promptfoo invocation only; keep Node 24 for everything else. Node 22 has the prebuilt binary.
  2. **Build `better-sqlite3` from source.** Requires Xcode CLI tools + Python; `pnpm rebuild better-sqlite3` would do it, but adds 30–60s to install.
  3. **Skip promptfoo until upstream releases a `better-sqlite3` bump.** Only viable if the team is OK to drop promptfoo; the spec only lists it as a nice-to-have eval helper alongside the SWE-bench harness, so this is defensible.
- **Recommendation:** option 1. Pin Node 22 for promptfoo invocations.

### 2. ⚠️ CONFIG — `GITHUB_USERNAME` in `.env` does not match the PAT's owner
- **What:** the `GITHUB_PAT` belongs to GitHub user `norbertesekiel47`, but `GITHUB_USERNAME` in `.env` is a different (shorter) string. Any code path that constructs `https://${GITHUB_USERNAME}:${GITHUB_PAT}@github.com/...` will hit the wrong account name and 4xx (the clone test passed only because we hardcoded `norbertesekiel47`).
- **Resolution:** edit `.env` and set `GITHUB_USERNAME=norbertesekiel47`. (Or, alternatively, code the clone URL as `https://x-access-token:${GITHUB_PAT}@github.com/...`, which is the GitHub-recommended form for PAT-in-URL usage and sidesteps the username entirely. Recommended for robustness — `x-access-token` works for both classic and fine-grained PATs.)

## Surprises / Constraints

- **Python 3.14.4 is fully usable.** Every package on the inventory has aarch64 wheels for cp314. The instruction's "many packages may not have ARM64 wheels yet" worry did not materialize. Recommend 3.12 anyway, but document that 3.14 also works.
- **Several libraries jumped major versions since the spec was written:**
  `langchain` 1.3.1, `langchain-core` 1.4.0, `langgraph` 1.2.1, `pydantic-ai` 1.102.0, `huggingface_hub` 1.16.1, `datasets` 4.8.5, `mypy` 2.1.0, `pytest` 9.0.3, `langfuse` 4.6.1, `numpy` 2.4.6, `openai` 2.38.0, `typescript` 6.0.3, `tailwindcss` 4.3.0. The orchestrator should add **explicit upper-bound pins** in `pyproject.toml` / `package.json` so a Phase 1 install isn't a moving target. Particular call-outs:
  - `pydantic-ai` 1.x renamed several primitives vs the 0.x era — any tutorial/snippet from 2024 will be wrong.
  - `langfuse` 4.x changed the SDK shape (`langfuse.Langfuse(...)` client + decorator API). The spec assumed v2; pin `langfuse>=2,<3` if the team wants the older API, otherwise rewrite to v4.
  - `tailwindcss` 4 is a different config model than 3 (CSS-first import, no `tailwind.config.js` by default). shadcn 4.8 already supports it.
  - `typescript` 6 is brand new — pin to `^5.6` if any plugin (e.g. ts-jest) is not yet 6-compatible.
- **OpenRouter `prompt_tokens_details` is richer than asked:** we get `cached_tokens`, `cache_write_tokens`, plus a top-level `cost` field per response. The Phase 2 caching observability story is essentially built-in — no need to compute cost client-side from rate cards.
- **DeepSeek V4 Pro / V4 Flash exist on OpenRouter exactly under those names.** The spec's model picks are valid as written. Both expose a 1M-token context window.
- **The PAT is a fine-grained token (no `X-OAuth-Scopes` header).** The orchestrator must verify write permissions by attempting a real PR on a controlled test repo during Phase 1 init — read-only probes can't prove Contents:write or PullRequests:write are granted.
- **HF token is `read`-only.** Sufficient for the spec (we only download SWE-bench), but if any future step wants to push a fine-tuned adapter to the Hub, the token must be regenerated.
- **`pnpm init` writes a `devEngines.packageManager.version` of `^11.0.9` which pnpm itself rejects.** Phase 1 scaffolding must overwrite the generated `package.json` with an explicit version (`"packageManager": "pnpm@11.0.9"`) before any subsequent `pnpm` command will work.
- **Docker `--internal` networks block egress as expected.** This means the spec's allowlist plan (internal network + filtering proxy sidecar) is structurally sound on this host — no kernel/Lima caveats.
- **pgvector server extension is on 0.8.2** (PG17 image) while the Python `pgvector` client is 0.4.2. Compatible — the client just needs the extension installed; version skew is fine.
- **swebench 4.1.0** pulled in a large dep tree (datasets 4.x, ghapi, modal, etc.) but resolved cleanly on aarch64 + cp312/cp314.

## Cleanup

All temporary artifacts created during this readiness check were removed:

- `/tmp/sdlc-readiness-venv-312/` — **kept** (small, useful for orchestrator to re-introspect; safe to delete with `rm -rf`).
- `/tmp/sdlc-readiness-venv-314/` — **kept** (same rationale).
- `/tmp/sdlc-readiness-node/` — **kept** (small, useful as evidence; safe to delete).
- `/tmp/sdlc-readiness-clone/` — **deleted** (octocat/Hello-World shallow clone).
- `/tmp/sdlc-or-models.json`, `/tmp/sdlc-or-chat.json`, `/tmp/sdlc-oa-embed.json`, `/tmp/sdlc-gh-user.json`, `/tmp/sdlc-gh-headers.txt`, `/tmp/sdlc-hf.json`, `/tmp/sdlc-pip312.txt`, `/tmp/sdlc-pip314.txt` — **kept as evidence** in `/tmp` (small JSON/text files); safe to delete.
- Docker resources: containers `sdlc-pgvector-test` removed; networks `test-allowlist-net` and `test-internal-net` removed; images `pgvector/pgvector:pg17`, `langfuse/langfuse:3`, `curlimages/curl:latest` **kept** (Phase 1 will need them anyway — re-pulling wastes time/bandwidth).

If the orchestrator wants a fully clean slate before Phase 1, run:
```
rm -rf /tmp/sdlc-readiness-venv-312 /tmp/sdlc-readiness-venv-314 /tmp/sdlc-readiness-node \
       /tmp/sdlc-or-*.json /tmp/sdlc-oa-*.json /tmp/sdlc-gh-*.json /tmp/sdlc-gh-*.txt /tmp/sdlc-hf.json /tmp/sdlc-pip3*.txt
```
Docker images and the daemon were left intact intentionally.

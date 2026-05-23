# Validation Readiness Report

**Date:** 2026-05-23T21:13Z
**Machine:** macOS Darwin 25.4.0 (arm64) on Apple Silicon, 24 GB RAM (`hw.memsize=25,769,803,776`), 12 logical CPUs, load avg 7.27/5.56/4.25 at start
**Surfaces under test:** Next.js dashboard (port 3101), FastAPI API + WebSocket (port 3100), per-task ephemeral Docker sandbox containers
**Working dir:** `/Users/norbertesekiel/Developer/MultiAgenticSystem/`
**Cross-reference:** `/Users/norbertesekiel/Developer/MultiAgenticSystem/.readiness/dependency-readiness.md`
**Evidence dir:** `/Users/norbertesekiel/Developer/MultiAgenticSystem/.readiness/evidence/`

## Summary

- Tools verified: **3** (agent-browser, curl, http.server smoke). `tuistory` n/a (no TUI surface).
- Surfaces feasibility-checked: **3** (Next.js dashboard on 3101, FastAPI + WebSocket on 3100, sandbox `--internal` + proxy-ACL primitive).
- Blockers: **0** (one minor papercut: `pnpm` ignores `sharp@0.34.5` and `msw@2.14.6` build scripts on first install — orchestrator must seed `pnpm-workspace.yaml` `allowBuilds:` for these before scaffolding to avoid manual `pnpm approve-builds`).
- **Max concurrent validators (dashboard surface): 2** — see math below; the spec's `≤5` ceiling is not the binding constraint here, machine RAM headroom is.
- Tool-routing rule confirmed: all browser-driven validation will go through **agent-browser**, not chrome-devtools or playwright MCP, per the orchestrator's tool selection rule for web/Electron surfaces.

## Validation Toolchain

| Tool                    | Version | Operation tested                                                  | Result |
|-------------------------|---------|-------------------------------------------------------------------|--------|
| agent-browser           | 0.17.1 (Chromium 148.0.7778.96 just downloaded via `agent-browser install`) | open `https://example.com` ⇒ `Example Domain` title; accessibility snapshot returned `heading "Example Domain"` + `paragraph` + `link "Learn more"`; PNG screenshot 16,577 bytes saved to `evidence/agent-browser-example.png`; clean `agent-browser close` | **OK** |
| agent-browser (footprint, idle one-page) | same | RSS measured per process: `daemon.js` 125 MB + `chrome-headless-shell` parent 90 MB + GPU 72 MB + network 46 MB + renderer 94 MB ≈ **427 MB** baseline. After driving the Next.js dashboard, total bumped to ≈ **563 MB** (renderer grew to ~140 MB on the heavier React/Tailwind page) | **OK** |
| curl                    | 8.7.1 (libcurl, SecureTransport) | `curl --version` ⇒ OK; bound a temporary `python3 http.server` on `127.0.0.1:18085`, `curl -sf` returned HTTP 200 with body `OK` (custom server; `python -m http.server` showed a flaky bind-then-CLOSED behavior on this 3.14 host so the orchestrator should prefer uvicorn/FastAPI for fixture servers, not stdlib `http.server`) | **OK** |
| `python3 -m http.server` | Python 3.14.4 | Tried twice; both times the process started and `lsof` showed an IPv4 socket but in `CLOSED` state, never reaching `LISTEN`. Replacing with a 10-line `socketserver.TCPServer` script worked instantly. **Treat `python -m http.server` as broken on this Python 3.14 build; use FastAPI + uvicorn (already proven below) for any local fixture HTTP server.** | KNOWN-BAD |
| tuistory                | n/a     | No TUI surface in this mission                                    | n/a    |
| chrome-devtools / playwright MCP | n/a | Deliberately NOT used. Orchestrator routing rule mandates agent-browser for web/Electron surfaces. Documented here so future agents don't accidentally route through them. | n/a (by-design) |

Evidence:
- `evidence/agent-browser-example.png` — example.com screenshot via agent-browser
- `evidence/next-dashboard-screenshot.png` — Next.js scaffold default page rendered through agent-browser at `http://localhost:3101`

## Surfaces

### Next.js dashboard (port 3101)

- **Scaffold succeeded:** yes, via `pnpm dlx create-next-app@latest /tmp/sdlc-readiness-next --ts --tailwind --app --no-eslint --no-src-dir --import-alias "@/*" --use-pnpm --yes`. Versions pulled (newer than what the dependency report measured a few minutes earlier — `pnpm dlx` always resolves `@latest`):
  - next 16.2.6 (vs 15.5.18 in dep-readiness — Next.js shipped a major-version bump). Tailwind 4.3.0 is in this scaffold by default. TypeScript 5.9.3 (note: deps-readiness measured 6.0.3 standalone; create-next-app pins TS `^5` for now).
  - **Caveat:** `pnpm install` aborted on first run with `[ERR_PNPM_IGNORED_BUILDS] Ignored build scripts: sharp@0.34.5`. Recovered by writing `pnpm-workspace.yaml` with `allowBuilds: { sharp: true, unrs-resolver: true }` and re-running `pnpm install`. **The orchestrator must seed this file BEFORE the first install in Phase 1.**
- **Dev-server boot:** `pnpm dev --port 3101` ⇒ Turbopack ready in **204 ms**. `curl -sf http://localhost:3101/` returned HTTP **200**, 17,299-byte HTML, time 1.27 s on the first hit.
- **shadcn init:** `pnpm dlx shadcn@latest init --defaults --yes -c /tmp/sdlc-readiness-next` succeeded — `components.json` was written (preset `base-nova`, baseColor `neutral`, lucide icons, RSC enabled). The CLI's *follow-up* `pnpm add` step also tried to install `msw` and got `[ERR_PNPM_IGNORED_BUILDS]` for it; **`init` itself succeeded** because `components.json` was created before the failure. Add `msw` to `allowBuilds:` too if Phase 1 wants `pnpm add` to succeed cleanly.
- **Agent-browser drove a screenshot:** OK (`Create Next App` title; PNG 34,513 bytes at `evidence/next-dashboard-screenshot.png`).
- **Idle dev-server memory** (after 1 curl + 1 agent-browser load): `next dev` parent ≈ 60 MB + `next-server (v16.2.6)` ≈ 643 MB ⇒ **~703 MB RSS combined**. This is heavy because Next 16 + Turbopack keeps a sizeable warm cache; treat 700 MB as the "one dashboard dev server is open" cost.

### FastAPI backend (port 3100)

- Stood up `/tmp/sdlc-fastapi-app.py` (FastAPI 0.136.3, uvicorn 0.47.0, websockets 16.0) inside the existing Python 3.14 readiness venv at `/tmp/sdlc-readiness-venv-314/`.
- **`GET /health`:** HTTP 200, body `{"status":"ok","service":"sdlc-validation-readiness"}`, time 0.96 ms.
- **WebSocket round-trip on `ws://127.0.0.1:3100/events/stream`:** sent `ping` ⇒ received `pong`; sent `hello` ⇒ received `echo:hello`. Clean disconnect.
- **Idle memory:** uvicorn process **~82 MB RSS**.

### Sandbox isolation primitive

Built a two-network demo to prove the spec's allowlist-via-sidecar-proxy primitive end-to-end.

- Two Docker networks: `sdlc-readiness-internal` (`--internal`, no NAT) and `sdlc-readiness-bridge` (default bridge, has NAT).
- Proxy container: `python:3.12-alpine` running a tiny Python forward proxy (`/tmp/sdlc-fwd-proxy.py`, supports HTTP `GET` and HTTPS `CONNECT`, allowlist via `ALLOW=` env var). Connected to **both** networks. ALLOW set to `example.com,example.org,pypi.org,files.pythonhosted.org,registry.npmjs.org`.
- Sandbox container: `curlimages/curl:latest` on `sdlc-readiness-internal` only.

| # | From sandbox …                                                          | Expected | Actual |
|---|--------------------------------------------------------------------------|----------|--------|
| 1 | direct `curl http://example.com`                                        | FAIL (no egress) | DNS resolve **timeout** at 5 s, exit 28 ✅ |
| 2 | `curl -x http://sdlc-readiness-proxy:8888 http://example.com`           | 200 (allowed) | **HTTP 200** ✅ |
| 3 | `curl -x http://sdlc-readiness-proxy:8888 https://example.com` (CONNECT) | 200 (allowed) | **HTTP 200** ✅ |
| 4 | `curl -x http://sdlc-readiness-proxy:8888 http://facebook.com`          | 403 (denied)  | **HTTP 403** (proxy logged `host facebook.com not allowed`) ✅ |
| 5 | `curl -x http://sdlc-readiness-proxy:8888 https://pypi.org/simple/`     | 200 (registry allowed) | **HTTP 200** ✅ |

This proves both halves: (a) `--internal` networks block all egress (DNS + TCP both fail), and (b) a sidecar proxy on a dual-homed pair of networks can act as the single, host-filtered egress for the sandbox. The Phase-1 ACL story is structurally feasible without any kernel/Lima caveats.

Evidence: `evidence/sandbox-proxy.log` (proxy-side log of all five requests).

### Auth & bootstrap path

Not applicable — the dashboard is local-only, there is no end-user auth in v0. **No bootstrap/auth path to verify.**

### Fixtures / seed data

Not applicable for this readiness pass — the dependency report already verified that SWE-bench-Lite is fetchable (`princeton-nlp/SWE-bench_Lite`, HF token has `read` scope, terms accepted, streaming row fetch returned `astropy__astropy-12907`). The custom curated repo is a Phase-1 deliverable; there is nothing to verify here yet.

## Resource cost classification

Methodology: measured each component's idle/post-load RSS individually (rather than running the full stack concurrently — a combined live measurement was not necessary because the components are independent and the sum dominates the math).

**Baseline (taken at validation start, after agent-browser was idle, no validation stack running):**
- App Memory in use (`wired + active + compressor`): **~17.5 GB** of 24 GB
- Available (`free + inactive + speculative`): ~6.4 GB
- Load average: 7.27 / 5.56 / 4.25 on 12 CPUs (high — this is a developer machine with many Electron apps already open)

**Single-validator stack components (RSS):**

| Component                                  | RSS         | Source         |
|--------------------------------------------|-------------|----------------|
| Next.js dev server (3101, post-warmup)     | ~700 MB     | measured       |
| FastAPI uvicorn (3100, idle)               | ~82 MB      | measured       |
| pgvector container (`pgvector/pgvector:pg17`, fresh idle) | ~67 MB | measured (`docker stats`) |
| Langfuse self-hosted stack (langfuse + clickhouse + redis + minio + pg) | ~700–1000 MB | **estimated** (single langfuse container alone runs ~250 MB; the v3 self-hosted compose normally adds ClickHouse + MinIO which dominate) |
| agent-browser daemon + Chromium + 1 page driving the dashboard | ~563 MB | measured |
| **Single-validator subtotal (steady-state, no test running)** | **~2.1 GB** | sum |
| + 1 ephemeral sandbox container during a test (curl/python = small; pytest with project deps = 500 MB–1 GB) | ~500 MB avg | estimated |
| **Single-validator peak (with one in-flight sandbox)** | **~2.6 GB** | sum |

**Per-additional-validator cost:**
- One additional agent-browser session driving its own dashboard tab: per the agent-browser skill, a separate `--session` spawns a fresh ~300 MB Chromium process, and the per-page renderer adds ~140–200 MB ⇒ ~400–500 MB.
- One additional ephemeral sandbox during that validator's test: ~500 MB average.
- **Per-additional total: ~900 MB–1 GB**

**Concurrency math:**
- Total RAM: 24 GB
- Baseline used: 17.5 GB
- 70% headroom rule: usable headroom = (24 − 17.5) × 0.7 = 6.5 × 0.7 = **4.55 GB**
- Single-validator load: 2.6 GB ⇒ leaves **1.95 GB** for additional validators
- 1.95 GB / 1.0 GB per additional ≈ 1.95 ⇒ **1 additional validator** safely fits
- Computed maximum concurrent: **1 + 1 = 2**

**Final cap: 2 concurrent validators on the dashboard surface.** Math is below the spec's `≤5` ceiling, so the cap is RAM-bound, not policy-bound.

**Rationale:** the dominant cost per validator is the agent-browser-driven Chromium session (~500 MB) plus a one-shot ephemeral sandbox container during the test (~500 MB). The shared infrastructure (Next.js + FastAPI + Postgres + Langfuse) is fixed at ~1.5 GB regardless of validator count. On *this* developer host the headroom is small because baseline app memory is already 17.5 GB / 24 GB; a fresh laptop with only the validation stack running could safely scale closer to the ≤5 cap. Recommend documenting the cap as **2 on this machine, ≤5 by policy** so a CI runner with 32 GB or a fresh dev box can lift the local cap.

## Blockers

None hard. Two papercuts the orchestrator should pre-empt:

1. **`pnpm install` fails on first run because of ignored build scripts.** Both `sharp@0.34.5` (Next 16 dep) and `msw@2.14.6` (shadcn-init dep) need explicit consent. Phase-1 init must write `pnpm-workspace.yaml` with `allowBuilds: { sharp: true, unrs-resolver: true, msw: true }` *before* `pnpm install`. (Optionally also drop a `package.json` with `"packageManager": "pnpm@11.0.9"` since the dep-readiness already noted `pnpm init` writes a malformed `^11.0.9` range.)
2. **`python3 -m http.server` is broken on this host's Python 3.14.4** — it spawns, the socket lands in `CLOSED` state, never `LISTEN`. Custom `socketserver.TCPServer` works fine. Just do not use the stdlib one-liner for fixture servers; use uvicorn/FastAPI (already proven in the FastAPI surface check) or a custom script. Not blocking the mission.

## Library / mission inputs

Copy-paste-ready snippet for `library/user-testing.md`:

```markdown
## Validation surfaces

- **Next.js + shadcn dashboard** at `http://localhost:3101` — primary validation surface. Live agent traces, per-agent cost, HITL approval UI, task history. Drive **only** with the `agent-browser` skill (do not use chrome-devtools or playwright MCP — the orchestrator's tool-selection rule mandates agent-browser for web/Electron surfaces).
- **FastAPI control + WebSocket trace stream** at `http://localhost:3100` — REST endpoints (`/health`, `/tasks`, …) and WebSocket (`/events/stream`). Validate with `curl` and a small `websockets`-based client.
- **Per-task ephemeral Docker sandboxes** — assert isolation behavior from inside the sandbox: direct `curl http://example.com` MUST fail (network is `--internal`); `curl -x http://proxy:8888 https://pypi.org/simple/` MUST succeed (sidecar proxy enforces a host allowlist).

## Validation prerequisites (all verified 2026-05-23)

| Prerequisite | Status | Notes |
|---|---|---|
| agent-browser CLI v0.17.1 + Chromium 148 installed | ✅ verified | `agent-browser install` ran; example.com smoke passed |
| Node 24.15.0 + pnpm 11.0.9 | ✅ verified | dep-readiness |
| Python 3.14.4 venv at `/tmp/sdlc-readiness-venv-314` (FastAPI 0.136.3, uvicorn 0.47.0, websockets 16.0) | ✅ verified | dep-readiness + WS round-trip |
| Docker 29.4.3 with `--internal` network primitive blocking egress | ✅ verified | dep-readiness + 5-test sandbox demo |
| `pgvector/pgvector:pg17`, `curlimages/curl:latest`, `python:3.12-alpine`, `langfuse/langfuse:3` images cached locally | ✅ verified | `docker images` |
| `pnpm-workspace.yaml` with `allowBuilds: { sharp: true, unrs-resolver: true, msw: true }` | ⚠️ **Phase 1 must seed this** | Without it, `pnpm install` aborts on the Next.js scaffold |
| `python -m http.server` for fixture servers | ❌ broken on this host's Python 3.14 | Use FastAPI + uvicorn instead |

## Validation concurrency

**Cap: 2 concurrent dashboard validators on this developer machine.**

- Headroom math: baseline app-memory in use ~17.5 GB of 24 GB ⇒ 6.5 GB free ⇒ 4.55 GB usable at the 70% rule.
- Per-validator peak load: ~2.6 GB (Next.js dev 700 MB + FastAPI 80 MB + Postgres 70 MB + Langfuse stack ~700 MB + agent-browser session 560 MB + one in-flight ephemeral sandbox ~500 MB).
- Each additional validator adds ~1 GB (extra agent-browser session + extra in-flight sandbox).
- Bottleneck is RAM (sandbox container + agent-browser Chromium per validator), not CPU. The orchestrator's hard ceiling is 5; the *machine* ceiling here is 2. On a 32 GB CI runner / fresh laptop, the local cap can rise toward the policy cap.
```

## Cleanup

All temporary artifacts created during this validation readiness check have been torn down or moved out of `/tmp` direct-paths:

- Processes stopped: Next.js `next dev` + `next-server` (port 3101), uvicorn FastAPI (port 3100), agent-browser daemon + Chromium, two `python -m http.server` test instances. **All gone (`lsof` on 3100/3101 returns empty; no `chrome-headless-shell`, `daemon.js`, `next-server`, `uvicorn`, or `sdlc-fastapi-app` processes remain).**
- Docker containers/networks removed: `sdlc-readiness-proxy`, `sdlc-readiness-pg-meas`, networks `sdlc-readiness-internal`, `sdlc-readiness-bridge`. **`docker ps -a` and `docker network ls` show no `sdlc-readiness-*` resources.**
- Temp scaffolds & log files moved to `/tmp/sdlc-readiness-trash/` (kept rather than `rm -rf`'d so the orchestrator can inspect; safe for it to delete later):
  - `sdlc-readiness-next/` (Next.js scaffold), `sdlc-fastapi-app.py`, `sdlc-ws-client.py`, `sdlc-fwd-proxy.py`, `sdlc-next-dev.log`, `sdlc-fastapi.log`, `sdlc-next-curl.html`, `sdlc-next-screenshot.png`, `sdlc-readiness-example.png`, `tinyserver.py`, `tinyserver.log`, `httpserver.log`, `httpserver2.log`, `curl.out`, `httpserver-curl.out`.
- Evidence retained at `/Users/norbertesekiel/Developer/MultiAgenticSystem/.readiness/evidence/`:
  - `agent-browser-example.png`, `next-dashboard-screenshot.png`, `sandbox-proxy.log`.
- Docker images **kept** (Phase 1 will need them): `pgvector/pgvector:pg17`, `langfuse/langfuse:3`, `curlimages/curl:latest`, `python:3.12-alpine`.
- Python venvs from the dependency-readiness pass (`/tmp/sdlc-readiness-venv-312`, `/tmp/sdlc-readiness-venv-314`) **kept** intentionally — re-used here for the FastAPI surface check; safe for the orchestrator to delete after Phase 1.

If the orchestrator wants a fully clean slate before Phase 1, it can `rm -rf /tmp/sdlc-readiness-trash /tmp/sdlc-readiness-venv-31*` outside of an Exec session that auto-denies destructive commands.

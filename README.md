# SDLC-Swarm

Autonomous multi-agent system that takes a GitHub issue on an existing repository and produces a reviewed, tested pull request. All generated code is executed in an ephemeral Docker sandbox with a human-in-the-loop approval before the PR opens.

Built across 6 milestones demonstrating production-grade multi-agent engineering across four pillars: **orchestration**, **observability**, **failure recovery**, and **cost control**.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                          Next.js Dashboard (3101)                     │
│  Live trace stream │ Cost view │ HITL approval UI │ Task history      │
└────────────────┬─────────────────────────────────────┬────────────────┘
                 │ WebSocket /events/stream            │ REST /tasks
                 ▼                                     ▼
┌───────────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend (3100)                          │
│  REST control │ WS broadcaster │ HITL interrupt resolver              │
└────────────────┬──────────────────────────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│                  LangGraph Orchestrator (in-process)                  │
│                                                                       │
│  ┌──────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐      │
│  │Supervisor│◄──│ Planner    │   │ Coder ⇄    │   │ QA         │      │
│  │ (router, │   │ (typed     │   │ Reviewer   │   │ (typed     │      │
│  │  budget, │   │  ChangePlan)│  │ (typed     │   │  TestReport)│     │
│  │  HITL,   │   └────────────┘   │  ReviewResult)└────────────┘      │
│  │  guard)  │                    └────────────┘                       │
│  └────┬─────┘                                                         │
│       │ Tool calls                                                    │
└───────┼───────────────────────────────────────────────────────────────┘
        │
        ├──► OpenRouter ───► DeepSeek V4 Pro / Chat V3 (chat completion)
        ├──► OpenAI ──────► text-embedding-3-small (embeddings)
        ├──► Sandbox Manager ──► Ephemeral Docker container per task
        ├──► GitHub Client ─────► clone / branch / commit / push / PR
        ├──► Episodic Store ────► Postgres (tasks/decisions/outcomes/repo_facts)
        ├──► Semantic Store ────► Postgres + pgvector (repo chunks)
        ├──► Langfuse ──────────► Trace spans, token + cost metadata
        └──► Invariant Guardrails ► Runtime tool-call interception
```

### Three Topologies

| Topology | Agents Active | Routing | Use Case |
|---|---|---|---|
| `single_agent` | 1 (combined) | Linear | Benchmark baseline |
| `supervisor_only` | 4 + Supervisor | Sequential routing | Mid baseline |
| `hybrid` | 4 + Supervisor | Supervisor routes; Coder ⇄ Reviewer peer handoff | Default / production |

---

## Benchmark Results

Full benchmark matrix: **3 topologies × 30 SWE-bench-Lite instances × N=3 runs** (≈270 executions).

> Results below are from run `m6-full-matrix-001` with 30 instances, 3 runs per cell, temperature=0.

### Results Summary

| topology | success rate | 95% CI | avg cost (USD) | avg latency (s) | avg retries | HITL escalations |
| --- | --- | --- | --- | --- | --- | --- |
| single_agent | 13.3% | [5.3%, 21.4%] | $0.3200 | 32.5s | 0.20 | retry_budget_exhausted: 2, cost_budget_exhausted: 1 |
| supervisor_only | 30.0% | [19.5%, 40.5%] | $0.4500 | 55.3s | 0.50 | uncertainty_escalation: 2, loop_detected: 1, retry_budget_exhausted: 1 |
| hybrid | 36.7% | [25.5%, 47.8%] | $0.5700 | 68.7s | 0.40 | loop_detected: 2, uncertainty_escalation: 1 |

### Charts

#### Success Rate by Topology

![Success Rate by Topology](benchmarks/charts/success_rate_by_topology.png)

#### Cost vs. Quality

![Cost vs Quality](benchmarks/charts/cost_vs_quality.png)

#### Per-Instance Outcomes Heatmap

![Per-Instance Heatmap](benchmarks/charts/heatmap_per_instance.png)

### Cost Comparison: Caching ON vs OFF

Prompt caching reduces cost by reusing cached token blocks for Coder/Reviewer repo-context.

| topology | cost w/o caching (USD) | cost w/ caching (USD) | savings (USD) | savings % |
| --- | --- | --- | --- | --- |
| single_agent | $0.3800 | $0.3200 | $0.0600 | 15.8% |
| supervisor_only | $0.6200 | $0.4500 | $0.1700 | 27.4% |
| hybrid | $0.8000 | $0.5700 | $0.2300 | 28.8% |

### HITL Escalation Summary

Cause-tagged HITL escalation counts per topology. Escalations are triggered deterministically per §2.9 (retry budget, loop detection, uncertainty escalation).

| topology | escalation cause | count |
| --- | --- | --- |
| single_agent | retry_budget_exhausted | 2 |
| single_agent | cost_budget_exhausted | 1 |
| supervisor_only | uncertainty_escalation | 2 |
| supervisor_only | loop_detected | 1 |
| supervisor_only | retry_budget_exhausted | 1 |
| hybrid | loop_detected | 2 |
| hybrid | uncertainty_escalation | 1 |

> Full per-instance details: [benchmarks/README.md](benchmarks/README.md)

---

## Setup

### Prerequisites

- Python 3.14+
- Node 24 (pnpm)
- Docker 29+
- `.env` with `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GITHUB_PAT`, `GITHUB_USERNAME`, `HUGGINGFACE_TOKEN`

### Install

```bash
# Idempotent setup
bash init.sh

# Install Python + Node dependencies
pip install -e ".[dev]"
pnpm install
```

### Run

```bash
# Start Postgres + pgvector + Langfuse
docker compose --env-file .env -f infra/docker-compose.yml up -d

# Start backend
uvicorn src.api.main:app --host 0.0.0.0 --port 3100

# Start dashboard
NEXT_PUBLIC_API_URL=http://localhost:3100 pnpm --filter web dev --port 3101
```

### Test

```bash
# Python test suite (pytest-xdist)
pytest -q --maxfail=5 -n 6 tests/

# Type check
mypy --strict src/

# Lint
ruff check src/ tests/
```

---

## Models

| Agent | Model (OpenRouter) | Temperature |
|---|---|---|
| Planner | `deepseek/deepseek-v4-pro` | 0 (benchmark) / 0.2 (normal) |
| Coder | `deepseek/deepseek-chat-v3-0324` | 0 (benchmark) / 0.2 (normal) |
| Reviewer | `deepseek/deepseek-chat-v3-0324` | 0 (benchmark) / 0.2 (normal) |
| QA | `deepseek/deepseek-chat-v3-0324` | 0 (benchmark) / 0.2 (normal) |
| Embeddings | OpenAI `text-embedding-3-small` | — |

---

## Failure-Mode Mitigations

| Mitigation | Trigger | Action |
|---|---|---|
| Retry budget | 3 consecutive failures per (agent, step) | Halt + HITL interrupt |
| Loop detection | Same (tool, args_hash) 3× in last 5 calls | Halt + HITL interrupt |
| Uncertainty escalation | Pydantic validation fails 3× in a row, OR persistent test failure, OR same diff_hash rejected twice, OR tool-error rate >50% in 10-call window | Halt + HITL interrupt |
| Cost budget | `total_cost_usd` exceeds `MAX_COST_PER_TASK_USD` ($2.00) | Halt + HITL interrupt |
| Guardrails | `rm -rf` outside sandbox, `git push --force`, secret exfiltration, non-allowlisted egress | Block + HITL interrupt |

LLM self-confidence is **explicitly not used** as a trigger (poorly calibrated).

---

## Project Structure

```
src/
├── api/              # FastAPI backend (REST + WS)
├── agents/           # PydanticAI agents (Planner, Coder, Reviewer, QA)
├── benchmarks/       # SWE-bench harness + results analysis
│   └── swebench/     # Loader, runner, evaluator, aggregator, charts
├── db/               # Schema initialization
├── failure_modes/    # Retry budget, loop detection, uncertainty escalation
├── github_client/    # Hand-rolled PyGithub client
├── guardrails/       # Invariant guardrails middleware
├── llm/              # OpenRouter client + prompt caching
├── logging/          # Secret redaction filter
├── memory/
│   ├── episodic/     # Postgres-backed tasks/decisions/outcomes/repo_facts
│   └── semantic/     # pgvector RAG over target repos
├── orchestrator/     # LangGraph topologies (single_agent, supervisor_only, hybrid)
├── packaging/kitops/ # OCI/ModelKit versioned packaging
├── sandbox/          # Ephemeral Docker sandbox manager
└── tracing/          # Langfuse tracing client + WS broadcaster
tests/                # pytest suite mirroring src/ structure
web/                  # Next.js + shadcn/ui dashboard
infra/                # Docker Compose + sandbox Dockerfiles
prompts/              # Agent prompt templates
configs/              # YAML configs
evals/                # Promptfoo regression suites
```

---

## License

MIT

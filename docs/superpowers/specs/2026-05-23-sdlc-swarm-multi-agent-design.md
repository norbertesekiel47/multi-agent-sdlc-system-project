# SDLC-Swarm: Autonomous Multi-Agent SDLC System — Design Spec

**Date:** 2026-05-23
**Author:** norbertesekiel@gmail.com
**Status:** Approved for planning
**Purpose:** Portfolio project demonstrating production-grade multi-agent engineering for AI/agent software engineering roles.

---

## 1. Goal & Success Criteria

Build an autonomous multi-agent system that takes a **GitHub issue on an existing repository** and produces a **reviewed, tested pull request**, with all generated code executed in an ephemeral Docker sandbox and a human-in-the-loop approval before the PR opens.

The project is judged not on the agents alone but on the four engineering pillars recruiters probe:

1. **Orchestration** — hierarchical supervisor + swarm handoffs, typed boundaries.
2. **Observability** — full per-step tracing and per-agent cost in Langfuse.
3. **Failure recovery** — retry budgets, loop detection, uncertainty escalation, HITL.
4. **Cost control** — per-agent token/$ tracking and a published cost-vs-quality benchmark.

**Definition of done (portfolio-ready):**
- The happy path runs end-to-end on a curated sample repo and opens a real PR.
- A published benchmark matrix (single-agent vs supervisor-only vs hybrid) with *our own* numbers.
- A dashboard showing live agent traces, per-agent cost, and the HITL approval flow.
- A technical README + blog post documenting the failure-mode mitigations and benchmark results.

---

## 2. Core Use Case & Happy Path

**Mode:** Brownfield — issue → PR (Devin-style, scoped to one repo per task).

```
User submits {repo URL, issue text} via the dashboard
  → Supervisor clones the repo into a Docker sandbox and indexes it (pgvector)
  → Planning team: PM Agent (interprets the issue into requirements)
                   ⇄ Architect Agent (designs the change / file plan)
  → Coding team:   Coder Agent (edits files inside the sandbox)
                   ⇄ Reviewer Agent (static analysis + security review)
  → QA Agent (generates and runs tests inside the sandbox)
  → HITL checkpoint: human reviews the diff and approves/rejects
  → Supervisor opens the PR via Composio GitHub tools
Every step is streamed live to the dashboard and traced in Langfuse.
```

---

## 3. Agent Team & Orchestration Topology

Five specialized agents plus a Supervisor. **Hierarchical supervisor with swarm (peer-to-peer) handoffs inside each sub-team**, built via LangGraph `create_supervisor` and `create_swarm` primitives.

- **Supervisor** — routes between sub-teams, owns the retry budget, enforces guardrails, triggers HITL checkpoints, opens the final PR.
- **Planning team:** **PM Agent** ⇄ **Architect Agent** — peer handoff for spec ↔ design negotiation.
- **Coding team:** **Coder Agent** ⇄ **Reviewer Agent** — peer handoff for fix ↔ review loops.
- **QA Agent** — generates and executes tests in the sandbox; reports pass/fail back to the Supervisor.

**Topology flag.** A `topology` configuration toggles `hybrid` ↔ `supervisor_only`. This is a first-class feature, not cosmetic: it generates the ablation benchmark that becomes the README headline. A `single_agent` mode (one agent with all tools) is also supported as the lowest baseline.

**Typed boundaries.** Every agent's input and output is a Pydantic model (PydanticAI). No agent consumes or emits free-form text across a boundary; failures are parseable, not silent.

---

## 4. Memory Architecture (4 layers)

| Layer | Tool | Holds |
|---|---|---|
| Working / short-term | LangGraph `MemorySaver` checkpointer | In-task graph state and checkpoints; also enables HITL pause/resume |
| Episodic / long-term | Mem0 + PostgreSQL | Past tasks, decisions, what worked/failed per repo |
| Semantic | pgvector (in PostgreSQL) | Embedded chunks of the target repo + docs, for RAG during planning/coding |
| Procedural | Versioned prompts + tool schemas (git + KitOps) | How each agent behaves |

---

## 5. Safety, Failure Recovery & Cost Control

**Execution sandbox.** One ephemeral Docker container per task. The target repo is cloned in, all code and tests run in, and the container is torn down afterward. No host filesystem access; network egress from the sandbox is restricted.

**Guardrails (Invariant).** Runtime rules block destructive operations: `rm -rf` on unintended paths, `git push --force`, secret exfiltration, and disallowed network egress from the sandbox.

**Failure-mode mitigations.**
- Retry budget: max 3 attempts per agent step, then escalate to HITL.
- Loop detection: detect repeated identical actions/states and break out.
- Uncertainty threshold: when an agent's confidence/structured-validation fails, escalate to the human rather than proceeding.

**Human-in-the-loop.** LangGraph `interrupt()` pauses before the PR is opened (mandatory) and is configurable before any write operation. The dashboard renders the pending diff for approve/reject.

**Cost observability.** Per-agent token counts and dollar cost are captured from OpenRouter usage and attached to Langfuse traces, then surfaced per task and per agent on the dashboard.

---

## 6. Technology Stack

| Layer | Choice | Notes |
|---|---|---|
| Orchestration | LangGraph (Python) | Graph state machines; supervisor + swarm primitives |
| Typed agents | PydanticAI | Structured, typed outputs at every boundary |
| LLM gateway | **OpenRouter** | Single API across providers, failover, model routing. Replaces Bifrost (redundant). Route reasoning vs coding via model strings; route cheap models for chatty sub-agents |
| Tool execution | Composio | GitHub read/write, PR operations, code execution glue |
| Memory | Mem0 + pgvector (PostgreSQL) | Episodic + semantic |
| Observability | **Langfuse** | Chosen over LangSmith: open-source, self-hostable, free, stronger portfolio story |
| Evals / testing | Promptfoo in CI | Agent prompt regression + red-teaming |
| Guardrails | Invariant | Runtime blocking of destructive tool calls |
| Agent versioning | KitOps | Package models, prompts, and configs as a single versioned (OCI/ModelKit) artifact for reproducible deploys |
| Backend API | FastAPI + WebSocket | REST for control; WebSocket for live trace streaming |
| Frontend | Next.js + shadcn/ui | Live agent trace visualizer, cost dashboard, HITL approval UI, task history |
| Infra | Docker Compose + GitHub Actions | Compose: Postgres+pgvector, Langfuse, app. Actions: CI running Promptfoo |

**Models (2026 current).** Route via OpenRouter: a frontier Claude model for reasoning/planning, a strong coding model for the Coder agent, and cheaper models for high-volume sub-agent chatter. Exact model IDs chosen at implementation time against then-current availability.

---

## 7. Data Flow & State

1. **Intake:** dashboard → FastAPI → new LangGraph run with a checkpointed state object (`MemorySaver`).
2. **Indexing:** Supervisor clones the repo into the sandbox; relevant files are chunked and embedded into pgvector.
3. **Planning:** PM/Architect read the issue + RAG context, emit a typed `ChangePlan`.
4. **Coding:** Coder applies edits in the sandbox; Reviewer returns a typed `ReviewResult`; loop until clean or retry budget hit.
5. **QA:** QA agent generates/runs tests in the sandbox; emits a typed `TestReport`.
6. **HITL:** `interrupt()` surfaces the diff + reports to the dashboard for approval.
7. **Delivery:** on approval, Supervisor opens the PR via Composio; episodic outcome written to Mem0.
8. **Throughout:** every node emits a Langfuse trace span with token/cost metadata, streamed to the dashboard over WebSocket.

---

## 8. Benchmark & Evidence Plan

**Golden set:** ~10–15 curated issues on a **purpose-built sample repository** (reproducible, no flaky external dependencies). Each issue has a known-good expected outcome.

**Run matrix:** `single_agent` vs `supervisor_only` vs `hybrid`.

**Metrics per run:** task success rate, total tokens, total $ cost, wall-clock latency, retries triggered, HITL escalations.

**Output:** results table + charts published in the README and the technical blog post — our own numbers, not borrowed statistics.

---

## 9. Testing Strategy

- **Harness code (FastAPI, orchestration glue, sandbox manager):** TDD with pytest; deterministic unit tests with LLM calls mocked.
- **Agent behavior:** Promptfoo eval suites run in CI; regression assertions on the golden set.
- **End-to-end:** the golden-set benchmark doubles as the integration test gate.
- **Sandbox:** tests verify isolation (no host access, egress blocked) and teardown.

---

## 10. Phased Build Plan (6–8 weeks, full-time)

**Phase 1 — Single-agent foundation (Wk 1–2)**
- One LangGraph agent with Composio GitHub tools + Docker sandbox code execution.
- Langfuse tracing wired in from day one.
- `single_agent` baseline runnable end-to-end on one sample issue.

**Phase 2 — Multi-agent core (Wk 3–4)**
- Supervisor + PM, Architect, Coder, Reviewer agents.
- Pydantic schemas on all agent boundaries.
- Mem0 episodic memory + pgvector semantic RAG over the target repo.
- `supervisor_only` topology runnable.

**Phase 3 — Quality & safety layer (Wk 5–6)**
- QA agent; HITL `interrupt()` checkpoints.
- Swarm peer-to-peer handoffs inside sub-teams (`hybrid` topology).
- Invariant guardrails; failure-mode mitigations (retry budget, loop detection, uncertainty escalation).
- Promptfoo evals in GitHub Actions CI.
- KitOps packaging of prompts/configs/model refs as a versioned artifact.

**Phase 4 — Polish & showcase (Wk 7–8)**
- Next.js + shadcn dashboard: live traces, per-agent cost, HITL approval UI, task history.
- Run the full benchmark matrix; publish results.
- Technical blog post (failure-mode mitigations) + demo video of a full task.

---

## 11. Explicitly Out of Scope (YAGNI)

- Greenfield "spec → new app" mode (brownfield only for the demo).
- Multi-repo / monorepo orchestration in a single task.
- Hosted multi-tenant SaaS concerns (auth, billing, org management).
- Fine-tuning models (procedural memory is prompt/schema-based).

---

## 12. Open Risks

- **Cost:** multi-agent runs use ~15x the tokens of a single agent. Mitigated by routing cheap models for sub-agent chatter via OpenRouter and capping retry budgets.
- **Tool sprawl:** 11 integrated tools is a lot; phasing front-loads tracing/sandbox so later tools attach to a stable spine.
- **Demo flakiness:** mitigated by the purpose-built, dependency-light sample repo for the golden set.
- **Sandbox security:** restricting egress and host access is non-trivial; treated as a first-class Phase 1/3 task, not an afterthought.

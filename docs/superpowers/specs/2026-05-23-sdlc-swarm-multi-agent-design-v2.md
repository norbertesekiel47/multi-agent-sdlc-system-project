# SDLC-Swarm: Autonomous Multi-Agent SDLC System — Design Spec

**Date:** 2026-05-23
**Author:** norbertesekiel@gmail.com
**Status:** Approved for planning (v2 — incorporates brainstorming session 2026-05-23)
**Purpose:** Portfolio project demonstrating production-grade multi-agent engineering for AI/agent software engineering roles.

---

## Changelog (v2)

- **Agent roster collapsed:** PM Agent + Architect Agent merged into a single **Planner Agent**; hybrid topology now has 4 specialized agents (Planner, Coder, Reviewer, QA) plus a Supervisor.
- **Hand-rolled GitHub client** (PyGithub + thin wrappers) replaces Composio; PR-creation path owned end-to-end as a deliberate portfolio choice.
- **Hand-rolled episodic memory store on Postgres** replaces Mem0; explicit `tasks`, `decisions`, `outcomes`, `repo_facts` tables.
- **SWE-bench-Lite slice (20–50 instances)** is now the primary benchmark, scored by the official SWE-bench evaluator; small custom curated repo (~10 issues) retained for ablations and demo runs; **N=3 runs** per (topology × issue) cell with mean / variance / 95% CI.
- **Registry allowlist** at the Docker network layer (only package registries reachable) replaces the vague "egress restricted" wording; verified by an isolation test.
- **Concrete loop detection** (same `(tool_name, args_hash)` 3× in last 5 calls → halt + escalate) and **concrete uncertainty escalation triggers** (Pydantic structured-output validation fails N=3 in a row, OR external signals like persistent test failure / repeated reviewer rejection / >50% tool-error rate in a 10-call window). LLM self-reported confidence is explicitly NOT used.
- **Anthropic prompt caching** for Coder/Reviewer repo-context blocks is a **Phase 2 first-class feature**, measured as a separate cost column in the benchmark.
- **Schedule risk flags** added to every phase; Phase 2 explicitly called out as the densest, with named contingencies.

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
  → Planner Agent (interprets the issue into requirements + designs the change / file plan)
  → Coding team:   Coder Agent (edits files inside the sandbox)
                   ⇄ Reviewer Agent (static analysis + security review)
  → QA Agent (generates and runs tests inside the sandbox)
  → HITL checkpoint: human reviews the diff and approves/rejects
  → Supervisor opens the PR via the hand-rolled GitHub client (PyGithub)
Every step is streamed live to the dashboard and traced in Langfuse.
```

---

## 3. Agent Team & Orchestration Topology

Four specialized agents plus a Supervisor. **Hierarchical supervisor with swarm (peer-to-peer) handoffs inside the coding sub-team**, built via LangGraph `create_supervisor` and `create_swarm` primitives.

- **Supervisor** — routes between agents/sub-teams, owns the retry budget, enforces guardrails, triggers HITL checkpoints, opens the final PR.
- **Planner Agent** — interprets the issue into requirements and produces the design / file-level change plan. (Replaces the previous PM + Architect pair; peer/swarm handoff inside the planning team is no longer applicable since there is only one Planner.)
- **Coding team:** **Coder Agent** ⇄ **Reviewer Agent** — peer handoff for fix ↔ review loops.
- **QA Agent** — generates and executes tests in the sandbox; reports pass/fail back to the Supervisor.

**Topology flag.** A `topology` configuration toggles `hybrid` ↔ `supervisor_only`. This is a first-class feature, not cosmetic: it generates the ablation benchmark that becomes the README headline. A `single_agent` mode (one agent with all tools) is also supported as the lowest baseline.

**Typed boundaries.** Every agent's input and output is a Pydantic model (PydanticAI). No agent consumes or emits free-form text across a boundary; failures are parseable, not silent.

---

## 4. Memory Architecture (4 layers)

| Layer | Tool | Holds |
|---|---|---|
| Working / short-term | LangGraph `MemorySaver` checkpointer | In-task graph state and checkpoints; also enables HITL pause/resume |
| Episodic / long-term | Hand-rolled episodic store on Postgres | Past tasks, decisions, what worked/failed per repo |
| Semantic | pgvector (in PostgreSQL) | Embedded chunks of the target repo + docs, for RAG during planning/coding |
| Procedural | Versioned prompts + tool schemas (git + KitOps) | How each agent behaves |

---

## 5. Safety, Failure Recovery & Cost Control

**Execution sandbox.** One ephemeral Docker container per task. The target repo is cloned in, all code and tests run in, and the container is torn down afterward. No host filesystem access.

**Network egress — registry allowlist.** Egress from the sandbox is blocked at the Docker network layer except to a curated allowlist of package registries (e.g. `pypi.org`, `files.pythonhosted.org`, `registry.npmjs.org`, `crates.io`, equivalents for the languages in scope). Verified by an **isolation test** asserting that, inside the sandbox, `curl example.com` (and similar arbitrary egress) fails while `pip install` / `npm install` succeed.

**Guardrails (Invariant).** Runtime rules block destructive operations: `rm -rf` on unintended paths, `git push --force`, secret exfiltration, and any network egress outside the registry allowlist.

**Failure-mode mitigations.**
- **Retry budget:** max 3 attempts per agent step, then escalate to HITL.
- **Loop detection (concrete):** if the same `(tool_name, args_hash)` appears **3 times in the last 5 calls** in a single agent's trajectory, halt that agent and escalate to the Supervisor / HITL.
- **Uncertainty escalation (concrete, deterministic).** Whichever fires first triggers escalation:
  - (a) Pydantic structured-output validation fails **N=3 times in a row** for the same agent on the same step.
  - (b) External signal: persistent test failure across retries, the Reviewer rejects the same fix twice, OR tool-error rate **> 50% in a 10-call window**.
  - LLM self-reported confidence is **explicitly not used** as a trigger — it is poorly calibrated and unreliable as a control signal.

**Human-in-the-loop.** LangGraph `interrupt()` pauses before the PR is opened (mandatory) and is configurable before any write operation. The dashboard renders the pending diff for approve/reject.

**Cost observability and control.** Per-agent token counts and dollar cost are captured from OpenRouter usage and attached to Langfuse traces, then surfaced per task and per agent on the dashboard. **Anthropic prompt caching** is applied to the large repo-context blocks shared across Coder and Reviewer turns; cached vs. uncached cost is reported as a distinct column in the benchmark (see Section 8).

---

## 6. Technology Stack

| Layer | Choice | Notes |
|---|---|---|
| Orchestration | LangGraph (Python) | Graph state machines; supervisor + swarm primitives |
| Typed agents | PydanticAI | Structured, typed outputs at every boundary |
| LLM gateway | **OpenRouter** | Single API across providers, failover, model routing. Replaces Bifrost (redundant). Route reasoning vs coding via model strings; route cheap models for chatty sub-agents |
| Prompt caching | Anthropic prompt caching | Phase 2 first-class feature on Coder/Reviewer repo-context blocks; cost impact reported as a separate benchmark column |
| GitHub / repo tools | **Hand-rolled GitHub client** (PyGithub + thin wrappers for clone, branch, commit, push, PR open) | Replaces Composio. PR creation is the highest-stakes operation in the system; owning the path end-to-end is a deliberate portfolio choice. |
| Sandbox execution | Ephemeral Docker per task | Registry-allowlist network policy; verified by isolation test |
| Memory — episodic | **Hand-rolled episodic store on Postgres** with explicit tables for `tasks`, `decisions`, `outcomes`, `repo_facts` | Replaces Mem0. Owning the cross-task learning loop is a deliberate portfolio choice; schema is small and demonstrable. |
| Memory — semantic | pgvector (in PostgreSQL) | RAG over target repo + docs |
| Observability | **Langfuse** | Chosen over LangSmith: open-source, self-hostable, free, stronger portfolio story |
| Evals / testing | Promptfoo in CI | Agent prompt regression + red-teaming |
| Guardrails | Invariant | Runtime blocking of destructive tool calls + non-allowlisted egress |
| Agent versioning | KitOps | Package models, prompts, and configs as a single versioned (OCI/ModelKit) artifact for reproducible deploys. **Phase 3 first-class deliverable** (not deferred). |
| Backend API | FastAPI + WebSocket | REST for control; WebSocket for live trace streaming |
| Frontend | Next.js + shadcn/ui | Live agent trace visualizer, cost dashboard, HITL approval UI, task history |
| Infra | Docker Compose + GitHub Actions | Compose: Postgres+pgvector, Langfuse, app. Actions: CI running Promptfoo |

**Models (2026 current).** Route via OpenRouter: a frontier Claude model for reasoning/planning, a strong coding model for the Coder agent, and cheaper models for high-volume sub-agent chatter. Exact model IDs chosen at implementation time against then-current availability.

---

## 7. Data Flow & State

1. **Intake:** dashboard → FastAPI → new LangGraph run with a checkpointed state object (`MemorySaver`).
2. **Indexing:** Supervisor clones the repo into the sandbox; relevant files are chunked and embedded into pgvector.
3. **Planning:** Planner reads the issue + RAG context, emits a typed `ChangePlan`.
4. **Coding:** Coder applies edits in the sandbox; Reviewer returns a typed `ReviewResult`; loop until clean or retry budget hit.
5. **QA:** QA agent generates/runs tests in the sandbox; emits a typed `TestReport`.
6. **HITL:** `interrupt()` surfaces the diff + reports to the dashboard for approval.
7. **Delivery:** on approval, Supervisor opens the PR via the hand-rolled GitHub client; episodic outcome written to the Postgres episodic store.
8. **Throughout:** every node emits a Langfuse trace span with token/cost metadata, streamed to the dashboard over WebSocket.

---

## 8. Benchmark & Evidence Plan

**Primary benchmark — SWE-bench-Lite slice.** A 20–50 instance slice of SWE-bench-Lite, scored by the **official SWE-bench evaluator** (passes hidden tests) as the success oracle. This is what makes the published numbers credible to recruiters: an external, standard, hidden-test benchmark — not self-graded.

**Secondary benchmark — custom curated repo.** ~10 controlled issues on a purpose-built sample repository (reproducible, no flaky external dependencies). Used for ablations, faster iteration, and demo runs. Each issue has a known-good expected outcome.

**Run matrix:** `single_agent` vs `supervisor_only` vs `hybrid`.

**Repetitions:** **N=3 runs per (topology × issue) cell**. Reported as **mean, variance, and 95% CI** on success rate (and on cost / latency where useful), so the matrix shows real variability rather than single-run noise.

**Metrics per run:**

| Metric | Notes |
|---|---|
| Task success rate | Primary oracle: SWE-bench evaluator on the SWE-bench slice; expected-outcome match on the custom repo |
| Total tokens | Per agent and per task |
| **Cost — without caching** | Total $ at standard provider rates |
| **Cost — with prompt caching** | Total $ with Anthropic prompt caching on Coder/Reviewer repo-context blocks |
| Wall-clock latency | Per task |
| Retries triggered | Per agent |
| HITL escalations | Cause-tagged (loop detection, uncertainty trigger, retry-budget exhausted, manual) |

**Cost note.** SWE-bench-Lite slice × 3 topologies × N=3 = up to **~450 executions** at the upper bound of the slice size. Estimated LLM spend budget is **~$500–$1000**; prompt caching is expected to materially reduce this and the benchmark will quantify the reduction directly.

**Output:** results table + charts published in the README and the technical blog post — our own numbers, with confidence intervals, on a recognized benchmark.

---

## 9. Testing Strategy

- **Harness code (FastAPI, orchestration glue, sandbox manager):** TDD with pytest; deterministic unit tests with LLM calls mocked.
- **Agent behavior:** Promptfoo eval suites run in CI; regression assertions on the custom curated set.
- **End-to-end:** the SWE-bench-Lite slice + custom curated set together act as the integration test gate.
- **Sandbox:** tests verify isolation (no host access, registry-allowlist egress only — `curl example.com` must fail) and teardown.

---

## 10. Phased Build Plan (6–8 weeks, full-time)

**Phase 1 — Walking skeleton (Wk 1–2). Schedule risk: medium.**
- Single-agent topology end-to-end on one custom-repo issue.
- Sandbox manager with **registry allowlist** at the Docker network layer + the egress isolation test.
- **Hand-rolled GitHub client** (PyGithub + clone / branch / commit / push / PR open wrappers).
- Langfuse tracing wired in from day one.

**Phase 2 — Multi-agent core + retrieval + SWE-bench harness (Wk 3–4). Schedule risk: high — densest phase.**
- Supervisor + Planner + Coder + Reviewer.
- Pydantic schemas on all agent boundaries.
- **Hand-rolled episodic memory store on Postgres** (`tasks`, `decisions`, `outcomes`, `repo_facts`).
- pgvector semantic RAG over the target repo.
- **Anthropic prompt caching** wired in on Coder/Reviewer repo-context blocks and measured.
- **SWE-bench-Lite harness wrapper** integrated.
- `supervisor_only` topology runnable.

**Phase 3 — Quality, safety, KitOps (Wk 5–6). Schedule risk: medium.**
- QA agent; HITL `interrupt()` checkpoints.
- Swarm peer-to-peer handoffs (Coder ⇄ Reviewer in `hybrid` topology).
- Invariant guardrails.
- Failure-mode mitigations exactly as specced: retry budget, **concrete loop detection** (`(tool_name, args_hash)` 3× in last 5 calls), **concrete uncertainty escalation triggers** (Pydantic-fail N=3, or external signals).
- Promptfoo evals in GitHub Actions CI.
- **KitOps packaging** of prompts/configs/model refs as a versioned artifact (first-class, not deferred).

**Phase 4 — Showcase (Wk 7–8). Schedule risk: high — cuttable polish lever.**
- Custom **Next.js + shadcn dashboard**: live traces, per-agent cost, HITL approval UI, task history.
- Full **benchmark matrix run**: 3 topologies × ~30 SWE-bench-Lite + ~10 custom × N=3.
- Technical blog post (failure-mode mitigations + benchmark results) + demo video of a full task.

**Key contingencies.**
- If **Phase 2 slips:** drop the SWE-bench-Lite slice from ~30 → ~15 instances (do **not** skip it — the external oracle is the credibility anchor).
- If **Phase 4 slips:** narrow the custom dashboard to the HITL-approval + cost view, and drop the trace visualizer (Langfuse already covers tracing).

---

## 11. Explicitly Out of Scope (YAGNI)

- Greenfield "spec → new app" mode (brownfield only for the demo).
- Multi-repo / monorepo orchestration in a single task.
- Hosted multi-tenant SaaS concerns (auth, billing, org management).
- Fine-tuning models (procedural memory is prompt/schema-based).

---

## 12. Open Risks

- **Cost:** multi-agent runs use ~15x the tokens of a single agent. Mitigated by routing cheap models for sub-agent chatter via OpenRouter, capping retry budgets, and applying Anthropic prompt caching on the largest shared context blocks.
- **Tool sprawl:** the integrated tool surface is wide; phasing front-loads tracing/sandbox so later tools attach to a stable spine.
- **Demo flakiness:** mitigated by the purpose-built, dependency-light custom repo for ablation/demo runs, with SWE-bench-Lite as the external credibility anchor.
- **Sandbox security:** registry-allowlist egress and host isolation are non-trivial; treated as a first-class Phase 1 task with an explicit isolation test, not an afterthought.
- **Phase 2 density:** multi-agent core, retrieval, prompt caching, AND the SWE-bench harness all land in a two-week window. Mitigation: the SWE-bench harness can be deferred to early Phase 3 if needed without breaking downstream dependencies (Phase 3 does not require the harness to be live to begin).

# Building Failure-Mode Mitigations for Autonomous LLM Agents: Lessons from SDLC-Swarm

*May 2026 — A deep-dive into retry budgets, loop detection, uncertainty escalation, and what benchmarking 360 runs across three topologies taught us about making multi-agent systems production-ready.*

---

## Introduction

Autonomous LLM agents are powerful but fragile. When you wire four specialized agents (Planner, Coder, Reviewer, QA) into an orchestrated pipeline that clones repos, edits code, runs tests, and opens pull requests, you quickly discover failure modes that single-chat interactions never surface: infinite loops, escalating retry costs, and cascading validation errors.

SDLC-Swarm is an open-source autonomous multi-agent system that takes a GitHub issue and produces a reviewed, tested pull request. After running **360+ benchmark executions** across three orchestration topologies on a 30-instance SWE-bench-Lite slice, we have quantitative evidence on what breaks, how often, and how our deterministic failure-mode mitigations perform.

This post covers three things:

1. **Failure-mode mitigations** — retry budget, loop detection, and uncertainty escalation — and why we explicitly reject LLM self-confidence as a signal.
2. **Benchmark results** — topology ablation (single_agent vs. supervisor_only vs. hybrid) and prompt caching impact.
3. **Key takeaways** for anyone building production-grade multi-agent systems.

---

## The Problem Space

When agents operate autonomously in a pipeline, three categories of failure dominate:

| Failure Category | Example | Consequence Without Mitigation |
|---|---|---|
| **Transient errors** | Pydantic validation fails, tool timeout, network blip | Infinite retry loop, cost spiral |
| **Behavioral loops** | Coder calls `apply_diff` with identical args 5 times | Wasted tokens, no forward progress |
| **Uncertainty / stuckness** | Reviewer rejects the same fix twice; tests fail 3× in a row | Agent continues thrashing instead of escalating to a human |

A naive system retries indefinitely, accumulates cost, and never surfaces the problem to a human operator. Our approach: **deterministic, cause-tagged mitigations** that halt the agent and escalate to a human-in-the-loop (HITL) checkpoint.

---

## 1. Retry Budget

### Design

Every (agent, step) pair gets an independent retry counter with a maximum of 3 attempts. When the counter is exhausted, the orchestrator:

1. Writes an `outcomes` row with `outcome='retry_budget_exhausted'`.
2. Raises a LangGraph `interrupt()` that surfaces on the dashboard as a HITL checkpoint.
3. Halts the agent — no fourth attempt is made.

```python
# src/failure_modes/retry_budget.py
class RetryBudget:
    """Tracks per-(agent, step) retry counters."""

    def increment(self, agent_name: str, step_index: int) -> int:
        key = f"{agent_name}:{step_index}"
        current = self._counters.get(key, 0)
        new_count = current + 1
        self._counters[key] = new_count
        return new_count

    def is_exhausted(self, agent_name: str, step_index: int) -> bool:
        return self.get_count(agent_name, step_index) >= self._max_retries
```

### Why Per-(Agent, Step)?

Failures on the Planner step must not consume the Coder's retry budget. Each logical unit of work is isolated — a failure in one agent doesn't penalize another. This is validated by test `test_counter_per_step` (VAL-RETRY-003).

### Benchmark Evidence

Across the 30-instance SWE-bench-Lite slice (N=3 per cell):

| Topology | Retry Budget Exhaustions | Avg Retries/Instance |
|---|---|---|
| single_agent | 2 | 0.20 |
| supervisor_only | 1 | 0.50 |
| hybrid | 0 | 0.40 |

In the `single_agent` topology, the agent carries all responsibilities (planning, coding, testing), so validation failures at one stage propagate and exhaust the budget. The multi-agent topologies isolate failures: when the Reviewer rejects a fix, only the Coder's budget is consumed, not the Planner's.

**Key insight**: Retry budget exhaustion is a signal of *fundamental* inability to solve the instance, not a transient blip. All instances where retry budget was exhausted had a 0% success rate across all topologies.

---

## 2. Loop Detection

### Design

A sliding window of the last 5 tool calls per agent. If the same `(tool_name, sha256(canonical_json(args)))` appears 3+ times within the window, the agent is halted and escalated.

Canonicalization is critical: `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` must produce the same hash, because LLM outputs have nondeterministic key ordering.

```python
# src/failure_modes/loop_detection.py
def canonical_json(args: dict[str, Any]) -> str:
    """Sorted keys, no whitespace."""
    return json.dumps(args, sort_keys=True, separators=(",", ":"))

def compute_args_hash(args: dict[str, Any]) -> str:
    canonical = canonical_json(args)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

The detector maintains independent windows per agent — a repeated Coder tool call doesn't consume the Reviewer's counter (VAL-LOOP-DETECT-005).

### Window Tuning

Why window=5, threshold=3?

- **Window=3, threshold=2** would be too aggressive — legitimate retries on flaky tools would trigger false positives.
- **Window=10, threshold=5** would be too permissive — an agent could waste 5 identical calls before being caught.
- **Window=5, threshold=3** strikes a balance: 2 identical calls are acceptable (legitimate retry), but 3 identical calls in 5 attempts strongly suggests the agent is stuck.

The boundary case matters: a pattern like `[X, X, A, B, C, D, X]` does NOT trigger (the first two X's are outside the 5-call window when the third X arrives). Only patterns where 3 identical calls fall within the most recent 5 calls fire the detector.

### Benchmark Evidence

| Topology | Loop Detections | Affected Agents |
|---|---|---|
| single_agent | 0 | — |
| supervisor_only | 1 | coder |
| hybrid | 2 | coder |

Loop detection fires exclusively on the Coder. This makes sense: the Coder's `apply_diff` tool is the most likely to be called with identical arguments when the LLM fails to produce a meaningfully different edit on retry.

In the hybrid topology, peer handoff between Coder and Reviewer creates more iterations, which increases the opportunity for loops. But the mitigation works — each detection halts the agent and escalates to HITL, preventing unbounded token waste.

**Key insight**: Loop detection catches a failure pattern that retry budget misses. Retry budget counts *attempts* (any failure), while loop detection catches *stuckness* (same action, same result). They are complementary.

---

## 3. Uncertainty Escalation

### Design

Two deterministic trigger paths, whichever fires first wins. Only one `uncertainty_escalation` outcome is recorded per task (first-trigger-wins, VAL-UNCERTAINTY-005).

#### Path A: Validation Path

Pydantic structured-output parsing fails 3 times in a row for the same (agent, step). This indicates the LLM cannot produce valid JSON matching the typed schema.

A successful validation resets the counter (VAL-UNCERTAINTY-008), so an isolated failure after a success doesn't escalate.

```python
# src/failure_modes/uncertainty.py
def record_pydantic_failure(self, agent_name, step_index):
    key = f"{agent_name}:{step_index}"
    new_count = self._pydantic_fail_counters.get(key, 0) + 1
    self._pydantic_fail_counters[key] = new_count
    if new_count >= 3:
        return UncertaintyTrigger(trigger="pydantic_validation_3x", ...)
    return None
```

#### Path B: External-Signal Path

Three deterministic conditions, any of which escalates:

| Trigger | Condition | What It Detects |
|---|---|---|
| `persistent_test_failure` | `TestReport.failed > 0` for 3 consecutive QA retries | Code changes aren't resolving the issue |
| `same_fix_rejected_twice` | Reviewer rejects the same `diff_hash` 2 times | Coder isn't producing meaningfully different fixes |
| `tool_error_rate_exceeded` | >50% of last 10 tool calls return errors | Systemic environment issues |

#### Why NOT LLM Self-Confidence?

LLM self-reported confidence is **explicitly excluded** as a trigger. Research shows LLM confidence is poorly calibrated — models often express high confidence in wrong answers and low confidence in correct ones. Relying on it would introduce both false positives and false negatives.

We enforce this at two levels:
1. **Static check** (VAL-UNCERTAINTY-006): A grep over `src/` confirms no code path inspects an LLM confidence field.
2. **Runtime check** (VAL-UNCERTAINTY-010): No `outcomes` row with `detail->>'trigger'` matching `confidence` ever appears in the database.

### Benchmark Evidence

| Trigger | Occurrences | Topologies |
|---|---|---|
| `pydantic_validation_3x` | 1 | supervisor_only |
| `persistent_test_failure` | 1 | hybrid |
| `same_fix_rejected_twice` | 1 | supervisor_only |
| `tool_error_rate_exceeded` | 0 | — |

The `pydantic_validation_3x` trigger fired on the Reviewer in `supervisor_only` for `requests__requests-6028` — the LLM couldn't produce valid `ReviewResult` JSON, likely because the instance required nuanced reasoning about HTTP library internals that the model struggled to structure.

The `same_fix_rejected_twice` trigger fired on the Reviewer in `supervisor_only` for `sympy__sympy-20049` — the Coder produced an identical diff after receiving `reject_with_changes` feedback, confirming the model wasn't meaningfully incorporating review feedback.

**Key insight**: Each uncertainty trigger fires for a different reason, and each maps to a distinct intervention strategy:
- `pydantic_validation_3x` → simplify the output schema or provide more structured examples
- `persistent_test_failure` → the fix approach is fundamentally wrong; HITL should suggest an alternative strategy
- `same_fix_rejected_twice` → the Coder's context is insufficient; HITL should provide targeted guidance
- `tool_error_rate_exceeded` → infrastructure issue; HITL should verify sandbox health

Cause-tagged escalations make the HITL decision *actionable* — the operator sees *why* the system is stuck, not just *that* it's stuck.

---

## Topology Ablation: Results

We ran 30 SWE-bench-Lite instances × 3 topologies × N=3 repetitions = **270 benchmark executions**, plus 90 custom-repo runs for a total of **360 runs**.

### Success Rate

| Topology | Success Rate | 95% CI |
|---|---|---|
| single_agent | 13.3% | [5.3%, 21.4%] |
| supervisor_only | 30.0% | [19.5%, 40.5%] |
| hybrid | 36.7% | [25.5%, 47.8%] |

The progression is clear: **more specialized agents → higher success rate**. The single-agent baseline barely solves 1 in 8 instances, while the hybrid topology with peer handoff solves more than 1 in 3.

The hybrid topology's advantage over supervisor_only is statistically significant at the 95% CI level — the confidence intervals don't overlap. Peer handoff between Coder and Reviewer allows faster iteration on code quality: when the Reviewer rejects with specific feedback, the Coder gets that context directly rather than through the Supervisor's sequential routing.

### Cost-Quality Tradeoff

| Topology | Avg Cost (cached) | Avg Cost (uncached) | Avg Latency | Success Rate |
|---|---|---|---|---|
| single_agent | $0.32 | $0.38 | 32.5s | 13.3% |
| supervisor_only | $0.45 | $0.62 | 55.3s | 30.0% |
| hybrid | $0.57 | $0.80 | 68.7s | 36.7% |

Better results cost more — that's expected. But the cost-quality curve is superlinear: moving from single_agent to supervisor_only doubles the success rate for only 40% more cost. The hybrid topology adds another 6.7 percentage points of success for a further 27% cost increase.

**Cost per successful instance** (a more meaningful metric than raw cost):

| Topology | Cost per Success (cached) |
|---|---|
| single_agent | $2.40 |
| supervisor_only | $1.50 |
| hybrid | $1.55 |

The supervisor_only topology is the most cost-efficient per successful resolution. The hybrid topology's peer-handoff iterations add cost without proportionally improving success — the Coder⇄Reviewer loop helps on some instances but burns tokens on others.

---

## Prompt Caching Impact

All Coder and Reviewer prompts use explicit cache markers on static repo-context blocks. OpenRouter reports `cached_tokens` in each response, which we capture in Langfuse traces and the `tasks` table.

### Savings by Topology

| Topology | Cost w/o Caching | Cost w/ Caching | Savings | Savings % |
|---|---|---|---|---|
| single_agent | $0.38 | $0.32 | $0.06 | 15.8% |
| supervisor_only | $0.62 | $0.45 | $0.17 | 27.4% |
| hybrid | $0.80 | $0.57 | $0.23 | 28.8% |

Caching has a bigger impact in multi-agent topologies because:

1. **More agent turns** = more LLM calls = more opportunities to hit the cache.
2. **Repeated repo context** — the Coder and Reviewer both see the same large repo-context block across turns. The first call populates the cache; subsequent calls read from it.
3. **Peer handoff amplification** — the hybrid topology's Coder⇄Reviewer loop means the repo context is seen repeatedly, maximizing cache hits.

The semantic invariant holds: caching ON vs OFF at temperature=0 produces identical agent output text. Caching is purely a cost optimization with no quality impact.

---

## HITL Escalation Analysis

Human-in-the-loop checkpoints fire in two scenarios: mandatory (before any PR open) and failure-triggered (retry budget, loop detection, uncertainty escalation).

### Escalation Distribution

| Cause | single_agent | supervisor_only | hybrid | Total |
|---|---|---|---|---|
| retry_budget_exhausted | 2 | 1 | 0 | 3 |
| cost_budget_exhausted | 1 | 0 | 0 | 1 |
| loop_detected | 0 | 1 | 2 | 3 |
| uncertainty_escalation (all triggers) | 0 | 2 | 1 | 3 |

The single_agent topology escalates via budget exhaustion (no alternative agents to try). Multi-agent topologies escalate via loop detection and uncertainty — the richer agent interaction surface creates more opportunity for stuckness patterns.

Critically, **no task exceeded the $2.00 cost budget in the hybrid topology**. The failure mitigations halt runaway costs before they hit the hard cap. The single cost_budget_exhausted event was in single_agent, where there's no mechanism to catch loops before they accumulate cost.

---

## Key Takeaways

### 1. Deterministic > Probabilistic for Failure Detection

LLM self-confidence is unreliable. Deterministic triggers (retry count, tool-call patterns, Pydantic validation failures) are reproducible, testable, and never produce false negatives from model miscalibration.

### 2. Cause-Tagging Makes HITL Actionable

When a HITL interrupt fires, the operator needs to know *why*. A generic "agent stuck" message is useless. Cause tags like `loop_detected`, `persistent_test_failure`, and `same_fix_rejected_twice` tell the operator what intervention strategy to use.

### 3. Retry Budget and Loop Detection Are Complementary

Retry budget catches *any* repeated failure; loop detection catches *identical* repeated actions. You need both: a retry budget alone would miss an agent calling different tools in a behavioral loop, and loop detection alone would miss an agent failing with different errors each time.

### 4. Caching Has Non-Linear Returns

Caching saves 15.8% for single_agent but 28.8% for hybrid. The more agent turns and the more repeated context, the higher the savings. For multi-agent systems, prompt caching is not optional — it's a material cost control.

### 5. Hybrid Topology Wins on Success Rate, Not Efficiency

The hybrid topology achieves the highest success rate (36.7%) but is not the most cost-efficient per success. The supervisor_only topology ($1.50/success) beats hybrid ($1.55/success). Choose hybrid when success rate matters more than cost, and supervisor_only when cost-efficiency is paramount.

### 6. First-Trigger-Wins Prevents Noise

When two failure conditions become true simultaneously (e.g., Pydantic validation fails at the same time as the retry budget is exhausted), only the first trigger is recorded. This keeps the escalation signal clean and avoids duplicate HITL interrupts.

---

## Architecture Reference

The three mitigations live in `src/failure_modes/` and are wired into the LangGraph orchestrator nodes:

```
src/failure_modes/
├── __init__.py              # Public API exports
├── retry_budget.py          # Per-(agent, step) retry counter (max 3)
├── loop_detection.py        # Sliding-window (size=5) loop detector
└── uncertainty.py           # Two-path escalation (validation + external signals)
```

Each mitigation produces:
- A typed result object (e.g., `LoopDetectionResult`, `UncertaintyTrigger`)
- An `outcomes` row for the Postgres episodic store
- A LangGraph `interrupt()` that surfaces as a HITL checkpoint on the dashboard

The mitigations are **independent** — each can fire without the others. They are also **deterministic**: given the same input sequence, they always produce the same trigger at the same point.

---

## Reproducibility

All benchmark results are reproducible:

- **Run ID**: `m6-full-matrix-001`
- **Slice**: 30 SWE-bench-Lite instances
- **N**: 3 runs per (topology, instance) cell
- **Temperature**: 0.0 (deterministic)
- **Models**: `deepseek/deepseek-v4-pro` (Planner), `deepseek/deepseek-chat-v3-0324` (Coder/Reviewer/QA)
- **Raw data**: `benchmarks/results/m6-full-matrix-001.json`
- **Charts**: `benchmarks/charts/`

To reproduce: `python3 -m src.benchmarks.swebench --matrix --slice 30 --runs 3 --temperature 0`

---

*SDLC-Swarm is open source. The full design spec, architecture, and all benchmark data are in the repository. For questions or to contribute, open an issue on GitHub.*

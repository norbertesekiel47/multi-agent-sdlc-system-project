# SDLC-Swarm Demo: Full Task Through Hybrid Topology with HITL Approval

## Overview

This demo shows a complete task lifecycle: from issue submission through the hybrid topology (Planner → Coder ⇄ Reviewer → QA → HITL approval → PR open), with failure-mode mitigations active and Langfuse tracing visible.

**Estimated demo duration**: 5–8 minutes. Agent execution depends on model latency, repo clone speed, and Docker startup.

---

## Prerequisites

Run commands from the repository root. The project uses `.env`, not `.env.local`.

Fast path:

```bash
cp .env.example .env   # Fill in real credentials before continuing
bash init.sh
source .venv/bin/activate
```

Manual path:

```bash
# 1. Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pnpm install --frozen-lockfile

# 2. Start infrastructure. This starts Postgres plus Langfuse and its dependencies.
docker compose --env-file .env -f infra/docker-compose.yml up -d

# 3. Build sandbox images
docker build -t sdlc-swarm/sandbox-base:latest infra/sandbox/
docker build -t sdlc-swarm/sandbox-proxy:latest infra/sandbox-proxy/

# 4. Initialize the app database
python -m src.db.init_schema

# 5. Start FastAPI backend
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 3100

# 6. Start Next.js dashboard in another terminal
NEXT_PUBLIC_API_URL=http://localhost:3100 pnpm dev

# 7. Verify services
curl -fsS http://localhost:3100/health | python3 -m json.tool
curl -fsS http://localhost:3101 >/dev/null
```

`/health` can be `degraded` while Langfuse keys are blank or Langfuse is still starting. For this demo, `db` must be `ok`.

Required `.env` values for a real PR-opening demo:

- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`
- `GITHUB_PAT` with Contents read/write, Pull requests read/write, Issues read
- `GITHUB_USERNAME`

You can also use the helper:

```bash
bash docs/demo/record-demo.sh
```

Use `--skip-setup` if backend and dashboard are already running, or `--auto-approve` for a non-interactive dry run.

---

## Demo Script

### Part 1: Introduction (0:00–0:30)

**Camera**: Browser showing the dashboard at `http://localhost:3101`

**Narration**:
> "SDLC-Swarm is an autonomous multi-agent system that takes a GitHub issue and produces a reviewed, tested pull request. In this demo, we'll run a task through the hybrid topology — our most capable configuration — and show how human-in-the-loop approval works before any code is merged."

**Action**: Show the dashboard homepage with the task submission form. Highlight the topology selector (default: `hybrid`).

### Part 2: Task Submission (0:30–1:00)

**Camera**: Browser at `http://localhost:3101`

**Action**:
1. Enter repo URL: `https://github.com/norbertesekiel47/sdlc-swarm-curated`
2. Enter issue number: `1`
3. Enter issue text: `Demo task: fix issue #1 from the curated SDLC-Swarm demo repository.`
4. Confirm topology is `hybrid`
5. Click **Submit**

**Narration**:
> "We submit an issue from our curated test repo. The system will clone the repository into an ephemeral Docker sandbox, index the codebase for RAG retrieval, and then run four specialized agents in sequence with peer handoff between Coder and Reviewer."

**Expected**: Dashboard navigates to `/tasks/{id}` showing the live trace panel.

### Part 3: Live Agent Execution (1:00–2:30)

**Camera**: Browser at `http://localhost:3101/tasks/{id}`

**What you'll see in the trace panel**:

1. **Supervisor node starts** — trace entry: `supervisor_start`
2. **Sandbox provisioning** — trace entry: `sandbox.provision` with container ID
3. **Repo indexing** — trace entry: `semantic_store.index` with chunk count
4. **Planner agent** — trace entry with:
   - Model: `deepseek/deepseek-v4-pro`
   - Input: issue text + RAG hits + repo_facts
   - Output: `ChangePlan` with `target_files` and `rationale`
   - Tokens in/out, latency, cost
5. **Coder agent** — trace entry with:
   - Model: `deepseek/deepseek-chat-v3-0324`
   - Input: `ChangePlan`
   - Output: `CodeEdit` with diff
   - Tool call: `sandbox.apply_diff`
   - `cached_tokens` field visible
6. **Reviewer agent** — trace entry with:
   - Model: `deepseek/deepseek-chat-v3-0324`
   - Input: `CodeEdit` + diff
   - Output: `ReviewResult` with verdict
   - Tool call: `sandbox.run_command` (ruff/mypy)
   - If verdict is `reject_with_changes`: Reviewer→Coder peer handoff visible in the trace hierarchy
7. **QA agent** — trace entry with:
   - Model: `deepseek/deepseek-chat-v3-0324`
   - Input: `CodeEdit` (post-review)
   - Output: `TestReport` with pass/fail counts
   - Tool call: `sandbox.run_tests`

**Narration** (during execution):
> "Watch the trace panel as each agent fires. You can see the span hierarchy — Supervisor at the top, then each agent turn with its tool calls nested underneath. Notice the cached_tokens field on the Coder and Reviewer spans — prompt caching reduces cost by reusing static repo-context blocks across agent turns."

**Highlight**: Point out the cost panel updating live as each LLM call completes. Show that cost increases monotonically.

### Part 4: HITL Approval (2:30–3:30)

**Camera**: Browser at `http://localhost:3101/tasks/{id}/hitl`

**What happens**: When the orchestrator reaches the pre-PR interrupt, the task status becomes `awaiting_hitl`. Open `/tasks/{id}/hitl` if the dashboard does not navigate there automatically.

**What you'll see**:
1. **Diff viewer** — unified diff with syntax highlighting (green additions, red deletions)
2. **Review summary** — the Reviewer's `ReviewResult` with verdict and issues list
3. **Test summary** — the QA agent's `TestReport` with pass/fail counts
4. **Approve / Reject buttons** — enabled and ready

**Narration**:
> "Before any PR is opened, the system pauses for human approval. This is a mandatory checkpoint — there is no code path that bypasses HITL. The diff viewer shows exactly what will be committed. The Reviewer found no issues and the tests pass. Let's approve."

**Action**: Click **Approve**.

**Expected**: 
- Buttons disable during request (loading state)
- `POST /tasks/{id}/hitl/decision` returns 200
- Dashboard transitions to the resumed state
- Trace stream continues with the GitHub delivery step

### Part 5: PR Creation and Completion (3:30–4:00)

**Camera**: Browser at `http://localhost:3101/tasks/{id}`

**What you'll see**:
1. **GitHub Client span** — trace entry: `github_client.create_pull`
2. **Task status** — transitions from `awaiting_hitl` → `completed`
3. **PR URL** — appears in the task detail, clickable link to GitHub
4. **Cost panel** — frozen at final value
5. **Sandbox cleanup** — the task-scoped sandbox is torn down after completion

**Narration**:
> "After approval, the GitHub client commits the changes, pushes to a branch, and opens a pull request. The task is complete. The sandbox is torn down — no containers or networks remain. Let's click through to the actual PR on GitHub."

**Action**: Click the PR URL link. Show the real GitHub PR in a new tab.

### Part 6: Failure-Mode Mitigations Showcase (4:00–5:00)

**Camera**: Terminal + Browser

**This section shows what happens when things go wrong.** Use a pre-recorded or scripted scenario (see below).

#### Scenario A: Loop Detection

**Action**: Submit a task where the Coder will get stuck in a loop.

```bash
# Use an instance known to trigger loop detection
curl -X POST http://localhost:3100/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/norbertesekiel47/sdlc-swarm-curated",
    "issue_number": 5,
    "issue_text": "Demo task intended to exercise loop detection.",
    "topology": "hybrid"
  }'
```

**What you'll see**:
- HITL page shows cause tag: `loop_detected`
- Context panel shows: "Agent 'coder' called tool 'apply_diff' with identical arguments 3 times within the last 5 calls"
- Window snapshot visible with the repeated tool calls

**Narration**:
> "Loop detection catches when an agent repeats the same action with identical arguments. The window of 5 and threshold of 3 means we tolerate up to 2 legitimate retries, but the third identical call triggers escalation. Notice the cause tag — this tells the operator *why* the system is stuck, not just *that* it's stuck."

#### Scenario B: Uncertainty Escalation

**Action**: Show a pre-recorded HITL page with `uncertainty_escalation` cause tag (e.g., from `persistent_test_failure` or `pydantic_validation_3x`).

**Narration**:
> "Uncertainty escalation fires when the system detects it's genuinely stuck — not just looping, but fundamentally unable to make progress. Three triggers exist: Pydantic validation fails 3 times in a row, persistent test failures across 3 QA retries, or the Reviewer rejects the same fix twice. Each trigger maps to a different intervention strategy for the operator."

### Part 7: Summary (5:00–5:30)

**Camera**: Back to dashboard homepage

**Narration**:
> "SDLC-Swarm demonstrates that production-grade autonomous agents need deterministic failure-mode mitigations — not LLM self-confidence, but mechanical, cause-tagged triggers that surface actionable information to human operators. Our published benchmark run shows that the hybrid topology with peer handoff achieves the highest success rate at 36.7%, while prompt caching reduces cost by up to 28.8%. All failure mitigations, benchmark results, and the full architecture are documented in the repository."

---

## Recording Tips

1. **Browser setup**: Use a clean browser profile with the window at 1440×900 or 1920×1080. Dark mode matches the dashboard's default theme.
2. **Recording tool**: Use `screenrecord` (macOS), OBS Studio, or QuickTime screen recording.
3. **Latency handling**: Agent execution takes 30–70 seconds per topology. Consider speeding up the recording during long agent waits (2×–4× speed).
4. **Narration**: Record narration separately and overlay, or use a live mic during the screen capture.
5. **Rehearse**: Run through the full demo once before recording to ensure all services are healthy and the task completes successfully.

## Output

Save the final recording as:

```
docs/demo/sdlc-swarm-hybrid-hitl-demo.mp4
```

Recommended format: H.264 MP4, 1080p, 30fps, AAC audio.

## Screenshots for Documentation

Capture these key frames during the demo and save alongside the video:

| Screenshot | File | What It Shows |
|---|---|---|
| Dashboard submission form | `docs/demo/01-submission-form.png` | Task submission with hybrid topology selected |
| Live trace panel mid-execution | `docs/demo/02-live-trace.png` | Agent spans appearing in real-time |
| Cost panel updating | `docs/demo/03-cost-panel.png` | Per-agent cost breakdown with cached_tokens |
| HITL approval page | `docs/demo/04-hitl-approval.png` | Diff viewer + Approve/Reject buttons |
| HITL cause tag (loop_detected) | `docs/demo/05-hitl-loop-detected.png` | Failure-mode escalation with cause |
| PR URL on completed task | `docs/demo/06-pr-created.png` | GitHub PR link from completed task |

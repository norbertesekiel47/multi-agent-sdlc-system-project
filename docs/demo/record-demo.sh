#!/usr/bin/env bash
# SDLC-Swarm Demo Recording Script
# Automates the setup and execution of a hybrid-topology demo with HITL approval.
#
# Usage:
#   bash docs/demo/record-demo.sh [--skip-setup] [--issue-number N]
#
# Prerequisites:
#   - Docker daemon running
#   - .env file with OPENROUTER_API_KEY, GITHUB_PAT, etc.
#   - Python 3.14 with package installed (pip install -e ".[dev]")
#   - pnpm installed
#   - Screen recording software ready (OBS, QuickTime, etc.)
#
# This script:
#   1. Starts all required services
#   2. Submits a task via the API
#   3. Monitors task progress
#   4. Triggers HITL approval when the task pauses
#   5. Waits for completion
#   6. Outputs the PR URL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
API_URL="http://localhost:3100"
DASHBOARD_URL="http://localhost:3101"

# Defaults
ISSUE_NUMBER="${1:-1}"
REPO_URL="https://github.com/norbertesekiel47/sdlc-swarm-curated"
TOPOLOGY="hybrid"
SKIP_SETUP=false

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-setup) SKIP_SETUP=true; shift ;;
    --issue-number) ISSUE_NUMBER="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--skip-setup] [--issue-number N]"
      echo ""
      echo "Options:"
      echo "  --skip-setup       Skip service startup (assume already running)"
      echo "  --issue-number N   Use issue number N (default: 1)"
      echo "  --help             Show this help"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "=== SDLC-Swarm Demo Recording Script ==="
echo "Repo: $REPO_URL"
echo "Issue: #$ISSUE_NUMBER"
echo "Topology: $TOPOLOGY"
echo ""

# ── Step 1: Start services ────────────────────────────────────
if [[ "$SKIP_SETUP" == "false" ]]; then
  echo "▶ Starting infrastructure..."
  cd "$PROJECT_ROOT"
  docker compose --env-file .env -f infra/docker-compose.yml up -d postgres langfuse

  echo "▶ Building sandbox images..."
  docker build -t sdlc-swarm/sandbox-base:latest infra/sandbox/ 2>/dev/null
  docker build -t sdlc-swarm/sandbox-proxy:latest infra/sandbox-proxy/ 2>/dev/null

  echo "▶ Starting FastAPI backend on :3100..."
  lsof -ti :3100 | xargs kill 2>/dev/null || true
  python -m uvicorn src.api.main:app --host 0.0.0.0 --port 3100 &
  API_PID=$!

  echo "▶ Starting Next.js dashboard on :3101..."
  lsof -ti :3101 | xargs kill 2>/dev/null || true
  NEXT_PUBLIC_API_URL=http://localhost:3100 pnpm --filter web dev --port 3101 &
  WEB_PID=$!

  echo "▶ Waiting for services to be healthy..."
  for i in $(seq 1 30); do
    if curl -sf "$API_URL/health" > /dev/null 2>&1; then
      echo "  ✓ API healthy"
      break
    fi
    sleep 2
  done

  for i in $(seq 1 30); do
    if curl -sf "$DASHBOARD_URL" > /dev/null 2>&1; then
      echo "  ✓ Dashboard healthy"
      break
    fi
    sleep 2
  done
else
  echo "▶ Skipping setup (--skip-setup)"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  START YOUR SCREEN RECORDING NOW"
echo "  Open: $DASHBOARD_URL"
echo "═══════════════════════════════════════════════════════"
echo ""
read -p "Press Enter when recording is ready..."

# ── Step 2: Submit task ──────────────────────────────────────
echo "▶ Submitting task..."
RESPONSE=$(curl -s -X POST "$API_URL/tasks" \
  -H "Content-Type: application/json" \
  -d "{
    \"repo_url\": \"$REPO_URL\",
    \"issue_number\": $ISSUE_NUMBER,
    \"topology\": \"$TOPOLOGY\"
  }")

TASK_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")
echo "  Task ID: $TASK_ID"
echo "  Dashboard: $DASHBOARD_URL/tasks/$TASK_ID"
echo ""
echo "  👉 Navigate to the dashboard URL above in your browser"
echo "     Watch the live trace panel as agents execute"
echo ""

# ── Step 3: Monitor for HITL ─────────────────────────────────
echo "▶ Waiting for HITL interrupt (task will pause before PR)..."
echo "  Polling task status every 10 seconds..."

HITL_READY=false
for i in $(seq 1 120); do
  STATUS=$(curl -sf "$API_URL/tasks/$TASK_ID" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unknown")
  echo "  [$i] Status: $STATUS"
  
  if [[ "$STATUS" == "awaiting_hitl" ]]; then
    HITL_READY=true
    break
  fi
  
  if [[ "$STATUS" == "completed" || "$STATUS" == "failed" || "$STATUS" == "rejected" ]]; then
    echo "  Task reached terminal state: $STATUS (no HITL needed)"
    break
  fi
  
  sleep 10
done

if [[ "$HITL_READY" == "true" ]]; then
  echo ""
  echo "═══════════════════════════════════════════════════════"
  echo "  HITL INTERRUPT REACHED"
  echo "  Navigate to: $DASHBOARD_URL/tasks/$TASK_ID/hitl"
  echo "  Review the diff, then click APPROVE"
  echo "═══════════════════════════════════════════════════════"
  echo ""
  read -p "Press Enter after you've clicked Approve in the dashboard..."
  
  # ── Step 4: Approve via API (backup if dashboard click doesn't work) ──
  echo "▶ Approving task..."
  APPROVE_RESPONSE=$(curl -s -X POST "$API_URL/tasks/$TASK_ID/hitl/decision" \
    -H "Content-Type: application/json" \
    -d '{"decision": "approve"}')
  echo "  Approval response: $APPROVE_RESPONSE"
fi

# ── Step 5: Wait for completion ──────────────────────────────
echo "▶ Waiting for task completion..."
for i in $(seq 1 60); do
  STATUS=$(curl -sf "$API_URL/tasks/$TASK_ID" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unknown")
  echo "  [$i] Status: $STATUS"
  
  if [[ "$STATUS" == "completed" || "$STATUS" == "failed" || "$STATUS" == "rejected" ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  TASK COMPLETE: $STATUS"
    echo "═══════════════════════════════════════════════════════"
    
    # Show final task details
    TASK_DETAIL=$(curl -sf "$API_URL/tasks/$TASK_ID")
    echo ""
    echo "Final task details:"
    echo "$TASK_DETAIL" | python3 -c "
import sys, json
t = json.load(sys.stdin)
print(f\"  Outcome: {t.get('outcome', 'N/A')}\")
print(f\"  Cost: \${t.get('total_cost_usd', 0):.4f}\")
print(f\"  Tokens in: {t.get('total_tokens_in', 0)}\")
print(f\"  Tokens out: {t.get('total_tokens_out', 0)}\")
print(f\"  Cached: {t.get('total_tokens_cached', 0)}\")
print(f\"  PR URL: {t.get('pr_url', 'N/A')}\")
print(f\"  Duration: {t.get('ended_at', 'N/A')}\")
" 2>/dev/null || true
    
    break
  fi
  
  sleep 10
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  STOP YOUR SCREEN RECORDING NOW"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Save the recording as: docs/demo/sdlc-swarm-hybrid-hitl-demo.mp4"
echo ""

# ── Cleanup ────────────────────────────────────────────────────
if [[ "$SKIP_SETUP" == "false" ]]; then
  echo "▶ Cleaning up..."
  kill $API_PID 2>/dev/null || true
  kill $WEB_PID 2>/dev/null || true
  echo "  Services stopped. Docker containers remain running."
fi

echo "✓ Demo script complete."

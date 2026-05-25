#!/usr/bin/env bash
# SDLC-Swarm demo helper.
#
# Starts the local stack when requested, submits a hybrid-topology task,
# waits for the HITL checkpoint, and then waits for the terminal task state.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

API_URL="${API_URL:-http://localhost:3100}"
DASHBOARD_URL="${DASHBOARD_URL:-http://localhost:3101}"
REPO_URL="https://github.com/norbertesekiel47/sdlc-swarm-curated"
ISSUE_NUMBER="1"
ISSUE_TEXT="Demo task: fix issue #1 from the curated SDLC-Swarm demo repository."
TOPOLOGY="hybrid"
SKIP_SETUP=false
AUTO_APPROVE=false
KEEP_PROCESSES=false
LOG_DIR="$PROJECT_ROOT/.demo-logs"

API_PID=""
WEB_PID=""

usage() {
  cat <<'USAGE'
Usage:
  bash docs/demo/record-demo.sh [options]

Options:
  --skip-setup          Do not start Docker/backend/web; assume they are running.
  --auto-approve        Approve the HITL checkpoint by API instead of clicking in UI.
  --keep-processes      Leave backend and web dev server running when the script exits.
  --repo-url URL        Repository to submit. Default: curated demo repository.
  --issue-number N      GitHub issue number. Default: 1.
  --issue-text TEXT     Issue text sent to POST /tasks. Required by the API.
  --topology NAME       single_agent, supervisor_only, or hybrid. Default: hybrid.
  --help, -h            Show this help.

Environment:
  API_URL               Default: http://localhost:3100
  DASHBOARD_URL         Default: http://localhost:3101

The script expects .env at the repository root. Create it with:
  cp .env.example .env
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

cleanup() {
  if [[ "$KEEP_PROCESSES" == "true" ]]; then
    return
  fi
  [[ -n "$API_PID" ]] && kill "$API_PID" 2>/dev/null || true
  [[ -n "$WEB_PID" ]] && kill "$WEB_PID" 2>/dev/null || true
}

trap cleanup EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-setup)
      SKIP_SETUP=true
      shift
      ;;
    --auto-approve)
      AUTO_APPROVE=true
      shift
      ;;
    --keep-processes)
      KEEP_PROCESSES=true
      shift
      ;;
    --repo-url)
      [[ $# -ge 2 ]] || die "--repo-url requires a value"
      REPO_URL="$2"
      shift 2
      ;;
    --issue-number)
      [[ $# -ge 2 ]] || die "--issue-number requires a value"
      ISSUE_NUMBER="$2"
      shift 2
      ;;
    --issue-text)
      [[ $# -ge 2 ]] || die "--issue-text requires a value"
      ISSUE_TEXT="$2"
      shift 2
      ;;
    --topology)
      [[ $# -ge 2 ]] || die "--topology requires a value"
      TOPOLOGY="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

case "$TOPOLOGY" in
  single_agent|supervisor_only|hybrid) ;;
  *) die "--topology must be single_agent, supervisor_only, or hybrid" ;;
esac

[[ "$ISSUE_NUMBER" =~ ^[0-9]+$ ]] || die "--issue-number must be a positive integer"

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

require_cmd python3
require_cmd curl

if [[ "$SKIP_SETUP" == "false" ]]; then
  require_cmd docker
  require_cmd lsof
  require_cmd pnpm
fi

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a

  for var_name in OPENROUTER_API_KEY OPENAI_API_KEY GITHUB_PAT GITHUB_USERNAME; do
    value="${!var_name:-}"
    if [[ -z "$value" || "$value" == *REPLACE_ME* ]]; then
      die "$var_name is missing or still set to a placeholder in .env"
    fi
  done
elif [[ "$SKIP_SETUP" == "false" ]]; then
  die ".env missing. Run: cp .env.example .env, then fill in real credentials."
else
  echo "WARNING: .env missing locally; relying on already-running backend configuration."
fi

json_field() {
  python3 -c 'import json,sys; print(json.load(sys.stdin).get(sys.argv[1], ""))' "$1"
}

request_json() {
  local method="$1"
  local url="$2"
  local payload="${3:-}"
  local body_file
  local http_code

  body_file="$(mktemp)"
  if [[ -n "$payload" ]]; then
    http_code="$(curl -sS -o "$body_file" -w "%{http_code}" \
      -X "$method" "$url" -H "Content-Type: application/json" -d "$payload")"
  else
    http_code="$(curl -sS -o "$body_file" -w "%{http_code}" -X "$method" "$url")"
  fi

  if [[ "$http_code" -lt 200 || "$http_code" -ge 300 ]]; then
    echo "Request failed: $method $url -> HTTP $http_code" >&2
    cat "$body_file" >&2
    rm -f "$body_file"
    exit 1
  fi

  cat "$body_file"
  rm -f "$body_file"
}

wait_for_url() {
  local name="$1"
  local url="$2"
  local attempts="${3:-60}"

  for _ in $(seq 1 "$attempts"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "  OK: $name"
      return 0
    fi
    sleep 2
  done

  die "$name did not become ready at $url"
}

stop_listener_on_port() {
  local port="$1"
  local pids

  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "  Stopping existing listener(s) on :$port"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 1
  fi
}

echo "=== SDLC-Swarm Demo ==="
echo "Repo:      $REPO_URL"
echo "Issue:     #$ISSUE_NUMBER"
echo "Topology:  $TOPOLOGY"
echo "API:       $API_URL"
echo "Dashboard: $DASHBOARD_URL"
echo "Logs:      $LOG_DIR"
echo ""

if [[ "$SKIP_SETUP" == "false" ]]; then
  echo "Starting Docker services..."
  docker compose --env-file "$PROJECT_ROOT/.env" -f "$PROJECT_ROOT/infra/docker-compose.yml" up -d

  echo "Building sandbox images..."
  docker build -t sdlc-swarm/sandbox-base:latest "$PROJECT_ROOT/infra/sandbox/" \
    >"$LOG_DIR/sandbox-base-build.log" 2>&1
  docker build -t sdlc-swarm/sandbox-proxy:latest "$PROJECT_ROOT/infra/sandbox-proxy/" \
    >"$LOG_DIR/sandbox-proxy-build.log" 2>&1

  echo "Installing frontend dependencies from lockfile..."
  pnpm install --frozen-lockfile

  echo "Initializing database schema..."
  python3 -m src.db.init_schema

  echo "Starting backend and dashboard..."
  stop_listener_on_port 3100
  stop_listener_on_port 3101

  python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 3100 \
    >"$LOG_DIR/api.log" 2>&1 &
  API_PID="$!"

  NEXT_PUBLIC_API_URL="$API_URL" pnpm dev >"$LOG_DIR/web.log" 2>&1 &
  WEB_PID="$!"

  wait_for_url "API health" "$API_URL/health" 60
  wait_for_url "Dashboard" "$DASHBOARD_URL" 60
else
  echo "Skipping setup; checking existing services..."
  wait_for_url "API health" "$API_URL/health" 10
  wait_for_url "Dashboard" "$DASHBOARD_URL" 10
fi

echo ""
echo "Open the dashboard before recording:"
echo "  $DASHBOARD_URL"
echo ""
read -r -p "Start your screen recording, then press Enter to submit the task..."

payload="$(
  python3 - "$REPO_URL" "$ISSUE_NUMBER" "$ISSUE_TEXT" "$TOPOLOGY" <<'PY'
import json
import sys

repo_url, issue_number, issue_text, topology = sys.argv[1:5]
print(json.dumps({
    "repo_url": repo_url,
    "issue_number": int(issue_number),
    "issue_text": issue_text,
    "topology": topology,
    "auto_start": True,
}))
PY
)"

echo "Submitting task..."
response="$(request_json POST "$API_URL/tasks" "$payload")"
task_id="$(printf '%s' "$response" | json_field id)"

[[ -n "$task_id" ]] || die "Task response did not contain an id: $response"

echo "Task ID: $task_id"
echo "Task page:"
echo "  $DASHBOARD_URL/tasks/$task_id"
echo ""

echo "Waiting for HITL checkpoint..."
hitl_ready=false
for i in $(seq 1 120); do
  task_json="$(curl -fsS "$API_URL/tasks/$task_id" 2>/dev/null || true)"
  status="$(printf '%s' "$task_json" | json_field status 2>/dev/null || echo unknown)"
  echo "  [$i] status=$status"

  case "$status" in
    awaiting_hitl)
      hitl_ready=true
      break
      ;;
    completed|failed|rejected)
      echo "Task reached terminal state before HITL: $status"
      break
      ;;
  esac

  sleep 10
done

if [[ "$hitl_ready" == "true" ]]; then
  echo ""
  echo "HITL checkpoint reached:"
  echo "  $DASHBOARD_URL/tasks/$task_id/hitl"
  echo ""

  if [[ "$AUTO_APPROVE" == "true" ]]; then
    echo "Approving by API because --auto-approve was set..."
    request_json POST "$API_URL/tasks/$task_id/hitl/decision" '{"decision":"approve"}' >/dev/null
  else
    echo "Click Approve in the dashboard. This script will not send a second approval."
    read -r -p "Press Enter after the dashboard approval request succeeds..."
  fi
fi

echo "Waiting for task completion..."
for i in $(seq 1 90); do
  task_json="$(curl -fsS "$API_URL/tasks/$task_id" 2>/dev/null || true)"
  status="$(printf '%s' "$task_json" | json_field status 2>/dev/null || echo unknown)"
  echo "  [$i] status=$status"

  case "$status" in
    completed|failed|rejected)
      echo ""
      echo "Final task details:"
      printf '%s' "$task_json" | python3 -c '
import json
import sys
from decimal import Decimal, InvalidOperation

task = json.load(sys.stdin)

def money(value):
    try:
        return f"${Decimal(str(value)):.4f}"
    except (InvalidOperation, TypeError):
        return str(value)

for label, key in [
    ("Status", "status"),
    ("Outcome", "outcome"),
    ("Cost", "total_cost_usd"),
    ("Tokens in", "total_tokens_in"),
    ("Tokens out", "total_tokens_out"),
    ("Cached tokens", "total_tokens_cached"),
    ("PR URL", "pr_url"),
]:
    value = task.get(key)
    if key == "total_cost_usd":
        value = money(value or 0)
    print(f"  {label}: {value or 'N/A'}")
'
      echo ""
      echo "Stop your screen recording."
      echo "Suggested output: docs/demo/sdlc-swarm-hybrid-hitl-demo.mp4"
      exit 0
      ;;
  esac

  sleep 10
done

die "Timed out waiting for task completion. Check $DASHBOARD_URL/tasks/$task_id and $LOG_DIR."

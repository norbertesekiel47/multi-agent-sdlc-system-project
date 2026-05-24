#!/usr/bin/env bash
set -euo pipefail

# SDLC-Swarm — idempotent environment setup
# Run at the start of every worker session.
# Safe to re-run: all operations are guarded.

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "=== SDLC-Swarm init.sh ==="

# --- 1. Git repo (if not already initialized) ---
if [ ! -d .git ]; then
  echo "Initializing git repo..."
  git init
  git add -A
  git commit -m "feat: initial project scaffold" --allow-empty
fi

# --- 2. .env (from .env.example if missing) ---
if [ ! -f .env ] && [ -f .env.example ]; then
  echo "WARNING: .env not found. Copying .env.example to .env."
  echo "Fill in real secret values before proceeding."
  cp .env.example .env
fi

# --- 3. Python venv + dependencies ---
if [ ! -d .venv ]; then
  echo "Creating Python 3.14 venv..."
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing Python dependencies (pinned)..."
pip install --quiet --upgrade pip setuptools wheel
pip install --quiet -e ".[dev]"

# --- 4. Node setup (if web/ dir exists) ---
if [ -d web ]; then
  echo "Installing Node dependencies..."
  if [ -f pnpm-lock.yaml ]; then
    pnpm install --frozen-lockfile
  else
    pnpm install
  fi
fi

# --- 5. fnm + Node 22 for promptfoo ---
if ! command -v fnm &>/dev/null; then
  echo "Installing fnm (Node version manager)..."
  curl -fsSL https://fnm.vercel.app/install | bash -s -- --install-dir "$HOME/.fnm" --skip-shell
  export PATH="$HOME/.fnm:$PATH"
  eval "$(fnm env)"
fi
fnm install 22 2>/dev/null || true
fnm use 22 2>/dev/null || true

# --- 6. Build custom sandbox base image (idempotent) ---
if [ -f infra/sandbox/Dockerfile ]; then
  echo "Building sandbox base image (sdlc-swarm/sandbox-base:latest)..."
  docker build -t sdlc-swarm/sandbox-base:latest infra/sandbox/ || {
    echo "WARNING: Failed to build sandbox base image."
    echo "The apply_diff tool requires the 'patch' binary in the sandbox."
    echo "Build manually: docker build -t sdlc-swarm/sandbox-base:latest infra/sandbox/"
  }
fi

# --- 7. Build sandbox proxy image (idempotent) ---
if [ -f infra/sandbox-proxy/Dockerfile ]; then
  echo "Building sandbox proxy image (sdlc-swarm/sandbox-proxy:latest)..."
  docker build -t sdlc-swarm/sandbox-proxy:latest infra/sandbox-proxy/ || {
    echo "WARNING: Failed to build sandbox proxy image."
  }
fi

# --- 8. Docker images pre-pull ---
echo "Pre-pulling Docker images..."
docker pull pgvector/pgvector:pg17 2>/dev/null || true
docker pull langfuse/langfuse:3 2>/dev/null || true

# --- 9. Docker Compose stack (if infra/docker-compose.yml exists) ---
if [ -f infra/docker-compose.yml ]; then
  echo "Starting Docker Compose stack..."
  docker compose --env-file "$REPO_ROOT/.env" -f infra/docker-compose.yml up -d
  echo "Waiting for Postgres..."
  for i in $(seq 1 30); do
    if pg_isready -h localhost -p 5433 -U sdlc_swarm 2>/dev/null; then
      echo "Postgres ready."
      break
    fi
    sleep 1
  done
  echo "Waiting for Langfuse..."
  for i in $(seq 1 60); do
    if curl -sf http://localhost:3110/ -o /dev/null 2>/dev/null; then
      echo "Langfuse ready."
      break
    fi
    sleep 2
  done
fi

# --- 10. Database migrations (if src/db/ exists) ---
if [ -d src/db ] && [ -f src/db/init_schema.py ]; then
  echo "Running database schema initialization..."
  # Load .env so DB credentials are available
  set -a; [ -f .env ] && source .env; set +a
  python -m src.db.init_schema || echo "Warning: DB schema init failed (may need Postgres)"
fi

echo "=== init.sh complete ==="

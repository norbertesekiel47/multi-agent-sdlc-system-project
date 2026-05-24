#!/bin/bash
# SDLC-Swarm sandbox proxy entrypoint
# Generates the allowlist file from ALLOWLIST_DOMAINS env var
# and starts Squid.

set -e

# Generate allowlist domains file from env var
ALLOWLIST_FILE="/etc/squid/allowlist_domains.txt"

if [ -n "${ALLOWLIST_DOMAINS:-}" ]; then
    echo "Generating allowlist from ALLOWLIST_DOMAINS..."
    echo "$ALLOWLIST_DOMAINS" | tr ',' '\n' > "$ALLOWLIST_FILE"
else
    echo "WARNING: ALLOWLIST_DOMAINS not set, using empty allowlist (all egress blocked)"
    echo "" > "$ALLOWLIST_FILE"
fi

echo "Allowlist contents:"
cat "$ALLOWLIST_FILE"

# Ensure log and cache directories exist with correct ownership
mkdir -p /var/log/squid
chown proxy:proxy /var/log/squid 2>/dev/null || true
mkdir -p /var/spool/squid
chown proxy:proxy /var/spool/squid 2>/dev/null || true

# Remove any stale PID file
rm -f /run/squid.pid

# Initialize Squid cache directory (required on first run)
# Use -N (no daemon) and wait for it to finish
squid -z 2>/dev/null || true
# Wait briefly for cache init to complete
sleep 1

# Remove any PID file left by the cache init
rm -f /run/squid.pid

# Start Squid in foreground
exec squid -N -d 1

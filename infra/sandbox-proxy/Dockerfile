# SDLC-Swarm sidecar sandbox proxy
# Squid-based HTTP proxy enforcing a registry allowlist.
# The allowlist is injected via ALLOWLIST_DOMAINS env var at runtime.

FROM ubuntu:24.04 AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends squid && \
    rm -rf /var/lib/apt/lists/*

COPY squid.conf /etc/squid/squid.conf
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 3128

ENTRYPOINT ["/entrypoint.sh"]

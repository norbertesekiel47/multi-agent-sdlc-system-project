# SDLC-Swarm sandbox base image
# Based on python:3.12-slim with the `patch` binary installed.
# The default python:3.12-slim image lacks `patch`, causing apply_diff
# to fail with exit code 127. This custom image fixes that.

FROM python:3.12-slim

# Install patch binary (required for apply_diff tool) and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends patch && \
    rm -rf /var/lib/apt/lists/*

# Verify patch is available
RUN which patch && patch --version

"""Janitor module for sweeping stale sandbox containers and networks.

This module is intended to be called periodically or after a crash
to clean up any orphaned Docker resources left behind by
unclean shutdowns (e.g., SIGKILL of the orchestrator process).
"""

from __future__ import annotations

import asyncio
import logging
import time

import docker
from docker.models.containers import Container

from src.sandbox.config import (
    JANITOR_MAX_AGE_SECONDS,
    SANDBOX_CONTAINER_PREFIX,
    SANDBOX_NETWORK_PREFIX,
    SANDBOX_PROXY_PREFIX,
)

logger = logging.getLogger(__name__)


async def sweep_stale_resources(
    max_age_seconds: int = JANITOR_MAX_AGE_SECONDS,
) -> int:
    """Remove all stale sdlc-swarm Docker containers and networks.

    A resource is considered stale if its creation timestamp is older
    than *max_age_seconds* ago.

    Returns the number of resources removed.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_sweep, max_age_seconds)


def _sync_sweep(max_age_seconds: int = JANITOR_MAX_AGE_SECONDS) -> int:
    """Synchronous sweep implementation."""
    client = docker.from_env()
    removed = 0
    now = time.time()
    cutoff = now - max_age_seconds

    # Remove stale containers with sdlc-swarm prefix
    for container in client.containers.list(all=True):
        name = container.name
        if name is None:
            continue
        if not (
            name.startswith(SANDBOX_CONTAINER_PREFIX)
            or name.startswith(SANDBOX_PROXY_PREFIX)
        ):
            continue
        created_ts = _container_created_timestamp(container)
        if created_ts is not None and created_ts < cutoff:
            try:
                container.remove(force=True)
                removed += 1
                logger.info("Janitor removed stale container %s", name)
            except docker.errors.NotFound:
                pass  # Already removed
            except Exception as e:
                logger.warning("Janitor failed to remove container %s: %s", name, e)
        elif created_ts is None:
            # Can't determine age — remove if prefix matches
            try:
                container.remove(force=True)
                removed += 1
                logger.info("Janitor removed unknown-age container %s", name)
            except docker.errors.NotFound:
                pass  # Already removed
            except Exception as e:
                logger.warning("Janitor failed to remove container %s: %s", name, e)

    # Remove stale networks with sdlc-swarm prefix
    for network in client.networks.list():
        name = network.name
        if name is None:
            continue
        if not name.startswith(SANDBOX_NETWORK_PREFIX):
            continue
        try:
            network.remove()
            removed += 1
            logger.info("Janitor removed stale network %s", name)
        except Exception as e:
            logger.warning("Janitor failed to remove network %s: %s", name, e)

    return removed


def _container_created_timestamp(
    container: Container,
) -> float | None:
    """Extract the Unix timestamp from a container's Created field."""
    try:
        created_str = container.attrs.get("Created", "")
        if not created_str:
            return None
        from datetime import datetime

        dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None

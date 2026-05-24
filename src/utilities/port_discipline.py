"""Port discipline utility for VAL-CROSS-030.

Verifies that mission services bind only to ports in the
allowed range [3100, 3199] plus Postgres on 5433.

Mission boundary ports (from AGENTS.md):
- 3100-3199: All mission services
- 5433: Postgres+pgvector (outside range deliberately)

Forbidden ports:
- 5432: Default Postgres slot
- 6379: Host Redis (other project)
- 5000, 7000: Apple ControlCenter / AirPlay
- 55068, 55247, 58059, 58208, 58105, 58106: User's Antigravity/Electron
"""

from __future__ import annotations

import subprocess
from typing import Any

# Mission-allowed port range
MISSION_PORT_MIN = 3100
MISSION_PORT_MAX = 3199

# Explicitly allowed ports outside the mission range
ALLOWED_EXTRA_PORTS: frozenset[int] = frozenset({5433})

# Explicitly forbidden ports
FORBIDDEN_PORTS: frozenset[int] = frozenset({
    5432,   # Default Postgres slot
    6379,   # Host Redis
    5000,   # Apple ControlCenter
    7000,   # Apple AirPlay
    55068,  # Antigravity/Electron
    55247,  # Antigravity/Electron
    58059,  # Antigravity/Electron
    58208,  # Antigravity/Electron
    58105,  # Antigravity/Electron
    58106,  # Antigravity/Electron
})

# Declared mission service ports (from services.yaml)
SERVICE_PORTS: frozenset[int] = frozenset({
    3100,   # FastAPI backend
    3101,   # Next.js dashboard
    3110,   # Langfuse self-hosted
    5433,   # Postgres+pgvector
})


def is_mission_port(port: int) -> bool:
    """Check if a port is in the allowed mission range.

    Args:
        port: Port number to check.

    Returns:
        True if the port is in the mission range [3100, 3199]
        or in the allowed extra ports set {5433}.
    """
    return (MISSION_PORT_MIN <= port <= MISSION_PORT_MAX) or port in ALLOWED_EXTRA_PORTS


def is_forbidden_port(port: int) -> bool:
    """Check if a port is in the forbidden set.

    Args:
        port: Port number to check.

    Returns:
        True if the port is explicitly forbidden.
    """
    return port in FORBIDDEN_PORTS


def get_listening_ports() -> set[int]:
    """Get the set of TCP ports currently in LISTEN state.

    Uses lsof to query listening TCP sockets on the host.

    Returns:
        Set of integer port numbers currently in LISTEN state.
        Returns an empty set if lsof is not available or fails.
    """
    try:
        result = subprocess.run(
            ["lsof", "-iTCP", "-sTCP:LISTEN", "-n", "-P"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return set()

    ports: set[int] = set()
    for line in result.stdout.splitlines()[1:]:  # Skip header
        parts = line.split()
        if len(parts) >= 9:
            addr_part = parts[8]
            try:
                port = int(addr_part.rsplit(":", 1)[-1])
                ports.add(port)
            except (ValueError, IndexError):
                continue
    return ports


def check_port_discipline() -> dict[str, Any]:
    """Check port discipline for VAL-CROSS-030.

    Verifies that:
    1. Mission service ports that are listening are in the allowed range
    2. No mission process is on a forbidden port
    3. Service ports don't collide with forbidden ports

    Returns:
        Dict with:
        - ok: bool — True if all checks pass
        - listening_ports: set[int] — all currently listening ports
        - mission_ports_listening: set[int] — mission service ports that are listening
        - violations: list[str] — descriptions of any violations found
    """
    listening_ports = get_listening_ports()

    violations: list[str] = []

    # Check that mission service ports that ARE listening are in the allowed range
    mission_ports_listening = listening_ports & SERVICE_PORTS
    for port in mission_ports_listening:
        if not is_mission_port(port):
            violations.append(
                f"Mission service on port {port} is outside allowed range"
            )

    # Check that no mission port is in the forbidden set
    for port in SERVICE_PORTS:
        if is_forbidden_port(port):
            violations.append(
                f"Declared service port {port} is in the forbidden set"
            )

    # Check that mission ports currently listening are not forbidden
    for port in mission_ports_listening:
        if is_forbidden_port(port):
            violations.append(
                f"Mission service listening on forbidden port {port}"
            )

    return {
        "ok": len(violations) == 0,
        "listening_ports": listening_ports,
        "mission_ports_listening": mission_ports_listening,
        "violations": violations,
    }

"""KitOps packaging module.

Packages prompts, configs, and pinned model references
as a versioned OCI/ModelKit artifact for reproducible deploys.

Usage:
    python -m sdlc_swarm.packaging.kitops build
    python -m sdlc_swarm.packaging.kitops build --output /tmp/artifact.tar
"""

from __future__ import annotations

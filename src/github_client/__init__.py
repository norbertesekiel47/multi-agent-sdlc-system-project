"""Hand-rolled GitHub client — replaces Composio with a self-owned, narrowly
scoped GitHub interface built on PyGithub.

Responsibilities:
  - Authenticated clone (PAT in URL, never logged)
  - Branch creation with the configured prefix (default ``sdlc-swarm/``)
  - Commit + push of changes
  - PR open via PyGithub
  - Issue read given repo + issue_number

Boundaries:
  - Only ever called by the Supervisor at a HITL-approved checkpoint
  - Token loaded from env, never echoed in logs
"""

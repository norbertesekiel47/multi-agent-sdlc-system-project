"""SDLC-Swarm agents — PydanticAI agents with typed IO.

Every agent boundary is Pydantic-typed; no free-form text
crosses boundaries.
"""

from src.agents.planner import PlannerDeps, planner, run_planner

__all__ = ["PlannerDeps", "planner", "run_planner"]

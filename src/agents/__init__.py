"""SDLC-Swarm agents — PydanticAI agents with typed IO.

Every agent boundary is Pydantic-typed; no free-form text
crosses boundaries.
"""

from src.agents.coder import CoderDeps, coder, run_coder
from src.agents.planner import PlannerDeps, planner, run_planner
from src.agents.qa import QADeps, qa, run_qa
from src.agents.reviewer import ReviewerDeps, reviewer, run_reviewer

__all__ = [
    "CoderDeps",
    "PlannerDeps",
    "QADeps",
    "ReviewerDeps",
    "coder",
    "planner",
    "qa",
    "reviewer",
    "run_coder",
    "run_planner",
    "run_qa",
    "run_reviewer",
]

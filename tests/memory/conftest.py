"""Shared fixtures and configuration for episodic memory tests."""

from __future__ import annotations

from dotenv import load_dotenv

# Load environment variables from .env so DB credentials are available
# before any test module imports EpisodicStore.
load_dotenv()

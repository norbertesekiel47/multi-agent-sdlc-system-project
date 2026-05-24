"""Database schema initializer.

Run via ``python -m src.db.init_schema`` or called from ``init.sh``.
Creates all episodic tables and indexes if they don't already exist.
"""

from __future__ import annotations

import asyncio
import logging
import os

import asyncpg
from dotenv import load_dotenv

from src.memory.episodic.schema import EPISODIC_SCHEMA_SQL
from src.memory.semantic.schema import SEMANTIC_SCHEMA_SQL

# Load environment variables from .env so DB credentials are available
# when this script is run standalone (not via init.sh).
load_dotenv()

logger = logging.getLogger(__name__)


def _dsn() -> str:
    """Build Postgres DSN from environment variables."""
    return (
        f"postgresql://{os.getenv('POSTGRES_USER', 'sdlc_swarm')}"
        f":{os.getenv('POSTGRES_PASSWORD', 'sdlc_swarm_dev')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}"
        f":{os.getenv('POSTGRES_PORT', '5433')}"
        f"/{os.getenv('POSTGRES_DB', 'sdlc_swarm')}"
    )


async def _init_schema() -> None:
    """Create episodic and semantic tables if they don't exist."""
    dsn = _dsn()
    logger.info("Connecting to Postgres at %s", dsn.split("@")[-1])
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(EPISODIC_SCHEMA_SQL)
        logger.info("Episodic schema initialized successfully.")
        await conn.execute(SEMANTIC_SCHEMA_SQL)
        logger.info("Semantic schema initialized successfully.")
    finally:
        await conn.close()


def main() -> None:
    """Entry point for ``python -m src.db.init_schema``."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_init_schema())


if __name__ == "__main__":
    main()

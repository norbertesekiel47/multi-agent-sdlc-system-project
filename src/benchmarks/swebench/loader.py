"""SWE-bench instance loader — fetches dataset rows from HuggingFace.

Pulls rows from ``princeton-nlp/SWE-bench_Lite`` (test split) and
filters to a configurable slice size (default 30 instances).

VAL-SWE-BENCH-001: Instance loader fetches typed SweBenchInstance rows.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]
from pydantic import ValidationError

from src.benchmarks.swebench.models import SweBenchInstance

logger = logging.getLogger(__name__)

# Dataset identifiers on HuggingFace
_DATASET_REPO = "princeton-nlp/SWE-bench_Lite"
_DATASET_SPLIT = "test"

# Default slice size
_DEFAULT_SLICE_SIZE = 30


class InstanceLoader:
    """Loads SWE-bench-Lite instances from HuggingFace.

    Usage::

        loader = InstanceLoader()
        instances = await loader.load(slice_size=30)

    The loader fetches the test split of princeton-nlp/SWE-bench_Lite,
    optionally filters to a specific slice, and returns typed
    ``SweBenchInstance`` objects.

    Rate limiting and caching are handled by the ``datasets`` library
    and HuggingFace Hub's built-in caching.  The HUGGINGFACE_TOKEN
    env var is used for authentication when available.
    """

    def __init__(
        self,
        *,
        dataset_repo: str = _DATASET_REPO,
        dataset_split: str = _DATASET_SPLIT,
        hf_token: str | None = None,
    ) -> None:
        self.dataset_repo = dataset_repo
        self.dataset_split = dataset_split
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")

    async def load(
        self,
        *,
        slice_size: int = _DEFAULT_SLICE_SIZE,
        instance_ids: list[str] | None = None,
    ) -> list[SweBenchInstance]:
        """Load SWE-bench instances, optionally filtering to a slice.

        Args:
            slice_size: Maximum number of instances to return.
                Defaults to 30.  Ignored if instance_ids is provided.
            instance_ids: Specific instance IDs to load.
                If provided, only these instances are returned,
                regardless of slice_size.

        Returns:
            List of typed SweBenchInstance objects.

        Raises:
            ValueError: If no instances can be loaded or all fail
                validation.
        """
        logger.info(
            "Loading SWE-bench instances from %s (%s split)",
            self.dataset_repo,
            self.dataset_split,
        )

        # Load dataset from HuggingFace (uses built-in caching)
        token = self.hf_token if self.hf_token else None
        dataset = load_dataset(
            self.dataset_repo,
            split=self.dataset_split,
            token=token,
            trust_remote_code=True,
        )

        # Convert to list of dicts for processing
        rows: list[dict[str, Any]] = list(dataset)
        logger.info("Fetched %d rows from dataset", len(rows))

        # Filter by specific instance IDs if provided
        if instance_ids is not None:
            id_set = set(instance_ids)
            rows = [r for r in rows if r.get("instance_id") in id_set]
            logger.info("Filtered to %d requested instance IDs", len(rows))
        else:
            # Apply slice size
            rows = rows[:slice_size]
            logger.info("Sliced to first %d instances", len(rows))

        # Parse each row into a typed SweBenchInstance
        instances: list[SweBenchInstance] = []
        parse_errors: list[str] = []

        for row in rows:
            try:
                instance = self._parse_row(row)
                instances.append(instance)
            except (ValidationError, KeyError, TypeError) as exc:
                instance_id = row.get("instance_id", "<unknown>")
                parse_errors.append(f"Failed to parse {instance_id}: {exc}")
                logger.warning("Failed to parse instance %s: %s", instance_id, exc)

        if parse_errors:
            logger.warning(
                "%d of %d instances failed to parse",
                len(parse_errors),
                len(rows),
            )

        if not instances and rows:
            msg = f"All {len(rows)} instances failed validation"
            raise ValueError(msg)

        logger.info("Successfully parsed %d instances", len(instances))
        return instances

    def _parse_row(self, row: dict[str, Any]) -> SweBenchInstance:
        """Parse a single dataset row into a SweBenchInstance.

        The HuggingFace dataset uses column names that map directly
        to the SweBenchInstance fields.  Some fields (FAIL_TO_PASS,
        PASS_TO_PASS) may be stored as JSON strings in the dataset
        and need to be deserialized.
        """
        # Make a mutable copy
        data = dict(row)

        # Handle FAIL_TO_PASS and PASS_TO_PASS which may be
        # stored as JSON strings in the dataset
        for field_name in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            val = data.get(field_name)
            if isinstance(val, str):
                import json

                try:
                    data[field_name] = json.loads(val)
                except json.JSONDecodeError:
                    data[field_name] = []
            elif not isinstance(val, list):
                data[field_name] = []

        # Ensure all expected fields are present with defaults
        data.setdefault("hints_text", "")
        data.setdefault("created_at", "")
        data.setdefault("version", "")
        data.setdefault("test_patch", "")
        data.setdefault("patch", "")

        return SweBenchInstance.model_validate(data)

    def list_available_ids(self) -> list[str]:
        """List all available instance IDs in the dataset.

        Useful for selecting specific instances before loading.

        Returns:
            Sorted list of instance ID strings.
        """
        token = self.hf_token if self.hf_token else None
        dataset = load_dataset(
            self.dataset_repo,
            split=self.dataset_split,
            token=token,
            trust_remote_code=True,
        )
        ids = [row["instance_id"] for row in dataset if "instance_id" in row]
        return sorted(ids)

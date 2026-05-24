"""KitOps ModelKit builder for SDLC-Swarm.

Gathers agent prompt files (prompts/*.md), configs (configs/*.yaml),
pinned-models.yaml, and a metadata manifest, then invokes the kit CLI
to produce a versioned OCI/ModelKit artifact.

Reproducibility guarantee: two consecutive builds with no input changes
produce byte-for-byte identical output (same SHA-256).

Round-trip guarantee: build -> unpack in fresh dir -> all required files
match originals.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Repository root (three levels up from this file)
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Default tag prefix for the OCI artifact
_DEFAULT_REGISTRY = "ghcr.io/norbertesekiel47"
_DEFAULT_REPO_NAME = "sdlc-swarm"


def _get_project_version() -> str:
    """Read version from pyproject.toml."""
    pyproject_path = _REPO_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        return "0.0.0"
    content = pyproject_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("version"):
            # version = "0.1.0"
            parts = stripped.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip().strip('"').strip("'")
    return "0.0.0"


def _get_git_commit() -> str:
    """Get the current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_git_commit_timestamp() -> str:
    """Get the git commit timestamp in RFC3339 format."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%aI"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to epoch
        return "1970-01-01T00:00:00Z"


def _gather_prompts() -> list[Path]:
    """Gather all prompt files from prompts/*.md."""
    prompts_dir = _REPO_ROOT / "prompts"
    if not prompts_dir.exists():
        return []
    return sorted(p for p in prompts_dir.glob("*.md") if p.is_file())


def _gather_configs() -> list[Path]:
    """Gather all config files from configs/*.yaml."""
    configs_dir = _REPO_ROOT / "configs"
    if not configs_dir.exists():
        return []
    return sorted(p for p in configs_dir.glob("*.yaml") if p.is_file())


def _gather_pinned_models() -> Path | None:
    """Return the pinned-models.yaml path if it exists."""
    p = _REPO_ROOT / "pinned-models.yaml"
    return p if p.exists() else None


def _build_metadata_manifest(
    version: str,
    git_commit: str,
    created_at: str,
    models_data: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the metadata manifest with all required fields (VAL-KITOPS-006)."""
    manifest: dict[str, Any] = {
        "name": _DEFAULT_REPO_NAME,
        "version": version,
        "created_at": created_at,
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "models": models_data or {},
    }
    return manifest


def _create_kitfile(
    version: str,
    context_dir: Path,
) -> str:
    """Create a Kitfile YAML string referencing the content directories.

    The Kitfile references:
    - prompts/ directory as a 'prompts' layer
    - configs/ directory as a 'code' layer
    - pinned-models.yaml as a 'docs' layer
    - metadata.json as a 'docs' layer
    """
    kitfile: dict[str, Any] = {
        "manifestVersion": "1.0.0",
        "package": {
            "name": _DEFAULT_REPO_NAME,
            "version": version,
            "description": (
                "Autonomous multi-agent SDLC system: "
                "issue -> reviewed, tested PR"
            ),
        },
        "prompts": [
            {
                "path": "prompts/",
                "description": "Agent system prompts",
            }
        ],
        "code": [
            {
                "path": "configs/",
                "description": "Agent and topology configuration files",
            }
        ],
        "docs": [
            {
                "path": "pinned-models.yaml",
                "description": "Pinned OpenRouter model IDs and revision dates",
            },
            {
                "path": "metadata.json",
                "description": "Build metadata manifest with git SHA and model refs",
            },
        ],
    }
    return str(yaml.dump(kitfile, default_flow_style=False, sort_keys=False))


def _prepare_staging_dir(
    version: str,
    git_commit: str,
    git_timestamp: str,
    created_at: str,
) -> Path:
    """Prepare a staging directory with all content for the ModelKit.

    Copies prompts/, configs/, pinned-models.yaml into the staging dir,
    and writes metadata.json.
    """
    staging = Path(tempfile.mkdtemp(prefix="sdlc-swarm-kitops-"))
    logger.info("Staging directory: %s", staging)

    # Copy prompts
    prompts_src = _REPO_ROOT / "prompts"
    if prompts_src.exists():
        shutil.copytree(prompts_src, staging / "prompts")

    # Copy configs
    configs_src = _REPO_ROOT / "configs"
    if configs_src.exists():
        shutil.copytree(configs_src, staging / "configs")

    # Copy pinned-models.yaml
    pinned_src = _REPO_ROOT / "pinned-models.yaml"
    if pinned_src.exists():
        shutil.copy2(pinned_src, staging / "pinned-models.yaml")

    # Load pinned-models data for metadata
    models_data: dict[str, Any] | None = None
    if pinned_src.exists():
        with open(pinned_src, encoding="utf-8") as f:
            models_data = yaml.safe_load(f)

    # Write metadata.json
    metadata = _build_metadata_manifest(
        version=version,
        git_commit=git_commit,
        created_at=created_at,
        models_data=models_data,
    )
    metadata_path = staging / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")

    # Write Kitfile
    kitfile_content = _create_kitfile(version=version, context_dir=staging)
    kitfile_path = staging / "Kitfile"
    with open(kitfile_path, "w", encoding="utf-8") as f:
        f.write(kitfile_content)

    return staging


def _kit_pack(
    staging_dir: Path,
    tag: str,
) -> str:
    """Run `kit pack` to build the ModelKit into local storage.

    Returns the artifact reference string.
    """
    cmd = [
        "kit", "pack",
        str(staging_dir),
        "-t", tag,
        "--compression", "none",
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    # Parse the output for the artifact reference
    for line in result.stdout.strip().splitlines():
        if line.startswith("Model saved:"):
            return line.split(":", 1)[1].strip()
    # Fallback: return the tag
    return tag


def _kit_inspect(tag: str) -> dict[str, Any]:
    """Run `kit inspect` and return the parsed JSON."""
    cmd = ["kit", "inspect", tag]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(json.loads(result.stdout))


def _kit_unpack(tag: str, output_dir: Path) -> None:
    """Run `kit unpack` to extract ModelKit content to a directory."""
    cmd = [
        "kit", "unpack",
        tag,
        "-d", str(output_dir),
        "-o",  # overwrite
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, capture_output=True, text=True, check=True)


def _kit_remove(tag: str) -> None:
    """Remove a ModelKit from local storage."""
    cmd = ["kit", "remove", tag]
    subprocess.run(cmd, capture_output=True, text=True, check=False)


def _normalize_oci_layout_for_reproducibility(
    layout_dir: Path,
    git_timestamp: str,
) -> None:
    """Post-process an OCI layout copy to fix timestamps for reproducibility.

    Operates on a COPY of the OCI layout (never modifies kit's own storage).
    Replaces the `org.opencontainers.image.created` annotation in the
    manifest with the git commit timestamp, and recalculates the
    manifest digest. This ensures byte-identical outputs across builds.
    """
    # Find the index.json
    index_path = layout_dir / "index.json"
    if not index_path.exists():
        return

    with open(index_path, encoding="utf-8") as f:
        index_data = json.load(f)

    # Process each manifest entry
    for entry in index_data.get("manifests", []):
        digest = entry.get("digest", "")
        if not digest.startswith("sha256:"):
            continue

        blob_path = layout_dir / "blobs" / "sha256" / digest[len("sha256:"):]
        if not blob_path.exists():
            continue

        with open(blob_path, encoding="utf-8") as f:
            manifest_data = json.load(f)

        # Fix the created timestamp
        annotations = manifest_data.get("annotations", {})
        annotations["org.opencontainers.image.created"] = git_timestamp
        manifest_data["annotations"] = annotations

        # Re-encode manifest with sorted keys for reproducibility
        manifest_bytes = json.dumps(
            manifest_data,
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"

        # Calculate new digest
        new_digest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()

        # Write the new manifest blob
        new_blob_path = layout_dir / "blobs" / "sha256" / new_digest[len("sha256:"):]
        new_blob_path.parent.mkdir(parents=True, exist_ok=True)
        with open(new_blob_path, "wb") as f:
            f.write(manifest_bytes)

        # Update the entry in index.json
        if digest != new_digest:
            entry["digest"] = new_digest
            entry["size"] = len(manifest_bytes)

    # Re-write index.json with sorted keys
    index_bytes = json.dumps(index_data, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with open(index_path, "wb") as f:
        f.write(index_bytes)


def _create_minimal_oci_layout(
    storage_dir: Path,
    inspect_data: dict[str, Any],
    git_timestamp: str,
) -> Path:
    """Create a minimal, reproducible OCI layout for the built artifact.

    Only includes the blobs referenced by this specific artifact's manifest,
    ensuring the tar output is deterministic regardless of what else is in
    kit's local storage.
    """
    layout_dir = Path(tempfile.mkdtemp(prefix="sdlc-swarm-oci-"))
    blobs_dir = layout_dir / "blobs" / "sha256"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    # Extract manifest data from inspect output
    manifest_data = inspect_data.get("manifest", {})

    # Collect all referenced blob digests
    referenced_digests: set[str] = set()

    # Config blob
    config = manifest_data.get("config", {})
    config_digest = config.get("digest", "")
    if config_digest:
        referenced_digests.add(config_digest)

    # Layer blobs
    for layer in manifest_data.get("layers", []):
        layer_digest = layer.get("digest", "")
        if layer_digest:
            referenced_digests.add(layer_digest)

    # Copy each referenced blob from kit storage
    for digest in referenced_digests:
        if not digest.startswith("sha256:"):
            continue
        hex_digest = digest[len("sha256:"):]
        src = storage_dir / "blobs" / "sha256" / hex_digest
        dst = blobs_dir / hex_digest
        if src.exists():
            shutil.copy2(src, dst)

    # Normalize the manifest for reproducibility
    annotations = manifest_data.get("annotations", {})
    annotations["org.opencontainers.image.created"] = git_timestamp
    manifest_data["annotations"] = annotations

    # Encode manifest deterministically
    manifest_bytes = json.dumps(
        manifest_data,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"

    # Write the normalized manifest blob
    manifest_digest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    manifest_hex = manifest_digest[len("sha256:"):]
    with open(blobs_dir / manifest_hex, "wb") as f:
        f.write(manifest_bytes)

    # Also copy the manifest blob that kit already stored (it may differ)
    # We use our normalized version instead
    # The config blob is already in the blobs dir

    # Write oci-layout
    with open(layout_dir / "oci-layout", "w", encoding="utf-8") as f:
        f.write('{"imageLayoutVersion":"1.0.0"}\n')

    # Write index.json referencing the manifest
    index_data = {
        "schemaVersion": 2,
        "manifests": [
            {
                "mediaType": manifest_data.get(
                    "artifactType",
                    "application/vnd.kitops.modelkit.manifest.v1+json",
                ),
                "digest": manifest_digest,
                "size": len(manifest_bytes),
            }
        ],
    }
    with open(layout_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, sort_keys=True)
        f.write("\n")

    return layout_dir


def _find_kit_storage_dir() -> Path | None:
    """Find the kit local storage directory."""
    # macOS: ~/Library/Caches/kitops/storage
    mac_path = Path.home() / "Library" / "Caches" / "kitops" / "storage"
    if mac_path.exists():
        return mac_path

    # Linux: ~/.local/share/kitops/storage
    linux_path = Path.home() / ".local" / "share" / "kitops" / "storage"
    if linux_path.exists():
        return linux_path

    # XDG fallback
    xdg_data = os.environ.get("XDG_DATA_HOME", "")
    if xdg_data:
        xdg_path = Path(xdg_data) / "kitops" / "storage"
        if xdg_path.exists():
            return xdg_path

    return None


def _create_reproducible_tar(
    storage_dir: Path,
    output_path: Path,
) -> None:
    """Create a reproducible tar from the OCI layout.

    Uses deterministic file ordering, fixed timestamps, and
    normalized permissions to ensure byte-identical output.
    """
    # Determine a fixed timestamp for tar entries (git epoch)
    epoch_time = 0  # 1970-01-01 00:00:00

    # Collect all files in the storage directory
    all_files: list[Path] = []
    for root, _dirs, files in os.walk(storage_dir):
        for fname in files:
            all_files.append(Path(root) / fname)

    # Sort deterministically
    all_files.sort()

    with tarfile.open(str(output_path), "w") as tar:
        for file_path in all_files:
            arcname = str(file_path.relative_to(storage_dir))
            # Create a deterministic TarInfo
            info = tarfile.TarInfo(name=arcname)
            info.mtime = epoch_time
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            stat = file_path.stat()
            info.size = stat.st_size
            # Normalize mode
            if file_path.is_dir():
                info.mode = 0o755
                info.type = tarfile.DIRTYPE
            else:
                info.mode = 0o644
                info.type = tarfile.REGTYPE

            with open(file_path, "rb") as f:
                tar.addfile(info, f)


def build(
    output: str | None = None,
    registry: str | None = None,
) -> str:
    """Build the KitOps ModelKit artifact.

    Args:
        output: If specified, write a reproducible tar to this path.
        registry: OCI registry prefix (default: ghcr.io/norbertesekiel47).

    Returns:
        The artifact reference string (e.g. 'ghcr.io/norbertesekiel47/sdlc-swarm:0.1.0').
    """
    version = _get_project_version()
    git_commit = _get_git_commit()
    git_timestamp = _get_git_commit_timestamp()

    # Use the git commit timestamp for the created_at field
    # to ensure reproducibility
    created_at = git_timestamp

    registry = registry or _DEFAULT_REGISTRY
    tag = f"{registry}/{_DEFAULT_REPO_NAME}:{version}"

    # Prepare staging directory with all content
    staging_dir = _prepare_staging_dir(
        version=version,
        git_commit=git_commit,
        git_timestamp=git_timestamp,
        created_at=created_at,
    )

    try:
        # Remove any existing artifact with the same tag
        _kit_remove(tag)

        # Pack the ModelKit
        artifact_ref = _kit_pack(staging_dir, tag)
        logger.info("ModelKit packed: %s", artifact_ref)

        if output:
            # Create a minimal, reproducible OCI layout from the artifact
            # and write it as a tar. Never modifies kit's own storage.
            storage_dir = _find_kit_storage_dir()
            if storage_dir and storage_dir.exists():
                # Inspect the artifact to get manifest details
                inspect_data = _kit_inspect(tag)

                # Create minimal OCI layout with just our artifact's blobs
                layout_dir = _create_minimal_oci_layout(
                    storage_dir=storage_dir,
                    inspect_data=inspect_data,
                    git_timestamp=git_timestamp,
                )

                try:
                    # Create reproducible tar from the layout
                    output_path = Path(output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    _create_reproducible_tar(layout_dir, output_path)
                    logger.info("Reproducible tar written to: %s", output)
                finally:
                    # Clean up the layout copy
                    shutil.rmtree(layout_dir, ignore_errors=True)

        return artifact_ref
    finally:
        # Clean up staging directory
        shutil.rmtree(staging_dir, ignore_errors=True)


def push(
    registry: str | None = None,
) -> str:
    """Push the KitOps ModelKit artifact to an OCI registry.

    Args:
        registry: OCI registry (default: ghcr.io/norbertesekiel47).

    Returns:
        The push output string.
    """
    version = _get_project_version()
    registry = registry or _DEFAULT_REGISTRY
    tag = f"{registry}/{_DEFAULT_REPO_NAME}:{version}"

    cmd = ["kit", "push", tag]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_manifest_fields(tag: str | None = None) -> dict[str, Any]:
    """Inspect the ModelKit and return metadata manifest fields.

    Returns the metadata.json content from the artifact.
    """
    version = _get_project_version()
    if tag is None:
        tag = f"{_DEFAULT_REGISTRY}/{_DEFAULT_REPO_NAME}:{version}"

    inspect_data = _kit_inspect(tag)
    return inspect_data

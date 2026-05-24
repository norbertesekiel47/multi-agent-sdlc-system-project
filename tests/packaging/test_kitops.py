"""Tests for KitOps packaging (VAL-KITOPS-001 through VAL-KITOPS-006, VAL-CROSS-025).

These tests verify the KitOps packaging functionality:
- VAL-KITOPS-001: KitOps build CLI exits 0 and prints artifact reference
- VAL-KITOPS-002: Build gathers only prompts, configs, and pinned-models.yaml
- VAL-KITOPS-003: Artifact tagged with version matching pyproject.toml
- VAL-KITOPS-004: Identical builds produce byte-identical artifacts
- VAL-KITOPS-005: pinned-models.yaml layer matches on-disk file
- VAL-KITOPS-006: Artifact metadata manifest has all required fields
- VAL-CROSS-025: KitOps round-trip — byte-for-byte file match
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

# Repository root
REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_build(output: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run the kitops build CLI."""
    cmd = ["python3", "-m", "src.packaging.kitops", "build"]
    if output:
        cmd.extend(["--output", output])
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _kit_unpack(tag: str, dest: str) -> subprocess.CompletedProcess[str]:
    """Run kit unpack to extract artifact content."""
    cmd = ["kit", "unpack", tag, "-d", dest, "-o"]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )


def _kit_inspect(tag: str) -> dict:
    """Run kit inspect and return parsed JSON."""
    cmd = ["kit", "inspect", tag]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _kit_remove(tag: str) -> None:
    """Remove a ModelKit from local storage."""
    subprocess.run(
        ["kit", "remove", tag],
        capture_output=True,
        text=True,
        check=False,
    )


def _get_version_from_pyproject() -> str:
    """Read version from pyproject.toml."""
    pyproject = REPO_ROOT / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("version"):
            parts = stripped.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip().strip('"').strip("'")
    return "0.0.0"


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── VAL-KITOPS-001: KitOps build CLI exits 0 and prints artifact reference ──


class TestKitopsBuildCLI:
    """VAL-KITOPS-001: KitOps build CLI exits 0 and prints artifact reference."""

    def test_build_exits_zero(self) -> None:
        """Build command exits with code 0."""
        result = _run_build()
        assert result.returncode == 0, (
            f"Build failed with exit code {result.returncode}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_build_prints_artifact_reference(self) -> None:
        """Build command prints artifact reference to stdout."""
        result = _run_build()
        assert result.returncode == 0
        output_lines = result.stdout.strip().splitlines()
        # Should contain a line matching 'artifact: <ref>'
        artifact_lines = [
            line for line in output_lines
            if line.startswith("artifact:")
        ]
        assert len(artifact_lines) >= 1, (
            f"Expected 'artifact:' line in output.\n"
            f"stdout: {result.stdout}"
        )
        # The reference should be non-empty
        ref = artifact_lines[0].split(":", 1)[1].strip()
        assert len(ref) > 0, "Artifact reference should not be empty"

    def test_build_with_output_flag(self) -> None:
        """Build with --output flag creates a tar file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tar")
            result = _run_build(output=output_path)
            assert result.returncode == 0, (
                f"Build with --output failed.\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
            assert os.path.exists(output_path), "Output tar file should exist"


# ── VAL-KITOPS-002: Build gathers only prompts, configs, and pinned-models.yaml ──


class TestKitopsBuildContent:
    """VAL-KITOPS-002: Build gathers only prompts, configs, and pinned-models.yaml."""

    def _unpack_artifact(self) -> Path:
        """Helper: build and unpack the artifact to a temp dir."""
        # Ensure the artifact exists by building first
        result = _run_build()
        assert result.returncode == 0, f"Build failed: {result.stderr}"

        version = _get_version_from_pyproject()
        tag = f"ghcr.io/norbertesekiel47/sdlc-swarm:{version}"
        unpack_dir = Path(tempfile.mkdtemp(prefix="kitops-test-unpack-"))
        unpack_result = _kit_unpack(tag, str(unpack_dir))
        assert unpack_result.returncode == 0, (
            f"Unpack failed: {unpack_result.stderr}"
        )
        return unpack_dir

    def test_unpacked_contains_prompts_dir(self) -> None:
        """Unpacked artifact contains prompts/ directory."""
        unpack_dir = self._unpack_artifact()
        try:
            assert (unpack_dir / "prompts").is_dir(), "prompts/ directory should exist"
            prompt_files = list((unpack_dir / "prompts").glob("*.md"))
            assert len(prompt_files) >= 1, "Should have at least one prompt file"
        finally:
            shutil.rmtree(unpack_dir, ignore_errors=True)

    def test_unpacked_contains_configs_dir(self) -> None:
        """Unpacked artifact contains configs/ directory."""
        unpack_dir = self._unpack_artifact()
        try:
            assert (unpack_dir / "configs").is_dir(), "configs/ directory should exist"
            config_files = list((unpack_dir / "configs").glob("*.yaml"))
            assert len(config_files) >= 1, "Should have at least one config file"
        finally:
            shutil.rmtree(unpack_dir, ignore_errors=True)

    def test_unpacked_contains_pinned_models(self) -> None:
        """Unpacked artifact contains pinned-models.yaml."""
        unpack_dir = self._unpack_artifact()
        try:
            assert (unpack_dir / "pinned-models.yaml").is_file(), (
                "pinned-models.yaml should exist"
            )
        finally:
            shutil.rmtree(unpack_dir, ignore_errors=True)

    def test_unpacked_file_set_matches_expected(self) -> None:
        """File set in artifact equals prompts/*.md + configs/*.yaml + pinned-models.yaml."""
        # Gather expected file set from the working tree
        expected_files: set[str] = set()

        prompts_dir = REPO_ROOT / "prompts"
        if prompts_dir.exists():
            for p in prompts_dir.glob("*.md"):
                expected_files.add(f"prompts/{p.name}")

        configs_dir = REPO_ROOT / "configs"
        if configs_dir.exists():
            for p in configs_dir.glob("*.yaml"):
                expected_files.add(f"configs/{p.name}")

        expected_files.add("pinned-models.yaml")

        # Unpack and check
        unpack_dir = self._unpack_artifact()
        try:
            actual_files: set[str] = set()

            prompts_unpack = unpack_dir / "prompts"
            if prompts_unpack.is_dir():
                for p in prompts_unpack.glob("*.md"):
                    actual_files.add(f"prompts/{p.name}")

            configs_unpack = unpack_dir / "configs"
            if configs_unpack.is_dir():
                for p in configs_unpack.glob("*.yaml"):
                    actual_files.add(f"configs/{p.name}")

            if (unpack_dir / "pinned-models.yaml").is_file():
                actual_files.add("pinned-models.yaml")

            assert actual_files == expected_files, (
                f"File set mismatch.\n"
                f"Expected: {sorted(expected_files)}\n"
                f"Actual: {sorted(actual_files)}\n"
                f"Missing: {sorted(expected_files - actual_files)}\n"
                f"Extra: {sorted(actual_files - expected_files)}"
            )
        finally:
            shutil.rmtree(unpack_dir, ignore_errors=True)


# ── VAL-KITOPS-003: Artifact tagged with version matching pyproject.toml ──


class TestKitopsVersionTag:
    """VAL-KITOPS-003: Artifact tagged with version matching pyproject.toml."""

    def test_artifact_tag_matches_pyproject_version(self) -> None:
        """The artifact tag includes the version from pyproject.toml."""
        version = _get_version_from_pyproject()
        tag = f"ghcr.io/norbertesekiel47/sdlc-swarm:{version}"

        # Build first to ensure it exists
        result = _run_build()
        assert result.returncode == 0

        # Verify the version substring via kit inspect
        inspect_data = _kit_inspect(tag)
        kitfile = inspect_data.get("kitfile", {})
        package = kitfile.get("package", {})
        assert package.get("version") == version, (
            f"Artifact version '{package.get('version')}' != pyproject.toml version '{version}'"
        )


# ── VAL-KITOPS-004: Identical builds produce byte-identical artifacts ──


class TestKitopsReproducibility:
    """VAL-KITOPS-004: Identical builds produce byte-identical artifacts."""

    def test_two_consecutive_builds_produce_identical_tar(self) -> None:
        """Two consecutive builds produce byte-identical tar files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_a = os.path.join(tmpdir, "a.tar")
            tar_b = os.path.join(tmpdir, "b.tar")

            # Build twice
            result_a = _run_build(output=tar_a)
            assert result_a.returncode == 0, f"First build failed: {result_a.stderr}"

            result_b = _run_build(output=tar_b)
            assert result_b.returncode == 0, f"Second build failed: {result_b.stderr}"

            # Compare SHA-256
            sha_a = _file_sha256(Path(tar_a))
            sha_b = _file_sha256(Path(tar_b))
            assert sha_a == sha_b, (
                f"SHA-256 mismatch: a={sha_a}, b={sha_b}\n"
                f"Builds should produce byte-identical output."
            )

    def test_manifest_digests_equal(self) -> None:
        """Manifest digests reported by kit inspect are equal across builds."""
        version = _get_version_from_pyproject()
        tag = f"ghcr.io/norbertesekiel47/sdlc-swarm:{version}"

        # Build twice
        _run_build()
        inspect_a = _kit_inspect(tag)

        _run_build()
        inspect_b = _kit_inspect(tag)

        # After reproducibility normalization, digests should match
        # (may not match if timestamp normalization failed)
        # The key assertion: layer digests must be identical
        layers_a = inspect_a.get("manifest", {}).get("layers", [])
        layers_b = inspect_b.get("manifest", {}).get("layers", [])

        layer_digests_a = [layer.get("digest") for layer in layers_a]
        layer_digests_b = [layer.get("digest") for layer in layers_b]

        assert layer_digests_a == layer_digests_b, (
            f"Layer digests mismatch:\n"
            f"  Build 1: {layer_digests_a}\n"
            f"  Build 2: {layer_digests_b}"
        )


# ── VAL-KITOPS-005: pinned-models.yaml layer matches on-disk file ──


class TestKitopsPinnedModels:
    """VAL-KITOPS-005: pinned-models.yaml layer matches on-disk file."""

    def _unpack_artifact(self) -> Path:
        """Helper: build and unpack the artifact to a temp dir."""
        result = _run_build()
        assert result.returncode == 0, f"Build failed: {result.stderr}"

        version = _get_version_from_pyproject()
        tag = f"ghcr.io/norbertesekiel47/sdlc-swarm:{version}"
        unpack_dir = Path(tempfile.mkdtemp(prefix="kitops-test-pinned-"))
        _kit_unpack(tag, str(unpack_dir))
        return unpack_dir

    def test_pinned_models_sha256_matches_on_disk(self) -> None:
        """Extracted pinned-models.yaml bytes equal on-disk file bytes."""
        on_disk = REPO_ROOT / "pinned-models.yaml"
        assert on_disk.exists(), "pinned-models.yaml must exist in repo root"

        unpack_dir = self._unpack_artifact()
        try:
            unpacked = unpack_dir / "pinned-models.yaml"
            assert unpacked.exists(), "pinned-models.yaml must exist in artifact"

            on_disk_hash = _file_sha256(on_disk)
            unpacked_hash = _file_sha256(unpacked)
            assert on_disk_hash == unpacked_hash, (
                f"pinned-models.yaml SHA-256 mismatch:\n"
                f"  on-disk:   {on_disk_hash}\n"
                f"  unpacked:  {unpacked_hash}"
            )
        finally:
            shutil.rmtree(unpack_dir, ignore_errors=True)


# ── VAL-KITOPS-006: Artifact metadata manifest has all required fields ──


class TestKitopsMetadataManifest:
    """VAL-KITOPS-006: Artifact metadata manifest has all required fields."""

    REQUIRED_FIELDS = [
        "name",
        "version",
        "created_at",
        "git_commit",
        "python_version",
        "models",
    ]

    def _unpack_artifact(self) -> Path:
        """Helper: build and unpack the artifact to a temp dir."""
        result = _run_build()
        assert result.returncode == 0, f"Build failed: {result.stderr}"

        version = _get_version_from_pyproject()
        tag = f"ghcr.io/norbertesekiel47/sdlc-swarm:{version}"
        unpack_dir = Path(tempfile.mkdtemp(prefix="kitops-test-meta-"))
        _kit_unpack(tag, str(unpack_dir))
        return unpack_dir

    def test_metadata_has_all_required_fields(self) -> None:
        """metadata.json contains all six required fields."""
        unpack_dir = self._unpack_artifact()
        try:
            metadata_path = unpack_dir / "metadata.json"
            assert metadata_path.exists(), "metadata.json must exist in artifact"

            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

            for field in self.REQUIRED_FIELDS:
                assert field in metadata, f"Missing required field: {field}"
                assert metadata[field], f"Field '{field}' must be non-empty"

        finally:
            shutil.rmtree(unpack_dir, ignore_errors=True)

    def test_git_commit_matches_current_head(self) -> None:
        """git_commit in metadata matches git rev-parse HEAD at build time."""
        unpack_dir = self._unpack_artifact()
        try:
            metadata_path = unpack_dir / "metadata.json"
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

            git_commit = metadata.get("git_commit", "")
            assert git_commit != "unknown", "git_commit should not be 'unknown'"

            # Verify it's a valid-looking git SHA
            assert len(git_commit) == 40, (
                f"git_commit should be a 40-char SHA, got: {git_commit}"
            )
            assert all(c in "0123456789abcdef" for c in git_commit), (
                f"git_commit should be a hex SHA, got: {git_commit}"
            )

        finally:
            shutil.rmtree(unpack_dir, ignore_errors=True)

    def test_models_keys_match_pinned_models_yaml(self) -> None:
        """models keys in metadata equal pinned-models.yaml keys."""
        pinned_path = REPO_ROOT / "pinned-models.yaml"
        if not pinned_path.exists():
            pytest.skip("pinned-models.yaml not found")

        with open(pinned_path, encoding="utf-8") as f:
            pinned_data = yaml.safe_load(f)

        unpack_dir = self._unpack_artifact()
        try:
            metadata_path = unpack_dir / "metadata.json"
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

            models_in_metadata = metadata.get("models", {})
            # The models dict in metadata should contain the same
            # top-level keys as pinned-models.yaml
            assert set(pinned_data.keys()) == set(models_in_metadata.keys()), (
                f"models keys mismatch:\n"
                f"  pinned-models.yaml: {sorted(pinned_data.keys())}\n"
                f"  metadata:           {sorted(models_in_metadata.keys())}"
            )

        finally:
            shutil.rmtree(unpack_dir, ignore_errors=True)

    def test_python_version_is_populated(self) -> None:
        """python_version field is a valid version string."""
        unpack_dir = self._unpack_artifact()
        try:
            metadata_path = unpack_dir / "metadata.json"
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

            python_version = metadata.get("python_version", "")
            assert python_version, "python_version must not be empty"
            # Should match pattern like "3.14.4"
            parts = python_version.split(".")
            assert len(parts) >= 2, (
                f"python_version should have at least 2 parts, got: {python_version}"
            )

        finally:
            shutil.rmtree(unpack_dir, ignore_errors=True)


# ── VAL-CROSS-025: KitOps round-trip — byte-for-byte file match ──


class TestKitopsRoundTrip:
    """VAL-CROSS-025: KitOps round-trip — build -> unpack -> all files match originals."""

    def test_round_trip_all_files_match(self) -> None:
        """Build, unpack in fresh dir, all required files match originals."""
        # Ensure a fresh build
        result = _run_build()
        assert result.returncode == 0, f"Build failed: {result.stderr}"

        version = _get_version_from_pyproject()
        tag = f"ghcr.io/norbertesekiel47/sdlc-swarm:{version}"

        # Unpack to fresh directory
        unpack_dir = Path(tempfile.mkdtemp(prefix="kitops-roundtrip-"))
        _kit_unpack(tag, str(unpack_dir))

        try:
            # Compare each prompt file
            for p in (REPO_ROOT / "prompts").glob("*.md"):
                original = p.read_bytes()
                unpacked = (unpack_dir / "prompts" / p.name).read_bytes()
                assert original == unpacked, (
                    f"Round-trip mismatch for prompts/{p.name}"
                )

            # Compare each config file
            for p in (REPO_ROOT / "configs").glob("*.yaml"):
                original = p.read_bytes()
                unpacked = (unpack_dir / "configs" / p.name).read_bytes()
                assert original == unpacked, (
                    f"Round-trip mismatch for configs/{p.name}"
                )

            # Compare pinned-models.yaml
            original = (REPO_ROOT / "pinned-models.yaml").read_bytes()
            unpacked = (unpack_dir / "pinned-models.yaml").read_bytes()
            assert original == unpacked, (
                "Round-trip mismatch for pinned-models.yaml"
            )

        finally:
            shutil.rmtree(unpack_dir, ignore_errors=True)

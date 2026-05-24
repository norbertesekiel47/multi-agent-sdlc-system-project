"""CLI entry point for KitOps packaging.

Usage:
    python -m sdlc_swarm.packaging.kitops build [--output PATH] [--registry REGISTRY]
    python -m sdlc_swarm.packaging.kitops push [--registry REGISTRY]
"""

from __future__ import annotations

import argparse

from src.packaging.kitops.builder import build, push


def _main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="sdlc_swarm.packaging.kitops",
        description="KitOps packaging for SDLC-Swarm",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build subcommand
    build_parser = subparsers.add_parser(
        "build",
        help="Build a versioned OCI/ModelKit artifact",
    )
    build_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Write a reproducible tar to this path",
    )
    build_parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="OCI registry prefix (default: ghcr.io/norbertesekiel47)",
    )

    # push subcommand
    push_parser = subparsers.add_parser(
        "push",
        help="Push the ModelKit artifact to an OCI registry",
    )
    push_parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="OCI registry (default: ghcr.io/norbertesekiel47)",
    )

    args = parser.parse_args(argv)

    if args.command == "build":
        artifact_ref = build(
            output=args.output,
            registry=args.registry,
        )
        # Print artifact reference (VAL-KITOPS-001)
        print(f"artifact: {artifact_ref}")

    elif args.command == "push":
        result = push(registry=args.registry)
        print(result)


if __name__ == "__main__":
    _main()

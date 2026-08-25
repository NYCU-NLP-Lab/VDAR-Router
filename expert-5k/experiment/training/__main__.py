from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import run_training_pipeline
from .settings import get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.training",
        description="Build Stage 2 router training artifacts.",
    )
    parser.add_argument("--train-manifest-path", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--variant")
    parser.add_argument(
        "--config-json",
        default="{}",
        help="JSON object string passed to the router trainer as training config.",
    )
    parser.add_argument(
        "--runtime-overrides-json",
        default="{}",
        help="JSON object string passed to the router trainer as runtime_overrides.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    variant = (
        args.variant or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    config = json.loads(args.config_json)
    runtime_overrides = json.loads(args.runtime_overrides_json)
    if not isinstance(config, dict):
        raise ValueError("--config-json must decode to a JSON object.")
    if not isinstance(runtime_overrides, dict):
        raise ValueError("--runtime-overrides-json must decode to a JSON object.")

    result = run_training_pipeline(
        train_manifest_path=Path(args.train_manifest_path),
        trainer_path=args.trainer,
        variant=variant,
        config=config,
        runtime_overrides=runtime_overrides,
        settings=get_settings(),
    )
    print(
        json.dumps(
            {
                "artifact_root": str(result.artifact_root),
                "manifest": str(result.manifest_path),
                "router_family": result.training_manifest.router_family,
                "router_variant": result.training_manifest.router_variant,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

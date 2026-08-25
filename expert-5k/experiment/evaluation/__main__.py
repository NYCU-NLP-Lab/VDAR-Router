from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from rich.console import Console

from .cli_output import build_warning_text
from .contracts import RouterTargetInput
from .pipeline import render_summary_table, run_evaluation_pipeline
from .settings import get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.evaluation",
        description="Evaluate one or more Stage 2 routers against a Stage 1 dataset.",
    )
    parser.add_argument("--dataset-manifest-path", required=True)
    parser.add_argument(
        "--router-manifest-path",
        action="append",
        dest="router_manifest_paths",
    )
    parser.add_argument("--run-label")
    parser.add_argument(
        "--router-target-json",
        action="append",
        help=(
            "JSON object describing one router target. It must contain "
            "router_manifest_path and may include runtime_config and target_suffix."
        ),
    )
    parser.add_argument(
        "--router-targets-json-file",
        help="Path to a JSON file containing an array of router target objects.",
    )
    parser.add_argument(
        "--fail-on-router-setup-error",
        action="store_true",
        help="Raise immediately on manifest_load or router_setup failures.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip router report markdown generation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_label = (
        args.run_label or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    router_targets = _build_router_targets_cli_input(
        router_manifest_paths=[
            Path(path) for path in (args.router_manifest_paths or [])
        ],
        inline_router_target_jsons=args.router_target_json or [],
        router_targets_json_file=args.router_targets_json_file,
    )
    result = run_evaluation_pipeline(
        dataset_manifest_path=Path(args.dataset_manifest_path),
        router_targets=cast(list[RouterTargetInput | dict[str, Any]], router_targets),
        run_label=run_label,
        fail_on_router_setup_error=args.fail_on_router_setup_error,
        report_enabled=not args.no_report,
        settings=get_settings(),
    )
    Console(width=200).print(render_summary_table(result.router_summaries))
    warning_text = build_warning_text(result.router_summaries)
    if warning_text:
        print(warning_text, file=sys.stderr)
    return 0


def _resolve_json_input(
    *,
    inline_value: str | None,
    file_path: str | None,
    default_value: str,
) -> str:
    if inline_value is not None:
        return inline_value
    if file_path is not None:
        return Path(file_path).read_text(encoding="utf-8")
    return default_value


def _build_router_targets_cli_input(
    *,
    router_manifest_paths: list[Path],
    inline_router_target_jsons: list[str],
    router_targets_json_file: str | None,
) -> list[dict[str, Any]]:
    router_targets: list[dict[str, Any]] = [
        {"router_manifest_path": str(path)} for path in router_manifest_paths
    ]
    if router_targets_json_file is not None:
        file_payload = json.loads(
            Path(router_targets_json_file).read_text(encoding="utf-8")
        )
        if not isinstance(file_payload, list):
            raise ValueError("--router-targets-json-file must decode to a JSON array.")
        for index, entry in enumerate(file_payload):
            router_targets.append(
                _require_json_object(
                    entry,
                    description=f"--router-targets-json-file[{index}]",
                )
            )
    for index, inline_value in enumerate(inline_router_target_jsons):
        entry = json.loads(inline_value)
        router_targets.append(
            _require_json_object(
                entry,
                description=f"--router-target-json[{index}]",
            )
        )
    if not router_targets:
        raise ValueError(
            "At least one --router-manifest-path or router target is required."
        )
    return router_targets


def _require_json_object(value: Any, *, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must decode to an object.")

    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{description} must use string keys.")
        normalized[key] = item
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())

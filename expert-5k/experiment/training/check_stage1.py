from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.training.check_stage1",
        description="Validate Stage 1 JSONL rows before Stage 2 training.",
    )
    parser.add_argument("--train-path", required=True)
    return parser


def inspect_stage1_rows(
    train_path: Path, *, manifest_path: Path | None = None
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    total_rows = 0
    has_output = _read_has_output_capability(train_path, manifest_path=manifest_path)

    with train_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            total_rows += 1

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                issues.append(
                    {
                        "line": line_number,
                        "row_id": None,
                        "field": "<json>",
                        "message": f"invalid JSON: {exc.msg}",
                    }
                )
                continue

            if not isinstance(payload, dict):
                issues.append(
                    {
                        "line": line_number,
                        "row_id": None,
                        "field": "<row>",
                        "message": "row must be a JSON object",
                    }
                )
                continue

            row_id = payload.get("id") if isinstance(payload.get("id"), str) else None
            output = payload.get("output")
            if not isinstance(output, str):
                issues.append(
                    {
                        "line": line_number,
                        "row_id": row_id,
                        "field": "output",
                        "message": "output must be a string",
                    }
                )
                continue
            if not output.strip():
                if has_output:
                    issues.append(
                        {
                            "line": line_number,
                            "row_id": row_id,
                            "field": "output",
                            "message": "output must be a non-empty string",
                        }
                    )

    return {
        "train_path": str(train_path),
        "total_rows": total_rows,
        "invalid_rows": len(issues),
        "issues": issues,
    }


def _read_has_output_capability(
    train_path: Path, *, manifest_path: Path | None = None
) -> bool:
    resolved_manifest_path = manifest_path or train_path.parent / "manifest.json"
    if not resolved_manifest_path.exists():
        return True

    manifest = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config")
    if not isinstance(config, dict):
        return True
    source_capabilities = config.get("source_capabilities")
    if source_capabilities is None:
        return True
    if not isinstance(source_capabilities, dict):
        raise ValueError(
            "Stage 1 manifest config.source_capabilities must be an object when present."
        )
    has_output = source_capabilities.get("has_output")
    if has_output is None:
        return True
    if not isinstance(has_output, bool):
        raise ValueError(
            "Stage 1 manifest config.source_capabilities.has_output must be a boolean when present."
        )
    return has_output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = inspect_stage1_rows(Path(args.train_path))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

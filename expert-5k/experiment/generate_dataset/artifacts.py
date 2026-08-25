from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import (
    BuildInputs,
    CanonicalRow,
    DatasetBuildConfig,
    DatasetBuildResult,
    DatasetCounts,
    DatasetMetrics,
)


def build_artifact_root(data_dir: Path, dataset_id: str, version: str) -> Path:
    return data_dir / "datasets" / dataset_id / version


def write_dataset_artifacts(
    *,
    artifact_root: Path,
    full_rows: list[CanonicalRow],
    train_rows: list[CanonicalRow],
    test_rows: list[CanonicalRow],
    has_output: bool,
    dropped_empty_output_rows: int,
    dropped_empty_output_prompt_groups: int,
    build_inputs: BuildInputs,
    build_config: DatasetBuildConfig,
) -> DatasetBuildResult:
    artifact_root.mkdir(parents=True, exist_ok=True)

    full_path = artifact_root / "full.jsonl"
    train_path = artifact_root / "train.jsonl"
    test_path = artifact_root / "test.jsonl"
    manifest_path = artifact_root / "manifest.json"

    _write_jsonl(full_path, full_rows)
    _write_jsonl(train_path, train_rows)
    _write_jsonl(test_path, test_rows)

    counts = DatasetCounts(
        total_rows=len(full_rows),
        train_rows=len(train_rows),
        test_rows=len(test_rows),
        total_prompt_groups=len({row.prompt_id for row in full_rows}),
        train_prompt_groups=len({row.prompt_id for row in train_rows}),
        test_prompt_groups=len({row.prompt_id for row in test_rows}),
    )
    metrics = _build_dataset_metrics(full_rows)
    manifest_payload = build_manifest(
        artifact_root=artifact_root,
        build_inputs=build_inputs,
        build_config=build_config,
        counts=counts,
        metrics=metrics,
        has_output=has_output,
        dropped_empty_output_rows=dropped_empty_output_rows,
        dropped_empty_output_prompt_groups=dropped_empty_output_prompt_groups,
    )
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return DatasetBuildResult(
        artifact_root=artifact_root,
        full_path=full_path,
        train_path=train_path,
        test_path=test_path,
        manifest_path=manifest_path,
        counts=counts,
        metrics=metrics,
    )


def build_manifest(
    *,
    artifact_root: Path,
    build_inputs: BuildInputs,
    build_config: DatasetBuildConfig,
    counts: DatasetCounts,
    metrics: DatasetMetrics,
    has_output: bool,
    dropped_empty_output_rows: int,
    dropped_empty_output_prompt_groups: int,
) -> dict[str, Any]:
    config_payload: dict[str, Any] = {
        "dataset_id": build_config.dataset_id,
        "version": build_config.version,
        "score_mode": build_config.score_mode,
        "split_strategy": build_config.split.strategy,
        "split_parameters": {
            "test_fraction": build_config.split.test_fraction,
            "seed": build_config.split.seed,
        },
        "source_normalization": {
            "llm_config_path": str(build_config.llm_config_path),
        },
        "source_capabilities": {
            "has_output": has_output,
        },
    }
    if build_config.split.occupational_tags is not None:
        config_payload["split_parameters"]["occupational_tags"] = list(
            build_config.split.occupational_tags
        )
    sampling_payload: dict[str, int] = {}
    if build_config.max_model_groups is not None:
        sampling_payload["max_model_groups"] = build_config.max_model_groups
    if build_config.max_prompt_groups is not None:
        sampling_payload["max_prompt_groups"] = build_config.max_prompt_groups
    sampling_payload["max_evaluation_order"] = build_config.max_evaluation_order
    if sampling_payload:
        config_payload["sampling"] = sampling_payload

    return {
        "inputs": {
            "source_adapter": build_inputs.source_adapter,
            "source_location": build_inputs.source_location,
            "source_config_summary": build_inputs.source_config_summary,
        },
        "outputs": {
            "artifact_root": str(artifact_root),
            "full": "full.jsonl",
            "train": "train.jsonl",
            "test": "test.jsonl",
            "manifest": "manifest.json",
        },
        "config": config_payload,
        "counts": {
            "total_rows": counts.total_rows,
            "train_rows": counts.train_rows,
            "test_rows": counts.test_rows,
            "total_prompt_groups": counts.total_prompt_groups,
            "train_prompt_groups": counts.train_prompt_groups,
            "test_prompt_groups": counts.test_prompt_groups,
        },
        "metrics": {
            "average_models_per_prompt": metrics.average_models_per_prompt,
        },
        "row_filtering": {
            "empty_output": {
                "dropped_rows": dropped_empty_output_rows,
                "dropped_prompt_groups": dropped_empty_output_prompt_groups,
            }
        },
    }


def validate_artifact_contract(result: DatasetBuildResult) -> None:
    required_paths = (
        result.full_path,
        result.train_path,
        result.test_path,
        result.manifest_path,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise ValueError(f"Dataset artifact root is missing required files: {missing}")

    for jsonl_path in (result.full_path, result.train_path, result.test_path):
        _validate_jsonl_rows(jsonl_path)

    _validate_unique_row_ids(result.full_path)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    required_sections = {
        "inputs",
        "outputs",
        "config",
        "counts",
        "metrics",
        "row_filtering",
    }
    missing_sections = sorted(required_sections - set(manifest))
    if missing_sections:
        raise ValueError(
            f"manifest.json is missing required sections: {missing_sections}"
        )

    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("manifest.json metrics must be an object.")
    average_models_per_prompt = metrics.get("average_models_per_prompt")
    if isinstance(average_models_per_prompt, bool) or not isinstance(
        average_models_per_prompt, int | float
    ):
        raise ValueError(
            "manifest.json metrics.average_models_per_prompt must be numeric."
        )

    source_capabilities = manifest.get("config", {}).get("source_capabilities")
    if source_capabilities is not None:
        if not isinstance(source_capabilities, dict):
            raise ValueError(
                "manifest.json config.source_capabilities must be an object when present."
            )
        has_output = source_capabilities.get("has_output")
        if has_output is not None and not isinstance(has_output, bool):
            raise ValueError(
                "manifest.json config.source_capabilities.has_output must be a boolean when present."
            )

    row_filtering = manifest.get("row_filtering")
    if not isinstance(row_filtering, dict):
        raise ValueError("manifest.json row_filtering must be an object.")
    empty_output = row_filtering.get("empty_output")
    if not isinstance(empty_output, dict):
        raise ValueError("manifest.json row_filtering.empty_output must be an object.")
    for field_name in ("dropped_rows", "dropped_prompt_groups"):
        value = empty_output.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"manifest.json row_filtering.empty_output.{field_name} must be an integer."
            )


def _write_jsonl(path: Path, rows: list[CanonicalRow]) -> None:
    content = "\n".join(json.dumps(row.to_dict(), sort_keys=True) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _build_dataset_metrics(rows: list[CanonicalRow]) -> DatasetMetrics:
    prompt_model_names: dict[str, set[str]] = {}
    for row in rows:
        model_names = prompt_model_names.setdefault(row.prompt_id, set())
        raw_model_name = row.metadata.get("raw_model_name")
        if isinstance(raw_model_name, str):
            model_names.add(raw_model_name)

    if not prompt_model_names:
        return DatasetMetrics(average_models_per_prompt=0.0)

    distinct_model_counts = [
        len(model_names) for model_names in prompt_model_names.values()
    ]
    return DatasetMetrics(
        average_models_per_prompt=sum(distinct_model_counts)
        / len(distinct_model_counts)
    )


def _validate_jsonl_rows(path: Path) -> None:
    allowed_fields = {
        "id",
        "prompt_id",
        "input",
        "output",
        "score",
        "input_token",
        "output_tokken",
        "metadata",
    }
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        extra_fields = sorted(set(payload) - allowed_fields)
        if extra_fields:
            raise ValueError(
                f"{path} line {line_number} contains non-contract top-level fields: {extra_fields}"
            )
        if set(payload) != allowed_fields:
            missing_fields = sorted(allowed_fields - set(payload))
            raise ValueError(
                f"{path} line {line_number} is missing contract fields: {missing_fields}"
            )
        if not isinstance(payload.get("metadata"), dict):
            raise ValueError(f"{path} line {line_number} metadata must be an object.")


def _validate_unique_row_ids(path: Path) -> None:
    seen_ids: set[str] = set()
    duplicates: set[str] = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row_id = json.loads(line)["id"]
        if row_id in seen_ids:
            duplicates.add(row_id)
        seen_ids.add(row_id)

    if duplicates:
        raise ValueError(
            f"full.jsonl contains duplicate canonical row ids: {sorted(duplicates)}"
        )

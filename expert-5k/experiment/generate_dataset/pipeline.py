from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from .adapters import (
    ArenaExpert5KAdapter,
    RouterBenchAdapter,
    RouterBenchJsonlAdapter,
    SourceAdapter,
)
from .artifacts import (
    build_artifact_root,
    validate_artifact_contract,
    write_dataset_artifacts,
)
from .contracts import (
    BuildInputs,
    CanonicalRow,
    DatasetBuildConfig,
    DatasetBuildResult,
    SplitConfig,
)
from .settings import Settings, get_settings
from .splitting import split_rows, validate_split_disjointness

SUPPORTED_SCORE_MODES = {"pairwise_binary", "observed_mean_score"}


def run_dataset_pipeline(
    *,
    source_adapter: str,
    dataset_id: str,
    version: str,
    llm_config_path: Path,
    test_fraction: float,
    seed: int,
    max_model_groups: int | None = None,
    max_prompt_groups: int | None = None,
    max_evaluation_order: int = 1,
    source_split: str = "train",
    source_config_summary: dict[str, object] | None = None,
    score_mode: str = "pairwise_binary",
    settings: Settings | None = None,
) -> DatasetBuildResult:
    effective_score_mode = _normalize_score_mode(source_adapter, score_mode)
    if effective_score_mode not in SUPPORTED_SCORE_MODES:
        raise ValueError(
            f"Unsupported score_mode '{score_mode}'. Supported modes: {sorted(SUPPORTED_SCORE_MODES)}"
        )
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be greater than 0 and less than 1.")
    if max_evaluation_order <= 0:
        raise ValueError("max_evaluation_order must be a positive integer.")

    runtime_settings = settings or get_settings()
    effective_source_split = _normalize_source_split(source_adapter, source_split)
    adapter = _build_adapter(
        source_adapter,
        llm_config_path,
        effective_source_split,
        source_config_summary,
    )
    source_rows, pre_split_train_rows, pre_split_test_rows = _load_source_rows(
        source_adapter=source_adapter,
        adapter=adapter,
        llm_config_path=llm_config_path,
        source_config_summary=source_config_summary,
    )
    has_output = _infer_has_output(source_rows)
    filtered_output_rows = _filter_rows_by_output_capability(
        source_rows, has_output=has_output
    )
    evaluation_filtered_rows = _filter_rows_by_max_evaluation_order(
        filtered_output_rows,
        max_evaluation_order=max_evaluation_order,
    )
    filtered_rows = _sample_model_groups(
        evaluation_filtered_rows,
        max_model_groups=max_model_groups,
    )
    full_rows = _sample_prompt_groups(
        filtered_rows,
        max_prompt_groups=max_prompt_groups,
        seed=seed,
    )
    split_config = _build_split_config(
        test_fraction=test_fraction,
        seed=seed,
        source_config_summary=source_config_summary,
    )
    train_rows, test_rows = _resolve_output_splits(
        source_adapter=source_adapter,
        source_split=effective_source_split,
        full_rows=full_rows,
        split_config=split_config,
        pre_split_train_rows=pre_split_train_rows,
        pre_split_test_rows=pre_split_test_rows,
    )

    artifact_root = build_artifact_root(runtime_settings.data_dir, dataset_id, version)
    normalized_source_config_summary = _normalize_source_config_summary_for_manifest(
        source_config_summary
    )
    build_inputs = BuildInputs(
        source_adapter=source_adapter,
        source_location=_build_source_location(source_adapter, adapter),
        source_config_summary={
            **normalized_source_config_summary,
            "split": effective_source_split,
        },
    )
    build_config = DatasetBuildConfig(
        dataset_id=dataset_id,
        version=version,
        score_mode=effective_score_mode,
        split=split_config,
        output_root=artifact_root,
        llm_config_path=llm_config_path,
        max_model_groups=max_model_groups,
        max_prompt_groups=max_prompt_groups,
        max_evaluation_order=max_evaluation_order,
    )
    result = write_dataset_artifacts(
        artifact_root=artifact_root,
        full_rows=full_rows,
        train_rows=train_rows,
        test_rows=test_rows,
        has_output=has_output,
        dropped_empty_output_rows=len(source_rows) - len(filtered_output_rows),
        dropped_empty_output_prompt_groups=len({row.prompt_id for row in source_rows})
        - len({row.prompt_id for row in filtered_output_rows}),
        build_inputs=build_inputs,
        build_config=build_config,
    )
    validate_artifact_contract(result)
    return result


def _build_adapter(
    source_adapter: str,
    llm_config_path: Path,
    source_split: str,
    source_config_summary: dict[str, object] | None,
) -> SourceAdapter:
    if source_adapter == ArenaExpert5KAdapter.adapter_name:
        return ArenaExpert5KAdapter(split=source_split)
    if source_adapter == RouterBenchAdapter.adapter_name:
        return RouterBenchAdapter(split=source_split)
    if source_adapter == RouterBenchJsonlAdapter.adapter_name:
        return RouterBenchJsonlAdapter(
            split=source_split,
            train_jsonl_path=_read_source_config_path(
                source_config_summary,
                config_key=RouterBenchJsonlAdapter.train_jsonl_config_key,
            ),
            test_jsonl_path=_read_source_config_path(
                source_config_summary,
                config_key=RouterBenchJsonlAdapter.test_jsonl_config_key,
            ),
        )
    raise ValueError(f"Unsupported source adapter '{source_adapter}'.")


def _load_source_rows(
    *,
    source_adapter: str,
    adapter: SourceAdapter,
    llm_config_path: Path,
    source_config_summary: dict[str, object] | None,
) -> tuple[list[CanonicalRow], list[CanonicalRow] | None, list[CanonicalRow] | None]:
    if source_adapter != RouterBenchJsonlAdapter.adapter_name:
        rows = adapter.load_rows()
        return rows, None, None

    train_adapter = _build_adapter(
        source_adapter,
        llm_config_path,
        "train",
        source_config_summary,
    )
    test_adapter = _build_adapter(
        source_adapter,
        llm_config_path,
        "test",
        source_config_summary,
    )
    train_rows = train_adapter.load_rows()
    test_rows = test_adapter.load_rows()
    return [*train_rows, *test_rows], train_rows, test_rows


def _build_source_location(source_adapter: str, adapter: SourceAdapter) -> str:
    if source_adapter == ArenaExpert5KAdapter.adapter_name:
        return ArenaExpert5KAdapter.dataset_name
    if source_adapter == RouterBenchAdapter.adapter_name:
        return f"{RouterBenchAdapter.dataset_name}/{RouterBenchAdapter.raw_data_file}"
    if source_adapter == RouterBenchJsonlAdapter.adapter_name:
        source_location = getattr(adapter, "source_location", None)
        if isinstance(source_location, str) and source_location:
            return source_location
        return RouterBenchJsonlAdapter.train_dataset_name
    return source_adapter


def _normalize_source_split(source_adapter: str, source_split: str) -> str:
    if source_adapter == RouterBenchAdapter.adapter_name:
        if source_split in {"train", "raw", RouterBenchAdapter.raw_data_file}:
            return "raw"
        raise ValueError(
            "router-bench only supports the raw row-wise source. Use source_split='raw'."
        )

    if source_adapter == RouterBenchJsonlAdapter.adapter_name:
        if source_split in {"train", "test"}:
            return source_split
        raise ValueError(
            "router-bench-jsonl only supports explicit train/test JSONL inputs. Use source_split='train' or source_split='test'."
        )

    return source_split


def _normalize_score_mode(source_adapter: str, score_mode: str) -> str:
    if (
        source_adapter
        in {RouterBenchAdapter.adapter_name, RouterBenchJsonlAdapter.adapter_name}
        and score_mode == "pairwise_binary"
    ):
        return "observed_mean_score"
    return score_mode


def _read_source_config_path(
    source_config_summary: dict[str, object] | None, *, config_key: str
) -> str | Path | None:
    if not source_config_summary:
        return None
    value = source_config_summary.get(config_key)
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if not isinstance(value, str):
        raise ValueError(
            f"source_config_summary.{config_key} must be a string path when provided."
        )
    return value


def _normalize_source_config_summary_for_manifest(
    source_config_summary: dict[str, object] | None,
) -> dict[str, object]:
    if not source_config_summary:
        return {}
    normalized: dict[str, object] = {}
    for key, value in source_config_summary.items():
        normalized[key] = str(value) if isinstance(value, Path) else value
    return normalized


def _infer_has_output(rows: list[CanonicalRow]) -> bool:
    if not rows:
        return True
    return any(row.output.strip() for row in rows)


def _build_split_config(
    *,
    test_fraction: float,
    seed: int,
    source_config_summary: dict[str, object] | None,
) -> SplitConfig:
    occupational_tags = _normalize_occupational_tags(source_config_summary)
    if occupational_tags is None:
        return SplitConfig(test_fraction=test_fraction, seed=seed)
    return SplitConfig(
        strategy="occupational_tag_ood",
        test_fraction=test_fraction,
        seed=seed,
        occupational_tags=occupational_tags,
    )


def _resolve_output_splits(
    *,
    source_adapter: str,
    source_split: str,
    full_rows: list[CanonicalRow],
    split_config: SplitConfig,
    pre_split_train_rows: list[CanonicalRow] | None,
    pre_split_test_rows: list[CanonicalRow] | None,
) -> tuple[list[CanonicalRow], list[CanonicalRow]]:
    if source_adapter == RouterBenchJsonlAdapter.adapter_name:
        return _project_pre_split_rows(
            full_rows,
            train_rows=pre_split_train_rows or [],
            test_rows=pre_split_test_rows or [],
        )

    train_rows, test_rows = split_rows(full_rows, split_config)
    validate_split_disjointness(train_rows, test_rows)
    return train_rows, test_rows


def _project_pre_split_rows(
    full_rows: list[CanonicalRow],
    *,
    train_rows: list[CanonicalRow],
    test_rows: list[CanonicalRow],
) -> tuple[list[CanonicalRow], list[CanonicalRow]]:
    train_keys = {_canonical_row_membership_key(row) for row in train_rows}
    test_keys = {_canonical_row_membership_key(row) for row in test_rows}
    projected_train_rows = [
        row for row in full_rows if _canonical_row_membership_key(row) in train_keys
    ]
    projected_test_rows = [
        row for row in full_rows if _canonical_row_membership_key(row) in test_keys
    ]
    return projected_train_rows, projected_test_rows


def _canonical_row_membership_key(
    row: CanonicalRow,
) -> tuple[str, str | None, str | None]:
    source_split = row.metadata.get("source_split")
    source_dataset = row.metadata.get("source_dataset")
    return (
        row.id,
        source_split if isinstance(source_split, str) else None,
        source_dataset if isinstance(source_dataset, str) else None,
    )


def _normalize_occupational_tags(
    source_config_summary: dict[str, object] | None,
) -> tuple[str, ...] | None:
    if not source_config_summary or "occupational_tags" not in source_config_summary:
        return None

    raw_tags = source_config_summary["occupational_tags"]
    if not isinstance(raw_tags, list) or not raw_tags:
        raise ValueError(
            "source_config_summary.occupational_tags must be a non-empty JSON array of strings."
        )

    normalized_tags: list[str] = []
    seen_tags: set[str] = set()
    for raw_tag in raw_tags:
        if not isinstance(raw_tag, str) or not raw_tag.strip():
            raise ValueError(
                "source_config_summary.occupational_tags must be a non-empty JSON array of strings."
            )
        tag = raw_tag.strip()
        if tag not in seen_tags:
            normalized_tags.append(tag)
            seen_tags.add(tag)
    return tuple(normalized_tags)


def _filter_rows_with_non_empty_output(rows: list[CanonicalRow]) -> list[CanonicalRow]:
    return [row for row in rows if row.output.strip()]


def _filter_rows_by_output_capability(
    rows: list[CanonicalRow], *, has_output: bool
) -> list[CanonicalRow]:
    if not has_output:
        return rows
    return _filter_rows_with_non_empty_output(rows)


def _filter_rows_by_max_evaluation_order(
    rows: list[CanonicalRow], *, max_evaluation_order: int
) -> list[CanonicalRow]:
    filtered_rows: list[CanonicalRow] = []
    for row in rows:
        evaluation_order = row.metadata.get("evaluation_order")
        if not isinstance(evaluation_order, int) or isinstance(evaluation_order, bool):
            filtered_rows.append(row)
            continue
        if evaluation_order <= max_evaluation_order:
            filtered_rows.append(row)
    return filtered_rows


def _sample_model_groups(
    rows: list[CanonicalRow], *, max_model_groups: int | None
) -> list[CanonicalRow]:
    if max_model_groups is None:
        return rows
    if max_model_groups <= 0:
        raise ValueError("max_model_groups must be a positive integer when provided.")

    rows_by_model: dict[str, list[CanonicalRow]] = defaultdict(list)
    for row in rows:
        rows_by_model[_extract_raw_model_name(row)].append(row)

    if max_model_groups >= len(rows_by_model):
        return rows

    selected_model_names = {
        model_name
        for model_name, _ in sorted(
            rows_by_model.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )[:max_model_groups]
    }
    return [row for row in rows if _extract_raw_model_name(row) in selected_model_names]


def _sample_prompt_groups(
    rows: list[CanonicalRow], *, max_prompt_groups: int | None, seed: int
) -> list[CanonicalRow]:
    if max_prompt_groups is None:
        return rows
    if max_prompt_groups <= 0:
        raise ValueError("max_prompt_groups must be a positive integer when provided.")

    grouped_rows: dict[str, list[CanonicalRow]] = defaultdict(list)
    for row in rows:
        grouped_rows[row.prompt_id].append(row)

    prompt_ids = sorted(grouped_rows)
    if max_prompt_groups >= len(prompt_ids):
        return rows

    rng = random.Random(seed)
    prompt_order = list(prompt_ids)
    rng.shuffle(prompt_order)
    prompt_order_index = {
        prompt_id: index for index, prompt_id in enumerate(prompt_order)
    }

    group_models = {
        prompt_id: _extract_group_models(group_rows)
        for prompt_id, group_rows in grouped_rows.items()
    }
    uncovered_models = set().union(*group_models.values())
    selected_prompt_ids: list[str] = []
    remaining_prompt_ids = set(prompt_ids)

    while uncovered_models and len(selected_prompt_ids) < max_prompt_groups:
        best_prompt_id = max(
            remaining_prompt_ids,
            key=lambda prompt_id: (
                len(group_models[prompt_id] & uncovered_models),
                -prompt_order_index[prompt_id],
            ),
        )
        selected_prompt_ids.append(best_prompt_id)
        remaining_prompt_ids.remove(best_prompt_id)
        uncovered_models -= group_models[best_prompt_id]

    if uncovered_models:
        fallback_prompt_ids = _find_coverage_preserving_prompt_groups(
            prompt_order=prompt_order,
            group_models=group_models,
            max_prompt_groups=max_prompt_groups,
        )
        if fallback_prompt_ids is None:
            raise ValueError(
                "max_prompt_groups is too small to preserve model coverage after filtering."
            )
        selected_prompt_ids = fallback_prompt_ids
        remaining_prompt_ids = set(prompt_ids) - set(selected_prompt_ids)

    for prompt_id in prompt_order:
        if len(selected_prompt_ids) >= max_prompt_groups:
            break
        if prompt_id in remaining_prompt_ids:
            selected_prompt_ids.append(prompt_id)

    selected_prompt_id_set = set(selected_prompt_ids)
    return [row for row in rows if row.prompt_id in selected_prompt_id_set]


def _extract_group_models(rows: list[CanonicalRow]) -> set[str]:
    models: set[str] = set()
    for row in rows:
        models.add(_extract_model_name(row))
    return models


def _extract_model_name(row: CanonicalRow) -> str:
    model_name = row.metadata.get("raw_model_name")
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()
    raise ValueError("Canonical rows must contain a non-empty metadata.raw_model_name.")


def _extract_raw_model_name(row: CanonicalRow) -> str:
    model_name = row.metadata.get("raw_model_name")
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()
    raise ValueError("Canonical rows must contain a non-empty metadata.raw_model_name.")


def _find_coverage_preserving_prompt_groups(
    *,
    prompt_order: list[str],
    group_models: dict[str, set[str]],
    max_prompt_groups: int,
) -> list[str] | None:
    required_models = set().union(*group_models.values())
    max_group_count = min(max_prompt_groups, len(prompt_order))

    for group_count in range(1, max_group_count + 1):
        for candidate_prompt_ids in combinations(prompt_order, group_count):
            covered_models = set().union(
                *(group_models[prompt_id] for prompt_id in candidate_prompt_ids)
            )
            if covered_models == required_models:
                return list(candidate_prompt_ids)
    return None

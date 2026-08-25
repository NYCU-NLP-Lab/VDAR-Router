from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def validate_neighbors_rows(
    *, router_key: str, rows: list[dict[str, Any]], validate_neighbor_row: Any
) -> None:
    seen_row_keys: set[tuple[str, str, str, int]] = set()
    for line_number, row in enumerate(rows, start=1):
        validate_neighbor_row(router_key=router_key, row=row, line_number=line_number)
        row_key = (
            str(row["prompt_id"]),
            str(row["test_item_id"]),
            str(row["retrieved_item_id"]),
            int(row["rank"]),
        )
        if row_key in seen_row_keys:
            raise ValueError(
                "neighbors.jsonl contains duplicate "
                f"(prompt_id, test_item_id, retrieved_item_id, rank) rows for router '{router_key}'"
            )
        seen_row_keys.add(row_key)


def validate_alignment_like_row(
    *,
    row: dict[str, Any],
    canonical_row_by_id: dict[str, Any],
    expected_router_key: str,
    context: str,
    train_item_ids: set[str],
    test_item_ids: set[str],
    numeric_tolerance: float,
) -> None:
    if row.get("router_key") != expected_router_key:
        raise ValueError(f"{context} router_key must match '{expected_router_key}'.")
    test_item_id = row.get("test_item_id")
    retrieved_item_id = row.get("retrieved_item_id")
    if (
        test_item_id not in canonical_row_by_id
        or retrieved_item_id not in canonical_row_by_id
    ):
        raise ValueError(f"{context} uses ids outside the Stage 1 canonical dataset.")
    if test_item_id not in test_item_ids:
        raise ValueError(
            f"{context} test_item_id must come from the Stage 1 test split."
        )
    if retrieved_item_id not in train_item_ids:
        raise ValueError(
            f"{context} retrieved_item_id must come from the Stage 1 train split."
        )
    b_test = row.get("b_test")
    b_retrieved = row.get("b_retrieved")
    abs_delta_b = row.get("abs_delta_b")
    delta_b = row.get("delta_b")
    if b_test is None or b_retrieved is None:
        if abs_delta_b is not None:
            raise ValueError(
                f"{context} must set abs_delta_b to null when difficulties are missing."
            )
        return
    expected_abs_delta = abs(float(b_test) - float(b_retrieved))
    expected_delta = float(b_retrieved) - float(b_test)
    if abs_delta_b is None or not math.isfinite(float(abs_delta_b)):
        raise ValueError(f"{context} must provide finite abs_delta_b.")
    if abs(float(abs_delta_b) - expected_abs_delta) > numeric_tolerance:
        raise ValueError(
            f"{context} abs_delta_b does not match |b_test - b_retrieved|."
        )
    if delta_b is not None and abs(float(delta_b) - expected_delta) > numeric_tolerance:
        raise ValueError(f"{context} delta_b does not match b_retrieved - b_test.")


def validate_irt_alignment_outputs(
    *,
    analysis_schema_version: str,
    evaluation_root: Path,
    analysis_root: Path,
    stage3_router_keys: set[str],
    canonical_row_by_id: dict[str, Any],
    train_item_ids: set[str],
    test_item_ids: set[str],
    numeric_tolerance: float,
    measurement_families: set[str],
    comparative_plot_paths: list[Path],
    query_embedding_rows_present: bool,
    read_jsonl_objects: Any,
) -> None:
    manifest_path = analysis_root / "manifest.json"
    summary_path = analysis_root / "summary.json"
    difficulty_path = analysis_root / "rasch_difficulties.jsonl"
    reward_neighbors_path = analysis_root / "reward_irt_neighbors.jsonl"
    random_neighbors_path = analysis_root / "random_neighbors.jsonl"
    per_query_path = analysis_root / "reward_irt_per_query.jsonl"
    comparative_summary_path = analysis_root / "reward_irt_summary.json"
    required_paths = [
        analysis_root,
        manifest_path,
        summary_path,
        difficulty_path,
        reward_neighbors_path,
        random_neighbors_path,
        per_query_path,
        comparative_summary_path,
        *comparative_plot_paths,
    ]
    if query_embedding_rows_present:
        required_paths.append(analysis_root / "query_embedding_knn_neighbors.jsonl")
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise ValueError(
            f"IRT alignment artifact root is missing required files: {missing_paths}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_manifest_keys = {
        "schema_version",
        "stage1_full_path",
        "stage3_artifact_root",
        "difficulty_artifact_path",
        "difficulty_source",
        "measurement_family",
        "routers",
        "outputs",
    }
    missing_manifest_keys = sorted(required_manifest_keys - set(manifest))
    if missing_manifest_keys:
        raise ValueError(
            f"IRT alignment manifest is missing required keys: {missing_manifest_keys}"
        )
    if manifest["schema_version"] != analysis_schema_version:
        raise ValueError(
            f"IRT alignment manifest schema_version must be '{analysis_schema_version}'."
        )
    if manifest["difficulty_source"] not in {"loaded", "fit"}:
        raise ValueError("IRT alignment difficulty_source must be 'loaded' or 'fit'.")
    if manifest["measurement_family"] not in measurement_families:
        raise ValueError(
            f"IRT alignment manifest measurement_family must be one of {sorted(measurement_families)}."
        )
    if Path(manifest["stage3_artifact_root"]).resolve() != evaluation_root:
        raise ValueError(
            "IRT alignment manifest stage3_artifact_root must match the evaluation root."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    routers = summary.get("routers")
    if not isinstance(routers, list):
        raise ValueError("IRT alignment summary must contain a routers list.")

    for router_summary in routers:
        if not isinstance(router_summary, dict):
            raise ValueError("IRT alignment summary routers entries must be objects.")
        router_key = router_summary.get("router_key")
        if not isinstance(router_key, str) or not router_key.strip():
            raise ValueError(
                "IRT alignment summary router_key must be a non-empty string."
            )
        if router_key not in stage3_router_keys:
            raise ValueError(
                f"IRT alignment summary router_key '{router_key}' is not present in the Stage 3 run."
            )
        alignment_rows_path = (
            analysis_root / "routers" / router_key / "alignment_rows.jsonl"
        )
        metrics_path = analysis_root / "routers" / router_key / "metrics.json"
        if not alignment_rows_path.exists() or not metrics_path.exists():
            raise ValueError(
                f"IRT alignment router '{router_key}' is missing alignment_rows.jsonl or metrics.json."
            )
        for row in read_jsonl_objects(alignment_rows_path):
            validate_alignment_like_row(
                row=row,
                canonical_row_by_id=canonical_row_by_id,
                expected_router_key=router_key,
                context=f"router '{router_key}' alignment_rows.jsonl",
                train_item_ids=train_item_ids,
                test_item_ids=test_item_ids,
                numeric_tolerance=numeric_tolerance,
            )

    for path, context in (
        (reward_neighbors_path, "reward_irt_neighbors.jsonl"),
        (random_neighbors_path, "random_neighbors.jsonl"),
    ):
        for row in read_jsonl_objects(path):
            validate_alignment_like_row(
                row=row,
                canonical_row_by_id=canonical_row_by_id,
                expected_router_key=str(row.get("router_key")),
                context=context,
                train_item_ids=train_item_ids,
                test_item_ids=test_item_ids,
                numeric_tolerance=numeric_tolerance,
            )

    if query_embedding_rows_present:
        for row in read_jsonl_objects(
            analysis_root / "query_embedding_knn_neighbors.jsonl"
        ):
            validate_alignment_like_row(
                row=row,
                canonical_row_by_id=canonical_row_by_id,
                expected_router_key=str(row.get("router_key")),
                context="query_embedding_knn_neighbors.jsonl",
                train_item_ids=train_item_ids,
                test_item_ids=test_item_ids,
                numeric_tolerance=numeric_tolerance,
            )

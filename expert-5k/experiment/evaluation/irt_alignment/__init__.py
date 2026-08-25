from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..pipeline import _read_canonical_rows, load_stage1_dataset_manifest
from .compare import (
    build_alignment_rows,
    build_comparative_summary,
    build_missing_reason,
    build_near_rate_thresholds,
    build_per_query_summary,
    build_query_embedding_knn_rows,
    build_random_baseline_rows,
    build_reward_neighbor_rows,
    build_router_metrics,
    build_split_prompt_difficulties,
    load_query_embedding_artifact,
    write_comparative_plots,
    write_router_comparison_ecdf_plot,
    write_router_ecdf_plot,
)
from .paths import ensure_safe_router_key as _ensure_safe_router_key
from .paths import resolve_under_root as _resolve_under_root
from .rasch import DifficultySurface, resolve_difficulty_surface
from .validation import (
    validate_irt_alignment_outputs as _validate_irt_alignment_outputs_strict,
)
from .validation import (
    validate_neighbors_rows as _validate_neighbors_rows_strict,
)

ANALYSIS_SCHEMA_VERSION = "stage3_1.irt_alignment.v1"
ANALYSIS_DIRNAME = "analysis"
IRT_ALIGNMENT_DIRNAME = "irt_alignment"
NUMERIC_TOLERANCE = 1e-9
DEFAULT_RANDOM_REPEATS = 5
DEFAULT_RANDOM_SEED = 42
MEASUREMENT_FAMILIES = {"bayesian_rasch", "rasch_score_fallback", "mirt"}
EXCLUDED_ROUTER_FAMILIES = {"icl_router"}


@dataclass(slots=True)
class IRTAlignmentRunResult:
    evaluation_root: Path
    analysis_root: Path | None
    manifest_path: Path | None
    summary_path: Path | None
    analyzed_router_keys: list[str]
    difficulty_source: str | None
    measurement_family: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.evaluation.irt_alignment",
        description="Run Stage 3-1 retrieval IRT alignment over a completed evaluation root.",
    )
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--difficulty-artifact-path", type=Path)
    parser.add_argument("--query-embedding-artifact-path", type=Path)
    parser.add_argument(
        "--measurement-family",
        choices=sorted(MEASUREMENT_FAMILIES),
        default="bayesian_rasch",
    )
    parser.add_argument("--random-repeats", type=int, default=DEFAULT_RANDOM_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_irt_alignment_analysis(
        evaluation_root=args.evaluation_root,
        difficulty_artifact_path=args.difficulty_artifact_path,
        query_embedding_artifact_path=args.query_embedding_artifact_path,
        measurement_family=args.measurement_family,
        random_repeats=args.random_repeats,
        seed=args.seed,
        plots_enabled=not args.no_plots,
    )
    if result.analysis_root is None:
        print("No neighbors.jsonl artifacts found; no Stage 3-1 outputs were written.")
        return 0
    print(result.analysis_root)
    return 0


def run_irt_alignment_analysis(
    *,
    evaluation_root: Path,
    difficulty_artifact_path: Path | None = None,
    query_embedding_artifact_path: Path | None = None,
    measurement_family: str = "bayesian_rasch",
    random_repeats: int = DEFAULT_RANDOM_REPEATS,
    seed: int = DEFAULT_RANDOM_SEED,
    plots_enabled: bool = True,
) -> IRTAlignmentRunResult:
    if measurement_family not in MEASUREMENT_FAMILIES:
        raise ValueError(
            f"measurement_family must be one of {sorted(MEASUREMENT_FAMILIES)}."
        )
    if random_repeats <= 0:
        raise ValueError("random_repeats must be a positive integer.")

    evaluation_root = evaluation_root.resolve()
    evaluation_manifest = _load_evaluation_manifest(evaluation_root / "manifest.json")
    stage1_manifest = load_stage1_dataset_manifest(
        Path(evaluation_manifest["inputs"]["dataset_manifest_path"])
    )
    canonical_rows = _read_canonical_rows(stage1_manifest.full_path)
    train_rows = _read_canonical_rows(stage1_manifest.train_path)
    test_rows = _read_canonical_rows(stage1_manifest.test_path)
    train_item_id_set = {row.id for row in train_rows}
    test_item_id_set = {row.id for row in test_rows}
    canonical_row_by_id = {row.id: row for row in canonical_rows}
    if len(canonical_row_by_id) != len(canonical_rows):
        raise ValueError("Stage 1 canonical row ids must be unique for IRT alignment.")

    analyzed_router_keys, stage3_router_keys = _extract_stage3_router_keys(
        evaluation_manifest
    )
    analysis_root = evaluation_root / ANALYSIS_DIRNAME / IRT_ALIGNMENT_DIRNAME
    router_neighbors = _collect_router_neighbors(
        evaluation_root=evaluation_root,
        stage3_router_keys=analyzed_router_keys,
    )
    if not router_neighbors:
        if analysis_root.exists():
            shutil.rmtree(analysis_root)
        return IRTAlignmentRunResult(
            evaluation_root=evaluation_root,
            analysis_root=None,
            manifest_path=None,
            summary_path=None,
            analyzed_router_keys=[],
            difficulty_source=None,
            measurement_family=None,
        )

    difficulty_surface = resolve_difficulty_surface(
        difficulty_artifact_path=difficulty_artifact_path,
        measurement_family=measurement_family,
        canonical_rows=canonical_rows,
        train_rows=train_rows,
        test_rows=test_rows,
        canonical_row_by_id=canonical_row_by_id,
        seed=seed,
    )
    embedding_lookup: dict[str, list[float]] | None = None
    resolved_embedding_path: Path | None = None
    if query_embedding_artifact_path is not None:
        resolved_embedding_path = query_embedding_artifact_path.resolve()
        embedding_lookup = load_query_embedding_artifact(
            resolved_embedding_path,
            canonical_row_by_id=canonical_row_by_id,
            read_jsonl_objects=_read_jsonl_objects,
        )

    if analysis_root.exists():
        shutil.rmtree(analysis_root)
    (analysis_root / "routers").mkdir(parents=True, exist_ok=True)

    _write_jsonl(analysis_root / "rasch_difficulties.jsonl", difficulty_surface.rows)

    router_summaries: list[dict[str, Any]] = []
    manifest_router_entries: list[dict[str, Any]] = []
    reward_neighbors_rows: list[dict[str, Any]] = []
    reward_rows_by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for router_key in sorted(router_neighbors):
        neighbors_path, neighbor_rows = router_neighbors[router_key]
        router_root = _resolve_under_root(analysis_root / "routers", router_key)
        router_root.mkdir(parents=True, exist_ok=True)
        alignment_rows = build_alignment_rows(
            router_key=router_key,
            neighbor_rows=neighbor_rows,
            canonical_row_by_id=canonical_row_by_id,
            train_item_ids=train_item_id_set,
            test_item_ids=test_item_id_set,
            difficulty_lookup=difficulty_surface.lookup,
            difficulty_source=difficulty_surface.source,
            validate_neighbor_row=_validate_neighbor_row,
            build_missing_reason=build_missing_reason,
        )
        metrics = build_router_metrics(
            router_key=router_key,
            alignment_rows=alignment_rows,
        )
        _write_jsonl(router_root / "alignment_rows.jsonl", alignment_rows)
        _write_json(router_root / "metrics.json", metrics)
        ecdf_path = None
        if metrics["matched_row_count"] > 0 and plots_enabled:
            ecdf_path = router_root / "ecdf_abs_delta_b.png"
            write_router_ecdf_plot(
                ecdf_path, router_key=router_key, alignment_rows=alignment_rows
            )

        top_level_reward_rows = build_reward_neighbor_rows(alignment_rows)
        reward_neighbors_rows.extend(top_level_reward_rows)
        for row in top_level_reward_rows:
            reward_rows_by_group.setdefault(
                (str(row["router_key"]), str(row["test_item_id"])), []
            ).append(row)

        router_summary = {
            "router_key": router_key,
            "neighbors_path": str(neighbors_path),
            "alignment_rows_path": str(router_root / "alignment_rows.jsonl"),
            "metrics_path": str(router_root / "metrics.json"),
            "row_count": metrics["row_count"],
            "matched_row_count": metrics["matched_row_count"],
            "missing_row_count": metrics["missing_row_count"],
        }
        if ecdf_path is not None:
            router_summary["ecdf_abs_delta_b_path"] = str(ecdf_path)
        router_summaries.append(router_summary)
        manifest_router_entries.append(router_summary)

    train_item_ids = [row.id for row in train_rows]
    random_generator = random.Random(seed)
    query_embedding_rows: list[dict[str, Any]] = []
    random_neighbors_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    train_prompt_values = build_split_prompt_difficulties(
        rows=train_rows,
        difficulty_lookup=difficulty_surface.lookup,
    )
    test_prompt_values = build_split_prompt_difficulties(
        rows=test_rows,
        difficulty_lookup=difficulty_surface.lookup,
    )
    near_rate_thresholds = build_near_rate_thresholds(train_prompt_values)

    for group_key in sorted(reward_rows_by_group):
        router_key, test_item_id = group_key
        reward_rows = sorted(
            reward_rows_by_group[group_key], key=lambda row: int(row["rank"])
        )
        top_k = len(reward_rows)
        query_rows: list[dict[str, Any]] = []
        if embedding_lookup is not None:
            query_rows = build_query_embedding_knn_rows(
                router_key=router_key,
                test_item_id=test_item_id,
                reward_rows=reward_rows,
                train_item_ids=train_item_ids,
                canonical_row_by_id=canonical_row_by_id,
                difficulty_lookup=difficulty_surface.lookup,
                embedding_lookup=embedding_lookup,
                top_k=top_k,
            )
            query_embedding_rows.extend(query_rows)

        random_rows = build_random_baseline_rows(
            router_key=router_key,
            test_item_id=test_item_id,
            reward_rows=reward_rows,
            train_item_ids=train_item_ids,
            canonical_row_by_id=canonical_row_by_id,
            difficulty_lookup=difficulty_surface.lookup,
            top_k=top_k,
            random_repeats=random_repeats,
            rng=random_generator,
        )
        random_neighbors_rows.extend(random_rows)
        per_query_rows.append(
            build_per_query_summary(
                reward_rows=reward_rows,
                query_embedding_rows=query_rows,
                random_rows=random_rows,
                thresholds=near_rate_thresholds,
            )
        )

    comparative_summary = build_comparative_summary(
        reward_rows=reward_neighbors_rows,
        query_embedding_rows=query_embedding_rows,
        random_rows=random_neighbors_rows,
        per_query_rows=per_query_rows,
        train_prompt_values=train_prompt_values,
        test_prompt_values=test_prompt_values,
        measurement_family=difficulty_surface.measurement_family,
        difficulty_source=difficulty_surface.source,
        random_repeats=random_repeats,
        query_embedding_artifact_path=resolved_embedding_path,
        thresholds=near_rate_thresholds,
    )

    _write_jsonl(analysis_root / "reward_irt_neighbors.jsonl", reward_neighbors_rows)
    if query_embedding_rows:
        _write_jsonl(
            analysis_root / "query_embedding_knn_neighbors.jsonl", query_embedding_rows
        )
    _write_jsonl(analysis_root / "random_neighbors.jsonl", random_neighbors_rows)
    _write_jsonl(analysis_root / "reward_irt_per_query.jsonl", per_query_rows)
    _write_json(analysis_root / "reward_irt_summary.json", comparative_summary)

    comparative_plot_paths: list[Path] = []
    if plots_enabled and write_comparative_plots(
        analysis_root=analysis_root,
        reward_rows=reward_neighbors_rows,
        query_embedding_rows=query_embedding_rows,
        random_rows=random_neighbors_rows,
        measurement_family=difficulty_surface.measurement_family,
    ):
        comparative_plot_paths = [
            analysis_root / "abs_delta_b_ecdf.png",
            analysis_root / "abs_delta_b_hist.png",
        ]
    router_comparison_plot_path: Path | None = None
    if plots_enabled:
        candidate_path = analysis_root / "router_comparison_ecdf.png"
        if write_router_comparison_ecdf_plot(
            candidate_path,
            reward_rows=reward_neighbors_rows,
            measurement_family=difficulty_surface.measurement_family,
        ):
            router_comparison_plot_path = candidate_path

    summary_payload = {
        "routers": router_summaries,
        "measurement_family": difficulty_surface.measurement_family,
        "difficulty_source": difficulty_surface.source,
    }
    summary_path = analysis_root / "summary.json"
    _write_json(summary_path, summary_payload)
    manifest_path = analysis_root / "manifest.json"
    outputs = {
        "artifact_root": str(analysis_root),
        "manifest": "manifest.json",
        "summary": "summary.json",
        "rasch_difficulties": "rasch_difficulties.jsonl",
        "reward_neighbors": "reward_irt_neighbors.jsonl",
        "random_neighbors": "random_neighbors.jsonl",
        "per_query": "reward_irt_per_query.jsonl",
        "comparative_summary": "reward_irt_summary.json",
        "routers": "routers",
    }
    if query_embedding_rows:
        outputs["query_embedding_knn_neighbors"] = "query_embedding_knn_neighbors.jsonl"
    if comparative_plot_paths:
        outputs["comparative_ecdf"] = "abs_delta_b_ecdf.png"
        outputs["comparative_hist"] = "abs_delta_b_hist.png"
    if router_comparison_plot_path is not None:
        outputs["router_comparison_ecdf"] = "router_comparison_ecdf.png"
    _write_json(
        manifest_path,
        {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "stage1_full_path": str(stage1_manifest.full_path),
            "stage3_artifact_root": str(evaluation_root),
            "difficulty_artifact_path": difficulty_surface.manifest_path,
            "difficulty_source": difficulty_surface.source,
            "measurement_family": difficulty_surface.measurement_family,
            "routers": manifest_router_entries,
            "outputs": outputs,
        },
    )

    _validate_irt_alignment_outputs_strict(
        analysis_schema_version=ANALYSIS_SCHEMA_VERSION,
        evaluation_root=evaluation_root,
        analysis_root=analysis_root,
        stage3_router_keys=stage3_router_keys,
        canonical_row_by_id=canonical_row_by_id,
        train_item_ids=train_item_id_set,
        test_item_ids=test_item_id_set,
        numeric_tolerance=NUMERIC_TOLERANCE,
        measurement_families=MEASUREMENT_FAMILIES,
        comparative_plot_paths=comparative_plot_paths,
        query_embedding_rows_present=bool(query_embedding_rows),
        read_jsonl_objects=_read_jsonl_objects,
    )
    return IRTAlignmentRunResult(
        evaluation_root=evaluation_root,
        analysis_root=analysis_root,
        manifest_path=manifest_path,
        summary_path=summary_path,
        analyzed_router_keys=sorted(router_neighbors),
        difficulty_source=difficulty_surface.source,
        measurement_family=difficulty_surface.measurement_family,
    )


def _load_evaluation_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Stage 3 evaluation manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Stage 3 evaluation manifest must be a JSON object.")
    inputs = payload.get("inputs")
    routers = payload.get("routers")
    if not isinstance(inputs, dict):
        raise ValueError("Stage 3 evaluation manifest inputs must be an object.")
    if not isinstance(routers, list):
        raise ValueError("Stage 3 evaluation manifest routers must be an array.")
    dataset_manifest_path = inputs.get("dataset_manifest_path")
    if not isinstance(dataset_manifest_path, str) or not dataset_manifest_path.strip():
        raise ValueError(
            "Stage 3 evaluation manifest inputs.dataset_manifest_path must be a non-empty string."
        )
    return payload


def _extract_stage3_router_keys(
    evaluation_manifest: dict[str, Any],
) -> tuple[set[str], set[str]]:
    analyzed_router_keys: set[str] = set()
    stage3_router_keys: set[str] = set()
    for index, raw_router in enumerate(evaluation_manifest.get("routers", [])):
        if not isinstance(raw_router, dict):
            raise ValueError(f"Stage 3 manifest routers[{index}] must be an object.")
        router_key = raw_router.get("router_key")
        router_family = raw_router.get("router_family")
        if not isinstance(router_key, str) or not router_key.strip():
            raise ValueError(
                f"Stage 3 manifest routers[{index}].router_key must be a non-empty string."
            )
        safe_router_key = _ensure_safe_router_key(router_key)
        stage3_router_keys.add(safe_router_key)
        if isinstance(router_family, str) and router_family in EXCLUDED_ROUTER_FAMILIES:
            continue
        analyzed_router_keys.add(safe_router_key)
    return analyzed_router_keys, stage3_router_keys


def _collect_router_neighbors(
    *,
    evaluation_root: Path,
    stage3_router_keys: set[str],
) -> dict[str, tuple[Path, list[dict[str, Any]]]]:
    routers_root = evaluation_root / "routers"
    if not routers_root.exists():
        raise ValueError(f"Stage 3 routers directory does not exist: {routers_root}")
    router_neighbors: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    for router_key in sorted(stage3_router_keys):
        neighbors_path = _resolve_under_root(
            routers_root, router_key, "neighbors.jsonl"
        )
        if not neighbors_path.exists():
            continue
        rows = _read_jsonl_objects(neighbors_path)
        if rows:
            _validate_neighbors_rows_strict(
                router_key=router_key,
                rows=rows,
                validate_neighbor_row=_validate_neighbor_row,
            )
            router_neighbors[router_key] = (neighbors_path, rows)
    return router_neighbors


def _validate_neighbor_row(
    *, router_key: str, row: dict[str, Any], line_number: int
) -> None:
    for field_name in ("prompt_id", "test_item_id", "retrieved_item_id", "router_key"):
        value = row.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"neighbors.jsonl field '{field_name}' must be a non-empty string at line {line_number}."
            )
    if row.get("router_key") != router_key:
        raise ValueError(
            f"neighbors.jsonl router_key must match router directory '{router_key}' at line {line_number}."
        )
    rank = row.get("rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError(
            f"neighbors.jsonl rank must be a positive integer at line {line_number}."
        )
    source_split = row.get("source_split")
    if not isinstance(source_split, str) or not source_split.strip():
        raise ValueError(
            f"neighbors.jsonl source_split must be a non-empty string at line {line_number}."
        )
    if "retrieval_score" not in row:
        raise ValueError(
            f"neighbors.jsonl retrieval_score must be present at line {line_number}."
        )
    retrieval_score = row.get("retrieval_score")
    if retrieval_score is not None and (
        isinstance(retrieval_score, bool)
        or not isinstance(retrieval_score, int | float)
        or not math.isfinite(float(retrieval_score))
    ):
        raise ValueError(
            f"neighbors.jsonl retrieval_score must be finite numeric or null at line {line_number}."
        )


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL rows must be objects in {path} line {line_number}")
        rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

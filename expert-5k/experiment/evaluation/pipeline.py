from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from rich import box
from rich.table import Table
from tqdm import tqdm

from cache import CacheDeferredRequest
from cache._constants import ENDPOINT_DIRECTORY_NAMES
from cache._storage import CacheStorage
from difficulty_aware_router.agents.settings import get_analysis_settings
from experiment.generate_dataset.model_mapping import (
    ModelPricing,
    load_llm_config,
    normalize_model_name,
)
from experiment.training.contracts import TrainingManifest
from experiment.training.pipeline import (
    load_router_from_manifest,
    load_training_manifest,
)

from .artifacts import (
    build_artifact_root,
    finalize_evaluation_artifacts,
    write_evaluation_summary,
    write_router_evaluation_artifacts,
)
from .contracts import (
    BaselineProfile,
    CanonicalRow,
    EvaluationManifest,
    EvaluationRunResult,
    RouterEvaluationCounts,
    RouterEvaluationMetrics,
    RouterEvaluationSummary,
    RouterEvaluationTarget,
    RouterTargetInput,
    Stage1DatasetManifest,
    _is_secret_like_key,
)
from .settings import Settings, get_settings

EVALUATION_SCHEMA_VERSION = "stage3.evaluation.v6"
PRIMARY_METRICS = [
    "pairwise_accuracy",
    "average_train_elo_spearman",
    "average_full_elo_spearman",
    "average_prompt_cost",
    "total_prompt_cost",
    "total_actual_prompt_cost",
]
CACHE_AWARE_ROUTER_FAMILIES = {
    "difficulty-aware-router",
    "difficulty-aware-query-embedding",
    "difficulty-aware-no-reward-ranking",
    "icl_router",
}
STAGE3_SANDBOXED_CACHE_ROUTER_FAMILIES = {
    "difficulty-aware-router",
    "difficulty-aware-query-embedding",
    "difficulty-aware-no-reward-ranking",
}
CACHE_SCOPE_ROUTER_PRIVATE = "router-private"
CACHE_SCOPE_SHARED_NAMESPACE = "shared-namespace"
STAGE3_RUNTIME_CACHE_ROOT = Path(__file__).resolve().parents[2] / ".cache" / "runtime"
COST_NORMALIZATION_SCALE_KEY = "cost_normalization_scale"


@dataclass(slots=True)
class PromptAggregate:
    prompt_id: str
    query: str
    model_scores: dict[str, float]
    test_item_ids_by_model: dict[str, list[str]]
    model_outputs: dict[str, list[str]] | None
    model_prices: dict[str, float | None]
    fallback_priced_model_names: set[str]


def load_stage1_dataset_manifest(dataset_manifest_path: Path) -> Stage1DatasetManifest:
    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    required_sections = {"inputs", "outputs", "config", "counts"}
    missing_sections = sorted(required_sections - set(manifest))
    if missing_sections:
        raise ValueError(
            f"Stage 1 manifest is missing required sections: {missing_sections}"
        )

    outputs = manifest["outputs"]
    required_outputs = {"artifact_root", "full", "train", "test", "manifest"}
    missing_outputs = sorted(required_outputs - set(outputs))
    if missing_outputs:
        raise ValueError(
            f"Stage 1 manifest outputs are missing required keys: {missing_outputs}"
        )

    source_normalization = manifest["config"].get("source_normalization")
    if not isinstance(source_normalization, dict):
        raise ValueError(
            "Stage 1 manifest config.source_normalization must be an object."
        )
    raw_llm_config_path = source_normalization.get("llm_config_path")
    if not isinstance(raw_llm_config_path, str) or not raw_llm_config_path.strip():
        raise ValueError(
            "Stage 1 manifest config.source_normalization.llm_config_path must be a non-empty string."
        )
    llm_config_path = _resolve_stage1_llm_config_path(raw_llm_config_path)
    has_output = _read_stage1_has_output(manifest)

    artifact_root = dataset_manifest_path.parent
    stage1_manifest = Stage1DatasetManifest(
        manifest_path=dataset_manifest_path,
        artifact_root=artifact_root,
        dataset_id=manifest["config"]["dataset_id"],
        dataset_version=manifest["config"]["version"],
        score_mode=manifest["config"]["score_mode"],
        has_output=has_output,
        llm_config_path=llm_config_path,
        full_path=artifact_root / outputs["full"],
        train_path=artifact_root / outputs["train"],
        test_path=artifact_root / outputs["test"],
    )
    required_paths = [
        stage1_manifest.full_path,
        stage1_manifest.train_path,
        stage1_manifest.test_path,
        stage1_manifest.manifest_path,
        stage1_manifest.llm_config_path,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise ValueError(
            f"Stage 1 dataset artifact is missing required files: {missing_paths}"
        )
    return stage1_manifest


def run_evaluation_pipeline(
    *,
    dataset_manifest_path: Path,
    router_targets: Sequence[RouterTargetInput | dict[str, Any]],
    run_label: str,
    fail_on_router_setup_error: bool = False,
    report_enabled: bool = True,
    settings: Settings | None = None,
) -> EvaluationRunResult:
    normalized_router_targets = _normalize_router_targets(router_targets)
    if not normalized_router_targets:
        raise ValueError("At least one router target is required.")

    dataset_manifest = load_stage1_dataset_manifest(dataset_manifest_path)
    llm_config = load_llm_config(dataset_manifest.llm_config_path)
    model_name_mapping = llm_config.model_name_mapping
    full_rows = _read_canonical_rows(dataset_manifest.full_path)
    train_rows = _read_canonical_rows(dataset_manifest.train_path)
    test_rows = _read_canonical_rows(dataset_manifest.test_path)

    train_baseline = BaselineProfile(
        name="train_global_ranking",
        source_split="train",
        semantic_role="predictive_baseline",
        allowed_metrics=[
            "pairwise_accuracy",
            "average_train_elo_spearman",
            "average_prompt_cost",
            "total_prompt_cost",
            "total_actual_prompt_cost",
        ],
        model_scores=_build_global_profile(train_rows),
    )
    full_reference_baseline = BaselineProfile(
        name="full_global_ranking",
        source_split="full",
        semantic_role="descriptive_reference",
        allowed_metrics=["average_full_elo_spearman"],
        model_scores=_build_global_profile(full_rows),
    )
    prompt_aggregates = _build_prompt_aggregates(
        test_rows,
        model_name_mapping,
        llm_config.model_pricing,
        llm_config.pricing_fallback_model_name,
        include_model_outputs=report_enabled and dataset_manifest.has_output,
    )
    avg_max_cost_per_query = _compute_avg_max_cost_per_query(
        train_rows,
        model_name_mapping,
        llm_config.model_pricing,
        llm_config.pricing_fallback_model_name,
    )
    artifact_root = build_artifact_root(
        (settings or get_settings()).data_dir,
        dataset_manifest.dataset_id,
        run_label,
    )

    evaluated_router_targets: list[RouterEvaluationTarget] = []
    router_summaries: list[RouterEvaluationSummary] = []
    seen_router_keys: set[str] = set()
    for target_request in normalized_router_targets:
        manifest_path = target_request.router_manifest_path
        try:
            training_manifest = load_training_manifest(manifest_path)
        except Exception as exc:  # noqa: BLE001
            if fail_on_router_setup_error:
                raise
            router_target = _build_fallback_router_target(
                manifest_path, target_suffix=target_request.target_suffix
            )
            if router_target.router_key in seen_router_keys:
                raise ValueError(
                    f"Duplicate router evaluation key: {router_target.router_key}"
                )
            seen_router_keys.add(router_target.router_key)
            evaluated_router_targets.append(router_target)
            payload = _build_router_failure_payload(
                failure_stage="manifest_load",
                router_key=router_target.router_key,
                router_family=router_target.router_family,
                router_variant=router_target.router_variant,
                manifest_path=manifest_path,
                target_suffix=router_target.target_suffix,
                shared_cache_namespace=target_request.shared_cache_namespace,
                prompt_aggregates=prompt_aggregates,
                runtime_config=target_request.runtime_config,
                exc=exc,
            )
            router_summaries.append(
                write_router_evaluation_artifacts(
                    artifact_root=artifact_root,
                    payload=payload,
                    report_enabled=report_enabled,
                )
            )
            write_evaluation_summary(
                artifact_root=artifact_root,
                router_summaries=router_summaries,
            )
            continue

        if training_manifest.dataset.manifest_path != dataset_manifest.manifest_path:
            raise ValueError(
                "Stage 2 manifest dataset manifest path does not match Stage 1 dataset."
            )

        base_router_key = _build_router_key(
            training_manifest.router_family,
            training_manifest.router_variant,
        )
        router_key = _build_target_router_key(
            base_router_key, target_request.target_suffix
        )
        if router_key in seen_router_keys:
            raise ValueError(f"Duplicate router evaluation key: {router_key}")
        seen_router_keys.add(router_key)

        runtime_config = (
            dict(target_request.runtime_config)
            if target_request.runtime_config is not None
            else None
        )
        shared_cache_namespace = target_request.shared_cache_namespace

        evaluated_router_targets.append(
            RouterEvaluationTarget(
                router_key=router_key,
                router_family=training_manifest.router_family,
                router_variant=training_manifest.router_variant,
                manifest_path=manifest_path,
                target_suffix=target_request.target_suffix,
            )
        )
        payload = _evaluate_router(
            router_key=router_key,
            target_suffix=target_request.target_suffix,
            training_manifest_path=manifest_path,
            training_manifest=training_manifest,
            runtime_config=_resolve_router_runtime_config(
                artifact_root=artifact_root,
                router_artifact_root=training_manifest.artifact_path,
                router_key=router_key,
                shared_cache_namespace=shared_cache_namespace,
                router_family=training_manifest.router_family,
                base_runtime_config=runtime_config,
                cost_normalization_scale=avg_max_cost_per_query,
            ),
            shared_cache_namespace=shared_cache_namespace,
            prompt_aggregates=prompt_aggregates,
            llm_config_path=dataset_manifest.llm_config_path,
            fallback_pricing_model_name=llm_config.pricing_fallback_model_name,
            model_name_mapping=model_name_mapping,
            full_reference_baseline=full_reference_baseline,
            fail_on_router_setup_error=fail_on_router_setup_error,
            report_enabled=report_enabled,
            train_baseline=train_baseline,
        )
        router_summaries.append(
            write_router_evaluation_artifacts(
                artifact_root=artifact_root,
                payload=payload,
                report_enabled=report_enabled,
            )
        )
        write_evaluation_summary(
            artifact_root=artifact_root,
            router_summaries=router_summaries,
        )

    manifest = EvaluationManifest(
        schema_version=EVALUATION_SCHEMA_VERSION,
        dataset_manifest_path=dataset_manifest.manifest_path,
        artifact_root=artifact_root,
        dataset_id=dataset_manifest.dataset_id,
        dataset_version=dataset_manifest.dataset_version,
        baselines=[train_baseline, full_reference_baseline],
        routers=evaluated_router_targets,
        router_targets=normalized_router_targets,
    )
    return finalize_evaluation_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        router_summaries=router_summaries,
    )


def render_summary_table(router_summaries: list[RouterEvaluationSummary]) -> Table:
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        show_edge=False,
        pad_edge=False,
        collapse_padding=True,
    )
    table.add_column("router", no_wrap=True)
    table.add_column("ok", justify="right", no_wrap=True)
    table.add_column("fail", justify="right", no_wrap=True)
    table.add_column("pairwise_acc", justify="right", no_wrap=True)
    table.add_column("train_sp", justify="right", no_wrap=True)
    table.add_column("full_sp", justify="right", no_wrap=True)
    table.add_column("avg_cost", justify="right", no_wrap=True)
    table.add_column("total_cost", justify="right", no_wrap=True)
    table.add_column("oracle_cost", justify="right", no_wrap=True)

    for summary in router_summaries:
        table.add_row(
            summary.router_key,
            str(summary.counts.succeeded_prompts),
            str(summary.counts.failed_prompts),
            _format_metric(summary.metrics.pairwise_accuracy),
            _format_metric(summary.metrics.average_train_elo_spearman),
            _format_metric(summary.metrics.average_full_elo_spearman),
            _format_metric(summary.metrics.average_prompt_cost),
            _format_metric(summary.metrics.total_prompt_cost),
            _format_metric(summary.metrics.total_actual_prompt_cost),
        )
    return table


def _evaluate_router(
    *,
    router_key: str,
    target_suffix: str | None,
    shared_cache_namespace: str | None,
    training_manifest_path: Path,
    training_manifest: TrainingManifest,
    runtime_config: dict[str, Any] | None,
    prompt_aggregates: list[PromptAggregate],
    llm_config_path: Path,
    fallback_pricing_model_name: str | None,
    model_name_mapping: dict[str, str],
    full_reference_baseline: BaselineProfile,
    fail_on_router_setup_error: bool,
    report_enabled: bool,
    train_baseline: BaselineProfile,
) -> dict[str, Any]:
    try:
        router = load_router_from_manifest(
            training_manifest, runtime_config=runtime_config
        )
    except Exception as exc:  # noqa: BLE001
        if fail_on_router_setup_error:
            raise
        return _build_router_failure_payload(
            failure_stage="router_setup",
            router_key=router_key,
            router_family=training_manifest.router_family,
            router_variant=training_manifest.router_variant,
            manifest_path=training_manifest_path,
            target_suffix=target_suffix,
            shared_cache_namespace=shared_cache_namespace,
            prompt_aggregates=prompt_aggregates,
            runtime_config=runtime_config,
            exc=exc,
        )

    prompt_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    neighbor_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []
    succeeded_prompt_count = 0
    failed_prompt_count = 0
    scored_pair_count = 0
    priced_prompt_count = 0
    pairwise_total = 0.0
    pairwise_count = 0
    train_spearman_total = 0.0
    train_spearman_count = 0
    full_spearman_total = 0.0
    full_spearman_count = 0
    prompt_cost_total = 0.0
    actual_prompt_cost_total = 0.0
    actual_priced_prompt_count = 0
    unmapped_raw_model_names: set[str] = set()
    missing_priced_model_names: set[str] = set()
    fallback_priced_model_names: set[str] = set()
    fallback_priced_prompt_count = 0
    skipped_prompt_count = 0

    for prompt in tqdm(
        prompt_aggregates,
        total=len(prompt_aggregates),
        desc=f"Evaluating {router_key}",
        unit="prompt",
        disable=None,
    ):
        try:
            routed = router.route_single_ranked({"query": prompt.query})
            if not routed.ranked_models:
                raise ValueError("Router returned an empty ranked_models list.")

            predicted_scores = {
                model.model_name: float(model.score) for model in routed.ranked_models
            }
            predicted_order = [model.model_name for model in routed.ranked_models]
            selected_model_name = predicted_order[0]
            (
                prompt_cost,
                unmapped_raw_model_name,
                missing_priced_model_name,
                fallback_priced_model_name,
            ) = _resolve_prompt_cost(prompt, selected_model_name, model_name_mapping)
            if unmapped_raw_model_name is not None:
                unmapped_raw_model_names.add(unmapped_raw_model_name)
            if missing_priced_model_name is not None:
                missing_priced_model_names.add(missing_priced_model_name)
            if fallback_priced_model_name is not None:
                fallback_priced_model_names.add(fallback_priced_model_name)
                fallback_priced_prompt_count += 1
            (
                actual_prompt_cost,
                unmapped_actual_raw_model_name,
                missing_actual_priced_model_name,
                actual_fallback_priced_model_name,
            ) = _resolve_actual_prompt_cost(prompt, model_name_mapping)
            if unmapped_actual_raw_model_name is not None:
                unmapped_raw_model_names.add(unmapped_actual_raw_model_name)
            if missing_actual_priced_model_name is not None:
                missing_priced_model_names.add(missing_actual_priced_model_name)
            if actual_fallback_priced_model_name is not None:
                fallback_priced_model_names.add(actual_fallback_priced_model_name)
                fallback_priced_prompt_count += 1

            pair_items, comparable_pair_count, pairwise_accuracy = _score_prompt_pairs(
                prompt=prompt,
                predicted_scores=predicted_scores,
                predicted_order=predicted_order,
                include_records=report_enabled,
            )

            train_elo_spearman = _compute_prompt_spearman(
                predicted_scores=predicted_scores,
                predicted_order=predicted_order,
                reference_scores=train_baseline.model_scores,
            )
            full_elo_spearman = _compute_prompt_spearman(
                predicted_scores=predicted_scores,
                predicted_order=predicted_order,
                reference_scores=full_reference_baseline.model_scores,
            )
            scored_pair_count += comparable_pair_count
            if pairwise_accuracy is not None:
                pairwise_total += pairwise_accuracy
                pairwise_count += 1
            if train_elo_spearman is not None:
                train_spearman_total += train_elo_spearman
                train_spearman_count += 1
            if full_elo_spearman is not None:
                full_spearman_total += full_elo_spearman
                full_spearman_count += 1
            if prompt_cost is not None:
                prompt_cost_total += prompt_cost
                priced_prompt_count += 1
            if actual_prompt_cost is not None:
                actual_prompt_cost_total += actual_prompt_cost
                actual_priced_prompt_count += 1
            if report_enabled:
                prompt_record = {
                    "prompt_id": prompt.prompt_id,
                    "query": prompt.query,
                    "selected_model_name": selected_model_name,
                    "ranked_models": [asdict(model) for model in routed.ranked_models],
                    "ground_truth_scores": dict(prompt.model_scores),
                    "ground_truth_model_outputs": {
                        model_name: list(outputs)
                        for model_name, outputs in sorted(
                            (prompt.model_outputs or {}).items()
                        )
                    },
                    "metric_values": {
                        "pairwise_accuracy": pairwise_accuracy,
                        "train_elo_spearman": train_elo_spearman,
                        "full_elo_spearman": full_elo_spearman,
                        "prompt_cost": prompt_cost,
                        "actual_prompt_cost": actual_prompt_cost,
                    },
                    "baseline_sources": {
                        "pairwise_accuracy": train_baseline.source_split,
                        "average_train_elo_spearman": train_baseline.source_split,
                        "prompt_cost": train_baseline.source_split,
                        "actual_prompt_cost": train_baseline.source_split,
                        "average_full_elo_spearman": full_reference_baseline.source_split,
                    },
                    "router_metadata": routed.metadata,
                }
                prompt_records.append(prompt_record)
                pair_records.extend(pair_items)
            neighbor_records.extend(
                _build_neighbor_records(
                    router_key=router_key,
                    target_suffix=target_suffix,
                    prompt=prompt,
                    selected_model_name=selected_model_name,
                    router_metadata=routed.metadata,
                )
            )
            succeeded_prompt_count += 1
        except CacheDeferredRequest:
            skipped_prompt_count += 1
            continue
        except Exception as exc:  # noqa: BLE001
            failure_records.append(
                {
                    "failure_stage": "prompt_route",
                    "prompt_id": prompt.prompt_id,
                    "query": prompt.query,
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
            )
            failed_prompt_count += 1

    counts = RouterEvaluationCounts(
        total_prompts=len(prompt_aggregates),
        succeeded_prompts=succeeded_prompt_count,
        failed_prompts=failed_prompt_count,
        setup_failures=0,
        skipped_prompts=skipped_prompt_count,
        scored_pairs=scored_pair_count,
        priced_prompts=priced_prompt_count,
    )
    cache_metadata = _build_router_cache_metadata(
        router_family=training_manifest.router_family,
        router_key=router_key,
        shared_cache_namespace=shared_cache_namespace,
        runtime_config=runtime_config,
    )
    summary = RouterEvaluationSummary(
        router_key=router_key,
        router_family=training_manifest.router_family,
        router_variant=training_manifest.router_variant,
        manifest_path=training_manifest_path,
        target_suffix=target_suffix,
        artifact_root=Path("."),
        counts=counts,
        metrics=RouterEvaluationMetrics(
            pairwise_accuracy=_mean_from_total(pairwise_total, pairwise_count),
            average_train_elo_spearman=_mean_from_total(
                train_spearman_total, train_spearman_count
            ),
            average_full_elo_spearman=_mean_from_total(
                full_spearman_total, full_spearman_count
            ),
            average_prompt_cost=_mean_from_total(
                prompt_cost_total, priced_prompt_count
            ),
            total_prompt_cost=float(prompt_cost_total) if priced_prompt_count else None,
            total_actual_prompt_cost=(
                float(actual_prompt_cost_total) if actual_priced_prompt_count else None
            ),
        ),
        evaluation_status=_derive_evaluation_status(
            counts=counts,
            cache_metadata=cache_metadata,
        ),
        runtime_config_effective=dict(runtime_config or {}),
        cache=cache_metadata,
        warnings=(
            {
                "unmapped_raw_model_names": sorted(unmapped_raw_model_names),
                "missing_priced_model_names": sorted(missing_priced_model_names),
                "fallback_priced_model_names": sorted(fallback_priced_model_names),
                "fallback_priced_prompt_count": fallback_priced_prompt_count,
                "fallback_pricing_model_name": fallback_pricing_model_name,
                "llm_config_path": str(llm_config_path),
            }
            if (
                unmapped_raw_model_names
                or missing_priced_model_names
                or fallback_priced_model_names
            )
            else None
        ),
    )
    return {
        "summary": summary,
        "prompt_records": prompt_records,
        "pair_records": pair_records,
        "neighbor_records": neighbor_records,
        "failure_records": failure_records,
    }


def _build_router_failure_payload(
    *,
    failure_stage: str,
    router_key: str,
    router_family: str,
    router_variant: str | None,
    manifest_path: Path,
    target_suffix: str | None,
    shared_cache_namespace: str | None,
    prompt_aggregates: list[PromptAggregate],
    runtime_config: dict[str, Any] | None,
    exc: Exception,
) -> dict[str, Any]:
    failure_records = [
        {
            "failure_stage": failure_stage,
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
            "affected_prompt_count": len(prompt_aggregates),
        }
    ]
    summary = RouterEvaluationSummary(
        router_key=router_key,
        router_family=router_family,
        router_variant=router_variant,
        manifest_path=manifest_path,
        target_suffix=target_suffix,
        artifact_root=Path("."),
        counts=RouterEvaluationCounts(
            total_prompts=len(prompt_aggregates),
            succeeded_prompts=0,
            failed_prompts=0,
            setup_failures=1,
            skipped_prompts=len(prompt_aggregates),
            scored_pairs=0,
            priced_prompts=0,
        ),
        metrics=RouterEvaluationMetrics(
            pairwise_accuracy=None,
            average_train_elo_spearman=None,
            average_full_elo_spearman=None,
            average_prompt_cost=None,
            total_prompt_cost=None,
            total_actual_prompt_cost=None,
        ),
        evaluation_status="router_setup_failed",
        runtime_config_effective=dict(runtime_config or {}),
        cache=_build_router_cache_metadata(
            router_family=router_family,
            router_key=router_key,
            shared_cache_namespace=shared_cache_namespace,
            runtime_config=runtime_config,
        ),
        warnings=None,
    )
    return {
        "summary": summary,
        "prompt_records": [],
        "pair_records": [],
        "neighbor_records": [],
        "failure_records": failure_records,
    }


def _build_global_profile(rows: list[CanonicalRow]) -> dict[str, float]:
    totals: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        model_name = row.metadata.get("raw_model_name")
        if isinstance(model_name, str) and model_name.strip():
            totals[model_name.strip()].append(float(row.score))
    return {
        model_name: float(sum(scores) / len(scores))
        for model_name, scores in sorted(totals.items())
        if scores
    }


def _build_prompt_aggregates(
    rows: list[CanonicalRow],
    model_name_mapping: dict[str, str],
    model_pricing: dict[str, ModelPricing],
    pricing_fallback_model_name: str | None,
    *,
    include_model_outputs: bool = True,
) -> list[PromptAggregate]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row.prompt_id.strip():
            raise ValueError(
                "Canonical Stage 1 rows must contain a non-empty prompt_id."
            )
        if not row.input.strip():
            raise ValueError(
                "Canonical Stage 1 rows must contain a non-empty input field."
            )
        model_name = row.metadata.get("raw_model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "Canonical Stage 1 evaluation rows must contain metadata.raw_model_name."
            )
        raw_model_name = model_name.strip()
        bucket = grouped.setdefault(
            row.prompt_id,
            {
                "query": row.input,
                "scores": defaultdict(list),
                "test_item_ids_by_model": defaultdict(list),
                "prices": defaultdict(list),
                "fallback_priced_model_names": set(),
            },
        )
        if bucket["query"] != row.input:
            raise ValueError(
                f"All rows for prompt_id '{row.prompt_id}' must share the same input text."
            )
        bucket["scores"][raw_model_name].append(float(row.score))
        test_item_ids_by_model = bucket["test_item_ids_by_model"]
        if row.id not in test_item_ids_by_model[raw_model_name]:
            test_item_ids_by_model[raw_model_name].append(row.id)
        if include_model_outputs:
            outputs_by_model = bucket.setdefault("outputs", defaultdict(list))
            normalized_output = row.output.strip()
            if (
                normalized_output
                and normalized_output not in outputs_by_model[raw_model_name]
            ):
                outputs_by_model[raw_model_name].append(normalized_output)
        computed_cost, used_fallback = _compute_row_cost(
            row,
            model_name_mapping,
            model_pricing,
            pricing_fallback_model_name,
        )
        if computed_cost is not None:
            bucket["prices"][raw_model_name].append(computed_cost)
        if used_fallback:
            bucket["fallback_priced_model_names"].add(raw_model_name)

    prompt_aggregates: list[PromptAggregate] = []
    for prompt_id in sorted(grouped):
        bucket = grouped[prompt_id]
        prompt_aggregates.append(
            PromptAggregate(
                prompt_id=prompt_id,
                query=str(bucket["query"]),
                model_scores={
                    model_name: float(sum(values) / len(values))
                    for model_name, values in sorted(bucket["scores"].items())
                },
                test_item_ids_by_model={
                    model_name: list(item_ids)
                    for model_name, item_ids in sorted(
                        bucket["test_item_ids_by_model"].items()
                    )
                },
                model_outputs=(
                    {
                        model_name: list(outputs)
                        for model_name, outputs in sorted(bucket["outputs"].items())
                    }
                    if include_model_outputs
                    else None
                ),
                model_prices={
                    model_name: _mean(values)
                    for model_name, values in sorted(bucket["prices"].items())
                },
                fallback_priced_model_names=set(bucket["fallback_priced_model_names"]),
            )
        )
    return prompt_aggregates


def _compute_avg_max_cost_per_query(
    rows: Sequence[CanonicalRow],
    model_name_mapping: dict[str, str],
    model_pricing: dict[str, ModelPricing],
    pricing_fallback_model_name: str | None,
) -> float:
    prompt_max_costs: dict[str, float] = {}
    for row in rows:
        computed_cost, _ = _compute_normalization_row_cost(
            row,
            model_name_mapping,
            model_pricing,
            pricing_fallback_model_name,
        )
        if computed_cost is None:
            continue
        current_max = prompt_max_costs.get(row.prompt_id)
        if current_max is None or computed_cost > current_max:
            prompt_max_costs[row.prompt_id] = computed_cost
    if not prompt_max_costs:
        return 0.0
    return float(sum(prompt_max_costs.values()) / len(prompt_max_costs))


def _read_stage1_has_output(manifest: dict[str, Any]) -> bool:
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


def _build_neighbor_records(
    *,
    router_key: str,
    target_suffix: str | None,
    prompt: PromptAggregate,
    selected_model_name: str,
    router_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    retrieved_evidence = router_metadata.get("retrieved_evidence")
    if not isinstance(retrieved_evidence, dict):
        return []

    neighbor_records: list[dict[str, Any]] = []
    for candidate_model_name, raw_evidence_rows in sorted(retrieved_evidence.items()):
        if (
            not isinstance(candidate_model_name, str)
            or not candidate_model_name.strip()
        ):
            continue
        if not isinstance(raw_evidence_rows, list):
            continue

        test_item_ids = prompt.test_item_ids_by_model.get(candidate_model_name)
        if not test_item_ids or len(test_item_ids) != 1:
            continue
        test_item_id = test_item_ids[0]

        for raw_evidence_row in raw_evidence_rows:
            if not isinstance(raw_evidence_row, dict):
                continue
            retrieved_item_id = raw_evidence_row.get("row_id")
            rank = raw_evidence_row.get("rank")
            if (
                not isinstance(retrieved_item_id, str)
                or not retrieved_item_id.strip()
                or isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank <= 0
            ):
                continue

            neighbor_record: dict[str, Any] = {
                "prompt_id": prompt.prompt_id,
                "test_item_id": test_item_id,
                "retrieved_item_id": retrieved_item_id.strip(),
                "rank": rank,
                "router_key": router_key,
                "source_split": "train",
                "retrieval_score": None,
                "query_text": prompt.query,
                "selected_model_name": selected_model_name,
                "candidate_model_name": candidate_model_name,
            }
            if target_suffix is not None:
                neighbor_record["target_suffix"] = target_suffix

            distance = raw_evidence_row.get("distance")
            if not isinstance(distance, bool) and isinstance(distance, int | float):
                neighbor_record["distance"] = float(distance)

            similarity = raw_evidence_row.get("similarity")
            if not isinstance(similarity, bool) and isinstance(similarity, int | float):
                neighbor_record["similarity"] = float(similarity)

            retrieval_metadata = {
                str(key): value
                for key, value in raw_evidence_row.items()
                if key not in {"row_id", "rank", "distance", "similarity"}
                and not _is_secret_like_key(str(key))
            }
            if retrieval_metadata:
                neighbor_record["retrieval_metadata"] = retrieval_metadata

            neighbor_records.append(neighbor_record)

    return neighbor_records


def _read_canonical_rows(path: Path) -> list[CanonicalRow]:
    rows: list[CanonicalRow] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        try:
            rows.append(
                CanonicalRow(
                    id=str(payload["id"]),
                    prompt_id=str(payload["prompt_id"]),
                    input=str(payload["input"]),
                    output=str(payload["output"]),
                    score=float(payload["score"]),
                    input_token=int(payload["input_token"]),
                    output_tokken=int(payload["output_tokken"]),
                    metadata=dict(payload["metadata"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid canonical Stage 1 row in {path} line {line_number}."
            ) from exc
    return rows


def _score_prompt_pairs(
    *,
    prompt: PromptAggregate,
    predicted_scores: dict[str, float],
    predicted_order: list[str],
    include_records: bool = True,
) -> tuple[list[dict[str, Any]], int, float | None]:
    pair_items: list[dict[str, Any]] = []
    concordant = 0
    comparable_pairs = 0

    model_names = sorted(prompt.model_scores)
    predicted_position = {name: index for index, name in enumerate(predicted_order)}
    for left_index, left_name in enumerate(model_names):
        for right_name in model_names[left_index + 1 :]:
            left_truth = prompt.model_scores[left_name]
            right_truth = prompt.model_scores[right_name]
            if math.isclose(left_truth, right_truth, rel_tol=0.0, abs_tol=1e-12):
                continue

            comparable_pairs += 1
            truth_winner = left_name if left_truth > right_truth else right_name
            predicted_winner = _preferred_model(
                left_name,
                right_name,
                predicted_scores=predicted_scores,
                predicted_position=predicted_position,
            )
            correct = predicted_winner == truth_winner
            if correct:
                concordant += 1
            if include_records:
                pair_items.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "left_model_name": left_name,
                        "right_model_name": right_name,
                        "truth_winner_model_name": truth_winner,
                        "predicted_winner_model_name": predicted_winner,
                        "correct": correct,
                    }
                )

    if comparable_pairs == 0:
        return pair_items, comparable_pairs, None
    pairwise_accuracy = concordant / comparable_pairs
    return pair_items, comparable_pairs, pairwise_accuracy


def _resolve_actual_prompt_cost(
    prompt: PromptAggregate,
    model_name_mapping: dict[str, str],
) -> tuple[float | None, str | None, str | None, str | None]:
    selected_model_name = _recover_ground_truth_selected_model_name(prompt)
    if selected_model_name is None:
        return None, None, None, None
    cost = prompt.model_prices.get(selected_model_name)
    if cost is not None:
        fallback_priced_model_name = (
            selected_model_name
            if selected_model_name in prompt.fallback_priced_model_names
            else None
        )
        return cost, None, None, fallback_priced_model_name
    if selected_model_name in prompt.model_scores:
        if selected_model_name not in model_name_mapping:
            return None, selected_model_name, None, None
        return (
            None,
            None,
            normalize_model_name(selected_model_name, model_name_mapping),
            None,
        )
    return None, None, None, None


def _resolve_prompt_cost(
    prompt: PromptAggregate,
    model_name: str,
    model_name_mapping: dict[str, str],
) -> tuple[float | None, str | None, str | None, str | None]:
    cost = prompt.model_prices.get(model_name)
    if cost is not None:
        fallback_priced_model_name = (
            model_name if model_name in prompt.fallback_priced_model_names else None
        )
        return cost, None, None, fallback_priced_model_name
    if model_name in prompt.model_scores:
        if model_name not in model_name_mapping:
            return None, model_name, None, None
        return None, None, normalize_model_name(model_name, model_name_mapping), None
    if model_name not in model_name_mapping:
        return None, model_name, None, None
    return None, None, None, None


def _resolve_stage1_llm_config_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[2] / candidate


def _recover_ground_truth_selected_model_name(prompt: PromptAggregate) -> str | None:
    if not prompt.model_scores:
        return None

    winning_score = max(prompt.model_scores.values())
    winners = [
        model_name
        for model_name, score in prompt.model_scores.items()
        if math.isclose(score, winning_score, rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(winners) != 1:
        return None
    return winners[0]


def _compute_row_cost(
    row: CanonicalRow,
    model_name_mapping: dict[str, str],
    model_pricing: dict[str, ModelPricing],
    pricing_fallback_model_name: str | None,
) -> tuple[float | None, bool]:
    model_name = row.metadata.get("raw_model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        return None, False
    normalized_model_name = normalize_model_name(model_name.strip(), model_name_mapping)
    pricing = model_pricing.get(normalized_model_name)
    if pricing is None:
        if pricing_fallback_model_name is None:
            return None, False
        pricing = model_pricing.get(pricing_fallback_model_name)
        if pricing is None:
            return None, False
        return (
            float(row.input_token) * pricing.input_cost_per_token,
            True,
        )
    return (
        float(row.input_token) * pricing.input_cost_per_token,
        False,
    )


def _compute_normalization_row_cost(
    row: CanonicalRow,
    model_name_mapping: dict[str, str],
    model_pricing: dict[str, ModelPricing],
    pricing_fallback_model_name: str | None,
) -> tuple[float | None, bool]:
    model_name = row.metadata.get("raw_model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        return None, False
    normalized_model_name = normalize_model_name(model_name.strip(), model_name_mapping)
    pricing = model_pricing.get(normalized_model_name)
    if pricing is None:
        if pricing_fallback_model_name is None:
            return None, False
        pricing = model_pricing.get(pricing_fallback_model_name)
        if pricing is None:
            return None, False
        return (
            float(row.input_token) * pricing.input_cost_per_token
            + float(row.output_tokken) * pricing.output_cost_per_token,
            True,
        )
    return (
        float(row.input_token) * pricing.input_cost_per_token
        + float(row.output_tokken) * pricing.output_cost_per_token,
        False,
    )


def _compute_prompt_spearman(
    *,
    predicted_scores: dict[str, float],
    predicted_order: list[str],
    reference_scores: dict[str, float],
) -> float | None:
    model_names = sorted(set(reference_scores) | set(predicted_scores))
    if len(model_names) < 2:
        return None

    predicted_fallback = len(predicted_order) + len(model_names)
    enriched_predicted_scores = {
        name: predicted_scores.get(name, float(-(predicted_fallback + index)))
        for index, name in enumerate(model_names)
    }
    reference_fallback = len(reference_scores) + len(model_names)
    enriched_reference_scores = {
        name: reference_scores.get(name, float(-(reference_fallback + index)))
        for index, name in enumerate(model_names)
    }
    predicted_ranks = _rank_map(enriched_predicted_scores)
    reference_ranks = _rank_map(enriched_reference_scores)
    return _pearson(
        [predicted_ranks[name] for name in model_names],
        [reference_ranks[name] for name in model_names],
    )


def _preferred_model(
    left_name: str,
    right_name: str,
    *,
    predicted_scores: dict[str, float],
    predicted_position: dict[str, int],
) -> str:
    left_score = predicted_scores.get(left_name, float("-inf"))
    right_score = predicted_scores.get(right_name, float("-inf"))
    if left_score > right_score:
        return left_name
    if right_score > left_score:
        return right_name

    left_position = predicted_position.get(left_name, len(predicted_position) + 1)
    right_position = predicted_position.get(right_name, len(predicted_position) + 1)
    if left_position < right_position:
        return left_name
    if right_position < left_position:
        return right_name
    return min(left_name, right_name)


def _rank_map(score_map: dict[str, float]) -> dict[str, float]:
    ordered = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        tie_end = index + 1
        while tie_end < len(ordered) and math.isclose(
            ordered[tie_end][1], ordered[index][1], rel_tol=0.0, abs_tol=1e-12
        ):
            tie_end += 1
        average_rank = (index + 1 + tie_end) / 2.0
        for tie_index in range(index, tie_end):
            ranks[ordered[tie_index][0]] = average_rank
        index = tie_end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_variance * right_variance)
    if denominator == 0.0:
        return None
    return numerator / denominator


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _mean_from_total(total: float, count: int) -> float | None:
    if count <= 0:
        return None
    return float(total / count)


def _normalize_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _build_router_key(router_family: str, router_variant: str | None) -> str:
    raw = router_family if not router_variant else f"{router_family}--{router_variant}"
    return _slugify_router_key_component(raw)


def _build_target_router_key(base_router_key: str, target_suffix: str | None) -> str:
    if target_suffix is None:
        return base_router_key
    normalized_suffix = _normalize_target_suffix(target_suffix)
    return f"{base_router_key}--{normalized_suffix}"


def _build_fallback_router_target(
    manifest_path: Path, *, target_suffix: str | None
) -> RouterEvaluationTarget:
    candidate_parts = [
        part
        for part in (manifest_path.parent.parent.name, manifest_path.parent.name)
        if part
    ]
    if len(candidate_parts) >= 2:
        router_family = candidate_parts[-2]
        router_variant = candidate_parts[-1]
    else:
        router_family = manifest_path.stem or "unknown-router"
        router_variant = None

    base_router_key = _build_router_key(router_family, router_variant)
    return RouterEvaluationTarget(
        router_key=_build_target_router_key(base_router_key, target_suffix),
        router_family=router_family,
        router_variant=router_variant,
        manifest_path=manifest_path,
        target_suffix=target_suffix,
    )


def _resolve_router_runtime_config(
    *,
    artifact_root: Path,
    router_artifact_root: Path | None = None,
    router_key: str,
    shared_cache_namespace: str | None,
    router_family: str,
    base_runtime_config: dict[str, Any] | None,
    cost_normalization_scale: float | None = None,
) -> dict[str, Any]:
    resolved = dict(base_runtime_config or {})
    if router_family not in CACHE_AWARE_ROUTER_FAMILIES:
        if shared_cache_namespace is not None:
            raise ValueError(
                "shared_cache_namespace is only supported for cache-aware router families."
            )
        return resolved

    if (
        shared_cache_namespace is not None
        and router_family not in STAGE3_SANDBOXED_CACHE_ROUTER_FAMILIES
    ):
        raise ValueError(
            "shared_cache_namespace is only supported for Stage 3 sandboxed cache router families."
        )

    if router_family == "icl_router":
        if "cache_dir" not in resolved:
            cache_root = router_artifact_root or artifact_root
            resolved["cache_dir"] = cache_root / "cache" / "runtime"
        return resolved

    cache_namespace = _resolve_stage3_cache_namespace(
        router_key=router_key,
        shared_cache_namespace=shared_cache_namespace,
    )

    analysis_settings = get_analysis_settings()
    effective_analysis_model = (
        _coerce_runtime_string(resolved.get("analysis_model"))
        or analysis_settings.llm_analysis_model
    )
    effective_embedding_model = (
        _coerce_runtime_string(resolved.get("embedding_model"))
        or analysis_settings.llm_analysis_embedding_model
    )
    request_options = _runtime_request_options_value(resolved.get("request_options"))

    resolved.setdefault("analysis_model", effective_analysis_model)
    resolved.setdefault("embedding_model", effective_embedding_model)
    resolved.setdefault("request_options", request_options)
    if cost_normalization_scale is not None:
        resolved[COST_NORMALIZATION_SCALE_KEY] = cost_normalization_scale
    if "cache_dir" in resolved:
        resolved["cache_dir"] = _resolve_stage3_cache_dir_override(
            artifact_root=artifact_root,
            cache_namespace=cache_namespace,
            cache_dir_value=resolved["cache_dir"],
        )
    else:
        resolved["cache_dir"] = _build_stage3_cache_dir(
            artifact_root=artifact_root,
            cache_namespace=cache_namespace,
            analysis_model=effective_analysis_model,
            embedding_model=effective_embedding_model,
            request_options=request_options,
        )
    return resolved


def _normalize_router_targets(
    value: Sequence[RouterTargetInput | dict[str, Any]] | None,
) -> list[RouterTargetInput]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("router_targets must be a list of target objects.")

    normalized: list[RouterTargetInput] = []
    for index, entry in enumerate(value):
        if isinstance(entry, RouterTargetInput):
            normalized_target_suffix = (
                _normalize_target_suffix(entry.target_suffix)
                if entry.target_suffix is not None
                else None
            )
            normalized_shared_cache_namespace = (
                _normalize_shared_cache_namespace(entry.shared_cache_namespace)
                if entry.shared_cache_namespace is not None
                else None
            )
            normalized.append(
                RouterTargetInput(
                    router_manifest_path=Path(entry.router_manifest_path),
                    runtime_config=(
                        dict(entry.runtime_config)
                        if isinstance(entry.runtime_config, dict)
                        else None
                    ),
                    target_suffix=normalized_target_suffix,
                    shared_cache_namespace=normalized_shared_cache_namespace,
                )
            )
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                f"router_targets[{index}] must be an object with router_manifest_path and runtime_config."
            )
        manifest_path = entry.get("router_manifest_path")
        if not isinstance(manifest_path, str) or not manifest_path.strip():
            raise ValueError(
                f"router_targets[{index}].router_manifest_path must be a non-empty string."
            )
        runtime_config = entry.get("runtime_config")
        if runtime_config is not None and not isinstance(runtime_config, dict):
            raise ValueError(
                f"router_targets[{index}].runtime_config must be an object."
            )
        target_suffix_value = entry.get("target_suffix")
        shared_cache_namespace_value = entry.get("shared_cache_namespace")
        if target_suffix_value is not None and (
            not isinstance(target_suffix_value, str) or not target_suffix_value.strip()
        ):
            raise ValueError(
                f"router_targets[{index}].target_suffix must be null or a non-empty string."
            )
        if shared_cache_namespace_value is not None and (
            not isinstance(shared_cache_namespace_value, str)
            or not shared_cache_namespace_value.strip()
        ):
            raise ValueError(
                f"router_targets[{index}].shared_cache_namespace must be null or a non-empty string."
            )
        normalized_target_suffix = None
        if isinstance(target_suffix_value, str):
            normalized_target_suffix = _normalize_target_suffix(target_suffix_value)
        normalized_shared_cache_namespace = None
        if isinstance(shared_cache_namespace_value, str):
            normalized_shared_cache_namespace = _normalize_shared_cache_namespace(
                shared_cache_namespace_value
            )
        normalized.append(
            RouterTargetInput(
                router_manifest_path=Path(manifest_path.strip()),
                runtime_config=dict(runtime_config)
                if isinstance(runtime_config, dict)
                else None,
                target_suffix=normalized_target_suffix,
                shared_cache_namespace=normalized_shared_cache_namespace,
            )
        )
    return normalized


def _normalize_target_suffix(value: str) -> str:
    normalized = _slugify_router_key_component(value)
    if not normalized:
        raise ValueError(
            "target_suffix must normalize to a non-empty router key suffix."
        )
    return normalized


def _normalize_shared_cache_namespace(value: str) -> str:
    normalized = _slugify_router_key_component(value)
    if not normalized:
        raise ValueError(
            "shared_cache_namespace must normalize to a non-empty cache namespace."
        )
    return normalized


def _slugify_router_key_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def _resolve_stage3_cache_dir_override(
    *,
    artifact_root: Path,
    cache_namespace: str,
    cache_dir_value: Any,
) -> Path:
    if not isinstance(cache_dir_value, (str, Path)):
        raise ValueError("runtime_config.cache_dir must be a string or path.")
    requested_path = Path(cache_dir_value)
    if requested_path.is_absolute():
        raise ValueError(
            "runtime_config.cache_dir must be a relative path under the Stage 3 runtime cache root."
        )
    normalized_parts = [part for part in requested_path.parts if part not in ("",)]
    if not normalized_parts or any(part in {".", ".."} for part in normalized_parts):
        raise ValueError(
            "runtime_config.cache_dir must be a safe relative path under the Stage 3 runtime cache root."
        )
    override_root = _build_stage3_cache_override_root(
        artifact_root=artifact_root, cache_namespace=cache_namespace
    )
    resolved_path = override_root.joinpath(*normalized_parts).resolve()
    approved_root = override_root.resolve()
    try:
        resolved_path.relative_to(approved_root)
    except ValueError as exc:
        raise ValueError(
            "runtime_config.cache_dir must stay within the Stage 3 runtime cache root."
        ) from exc
    return resolved_path


def _build_stage3_cache_override_root(
    *, artifact_root: Path, cache_namespace: str
) -> Path:
    dataset_id = artifact_root.parent.name
    return (
        STAGE3_RUNTIME_CACHE_ROOT / "overrides" / "runs" / dataset_id / cache_namespace
    )


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    magnitude = abs(value)
    if magnitude == 0.0:
        return "0.0000"
    if magnitude < 1.0e-4:
        return f"{value:.8f}"
    return f"{value:.4f}"


def _coerce_runtime_string(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _runtime_request_options_value(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {"chat_completions": {}, "embeddings": {}}


def _build_stage3_cache_dir(
    *,
    artifact_root: Path,
    cache_namespace: str,
    analysis_model: str,
    embedding_model: str,
    request_options: dict[str, Any],
) -> Path:
    dataset_id = artifact_root.parent.name
    cache_variant = _build_stage3_cache_variant(
        analysis_model=analysis_model,
        embedding_model=embedding_model,
        request_options=request_options,
    )
    return (
        STAGE3_RUNTIME_CACHE_ROOT
        / cache_variant
        / "runs"
        / dataset_id
        / cache_namespace
    )


def _resolve_stage3_cache_namespace(
    *, router_key: str, shared_cache_namespace: str | None
) -> str:
    if shared_cache_namespace is None:
        return router_key
    return shared_cache_namespace


def _build_stage3_cache_variant(
    *,
    analysis_model: str,
    embedding_model: str,
    request_options: dict[str, Any],
) -> str:
    model_label = re.sub(r"[^A-Za-z0-9]+", "-", analysis_model).strip("-").lower()
    payload = json.dumps(
        {
            "analysis_model": analysis_model,
            "embedding_model": embedding_model,
            "request_options": request_options,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    variant_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{model_label or 'default'}-{variant_hash}"


def _build_router_cache_metadata(
    *,
    router_family: str,
    router_key: str,
    shared_cache_namespace: str | None,
    runtime_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if router_family not in CACHE_AWARE_ROUTER_FAMILIES or not runtime_config:
        return None
    cache_dir_value = runtime_config.get("cache_dir")
    if not isinstance(cache_dir_value, (str, Path)):
        return None

    cache_dir = Path(cache_dir_value)
    cache_namespace = _resolve_stage3_cache_namespace(
        router_key=router_key,
        shared_cache_namespace=shared_cache_namespace,
    )
    storage = CacheStorage(cache_dir)
    endpoint_payloads: dict[str, Any] = {}
    unique_pending_total = 0
    raw_pending_total = 0
    unique_request_total = 0
    unique_response_total = 0

    for endpoint, endpoint_dir_name in ENDPOINT_DIRECTORY_NAMES.items():
        manifest_paths = storage.manifest_paths(endpoint)
        request_rows = list(storage.iter_request_rows(endpoint))
        request_hashes = {row.request_hash for row in request_rows}
        response_hashes = storage.response_request_hashes_from_path(
            manifest_paths.responses_path,
            endpoint=endpoint,
        )
        pending_hashes = request_hashes - response_hashes

        unique_pending_total += len(pending_hashes)
        unique_request_total += len(request_hashes)
        unique_response_total += len(response_hashes)
        raw_request_count = storage.count_jsonl_rows_from_path(
            manifest_paths.requests_path
        )
        raw_response_count = storage.count_jsonl_rows_from_path(
            manifest_paths.responses_path
        )
        raw_pending_total += max(raw_request_count - raw_response_count, 0)

        endpoint_payloads[endpoint_dir_name] = {
            "requests_path": str(manifest_paths.requests_path),
            "responses_path": str(manifest_paths.responses_path),
            "raw_request_count": raw_request_count,
            "raw_response_count": raw_response_count,
            "raw_pending_count": max(raw_request_count - raw_response_count, 0),
            "unique_request_count": len(request_hashes),
            "unique_response_count": len(response_hashes),
            "unique_pending_count": len(pending_hashes),
        }

    analysis_model = (
        _coerce_runtime_string(runtime_config.get("analysis_model")) or "default"
    )
    embedding_model = (
        _coerce_runtime_string(runtime_config.get("embedding_model")) or "default"
    )
    request_options = _runtime_request_options_value(
        runtime_config.get("request_options")
    )
    return {
        "cache_dir": str(cache_dir),
        "cache_mode": runtime_config.get("cache_mode"),
        "cache_scope": (
            CACHE_SCOPE_SHARED_NAMESPACE
            if shared_cache_namespace is not None
            else CACHE_SCOPE_ROUTER_PRIVATE
        ),
        "cache_namespace": cache_namespace,
        "cache_variant": _build_stage3_cache_variant(
            analysis_model=analysis_model,
            embedding_model=embedding_model,
            request_options=request_options,
        ),
        "deferred_requests": endpoint_payloads,
        "raw_pending_count": raw_pending_total,
        "unique_pending_count": unique_pending_total,
        "unique_request_count": unique_request_total,
        "unique_response_count": unique_response_total,
    }


def _derive_evaluation_status(
    *,
    counts: RouterEvaluationCounts,
    cache_metadata: dict[str, Any] | None,
) -> str:
    if counts.setup_failures:
        return "router_setup_failed"
    if cache_metadata is not None and cache_metadata.get("unique_pending_count", 0) > 0:
        return "deferred_pending"
    if counts.failed_prompts:
        return "completed_with_prompt_failures"
    return "completed"

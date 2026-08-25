from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console

from cache._processor import DeferredRequestProcessingError, process_deferred_requests
from cache.settings import get_settings as get_cache_settings

from .artifacts import build_artifact_root
from .cli_output import build_warning_text
from .contracts import EvaluationRunResult, RouterTargetInput
from .pipeline import (
    _build_router_cache_metadata,
    _normalize_router_targets,
    _resolve_router_runtime_config,
    render_summary_table,
    run_evaluation_pipeline,
)
from .settings import Settings, get_settings

RETRYABLE_EVALUATION_STATUSES = (
    "completed_with_prompt_failures",
    "router_setup_failed",
    "deferred_pending",
)


@dataclass(slots=True)
class PriorRouterRun:
    router_key: str
    router_family: str
    router_variant: str | None
    manifest_path: Path
    target_suffix: str | None
    evaluation_status: str
    artifact_root: Path
    failure_records: list[dict[str, Any]]
    runtime_target_summary: dict[str, Any] | None

    @property
    def used_runtime_config(self) -> bool:
        return (
            isinstance(self.runtime_target_summary, dict)
            and "runtime_config" in self.runtime_target_summary
        )


@dataclass(slots=True)
class PriorEvaluationRun:
    manifest_path: Path
    summary_path: Path
    artifact_root: Path
    dataset_manifest_path: Path
    dataset_id: str
    routers: list[PriorRouterRun]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retry failed or deferred router targets from a prior Stage 3 evaluation run.",
    )
    parser.add_argument("--evaluation-manifest-path", required=True)
    parser.add_argument("--run-label")
    parser.add_argument(
        "--router-key",
        action="append",
        help="Retry only the selected router key. Repeat to include multiple routers.",
    )
    parser.add_argument(
        "--include-status",
        action="append",
        choices=sorted(RETRYABLE_EVALUATION_STATUSES),
        help=(
            "Retry only routers with the selected evaluation status. "
            "Repeat to include multiple statuses."
        ),
    )
    parser.add_argument(
        "--router-targets-json-file",
        help=(
            "Path to a JSON file containing router target objects from the prior run. "
            "Use this when a retried router depended on runtime_config values that are not "
            "preserved in Stage 3 artifacts."
        ),
    )
    parser.add_argument(
        "--process-deferred-cache",
        action="store_true",
        help=(
            "For deferred_pending routers, fulfill any pending cache requests from the prior run "
            "before rerunning, then replay the copied cache in the new Stage 3 run."
        ),
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
    result = retry_failed_evaluation_run(
        evaluation_manifest_path=Path(args.evaluation_manifest_path),
        run_label=run_label,
        router_keys=args.router_key or [],
        include_statuses=args.include_status or [],
        router_targets_json_file=(
            Path(args.router_targets_json_file)
            if args.router_targets_json_file is not None
            else None
        ),
        process_deferred_cache=args.process_deferred_cache,
        report_enabled=not args.no_report,
        settings=get_settings(),
    )
    Console(width=200).print(render_summary_table(result.router_summaries))
    warning_text = build_warning_text(result.router_summaries)
    if warning_text:
        print(warning_text, file=sys.stderr)
    return 0


def retry_failed_evaluation_run(
    *,
    evaluation_manifest_path: Path,
    run_label: str,
    router_keys: Sequence[str] | None = None,
    include_statuses: Sequence[str] | None = None,
    router_targets_json_file: Path | None = None,
    process_deferred_cache: bool = False,
    report_enabled: bool = True,
    settings: Settings | None = None,
) -> EvaluationRunResult:
    prior_run = load_prior_evaluation_run(evaluation_manifest_path)
    runtime_settings = settings or get_settings()
    include_status_set = set(include_statuses or RETRYABLE_EVALUATION_STATUSES)
    _validate_requested_statuses(include_status_set)
    selected_routers = select_retry_routers(
        prior_run,
        include_statuses=include_status_set,
        router_keys=set(router_keys or []),
    )
    new_artifact_root = build_artifact_root(
        runtime_settings.data_dir,
        prior_run.dataset_id,
        run_label,
    )
    _validate_new_artifact_root(
        prior_manifest_path=prior_run.manifest_path,
        prior_artifact_root=prior_run.artifact_root,
        new_artifact_root=new_artifact_root,
    )
    router_targets_lookup = load_router_targets_lookup(router_targets_json_file)
    retry_targets = build_retry_router_targets(
        prior_run=prior_run,
        selected_routers=selected_routers,
        new_artifact_root=new_artifact_root,
        router_targets_lookup=router_targets_lookup,
        process_deferred_cache=process_deferred_cache,
    )
    return run_evaluation_pipeline(
        dataset_manifest_path=prior_run.dataset_manifest_path,
        router_targets=retry_targets,
        run_label=run_label,
        report_enabled=report_enabled,
        settings=runtime_settings,
    )


def load_prior_evaluation_run(evaluation_manifest_path: Path) -> PriorEvaluationRun:
    manifest_payload = _load_json_object(
        evaluation_manifest_path,
        description="Stage 3 evaluation manifest",
    )
    artifact_root = evaluation_manifest_path.parent
    summary_path = artifact_root / "summary.json"
    summary_payload = _load_json_object(
        summary_path,
        description="Stage 3 evaluation summary",
    )
    inputs = manifest_payload.get("inputs")
    dataset = manifest_payload.get("dataset")
    manifest_routers = manifest_payload.get("routers")
    summary_routers = summary_payload.get("routers")
    if not isinstance(inputs, dict):
        raise ValueError("Stage 3 evaluation manifest inputs must be an object.")
    if not isinstance(dataset, dict):
        raise ValueError("Stage 3 evaluation manifest dataset must be an object.")
    if not isinstance(manifest_routers, list):
        raise ValueError("Stage 3 evaluation manifest routers must be a list.")
    if not isinstance(summary_routers, list):
        raise ValueError("Stage 3 evaluation summary routers must be a list.")

    dataset_manifest_path_value = inputs.get("dataset_manifest_path")
    dataset_id_value = dataset.get("dataset_id")
    if (
        not isinstance(dataset_manifest_path_value, str)
        or not dataset_manifest_path_value.strip()
    ):
        raise ValueError(
            "Stage 3 evaluation manifest inputs.dataset_manifest_path must be a non-empty string."
        )
    if not isinstance(dataset_id_value, str) or not dataset_id_value.strip():
        raise ValueError(
            "Stage 3 evaluation manifest dataset.dataset_id must be a non-empty string."
        )

    manifest_router_by_key: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(manifest_routers):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest.routers[{index}] must be an object.")
        router_key = entry.get("router_key")
        if not isinstance(router_key, str) or not router_key.strip():
            raise ValueError(
                f"manifest.routers[{index}].router_key must be a non-empty string."
            )
        manifest_router_by_key[router_key] = entry

    router_target_summary_by_selector = _build_router_target_summary_lookup(inputs)
    routers: list[PriorRouterRun] = []
    for index, entry in enumerate(summary_routers):
        if not isinstance(entry, dict):
            raise ValueError(f"summary.routers[{index}] must be an object.")
        router_key = entry.get("router_key")
        if not isinstance(router_key, str) or not router_key.strip():
            raise ValueError(
                f"summary.routers[{index}].router_key must be a non-empty string."
            )
        manifest_router = manifest_router_by_key.get(router_key)
        if manifest_router is None:
            raise ValueError(
                f"Stage 3 evaluation manifest is missing router metadata for '{router_key}'."
            )
        manifest_path_value = manifest_router.get("manifest_path")
        router_family_value = manifest_router.get("router_family")
        router_variant_value = manifest_router.get("router_variant")
        target_suffix_value = manifest_router.get("target_suffix")
        if not isinstance(manifest_path_value, str) or not manifest_path_value.strip():
            raise ValueError(
                f"Router '{router_key}' manifest_path must be a non-empty string."
            )
        if not isinstance(router_family_value, str) or not router_family_value.strip():
            raise ValueError(
                f"Router '{router_key}' router_family must be a non-empty string."
            )
        if router_variant_value is not None and not isinstance(
            router_variant_value, str
        ):
            raise ValueError(
                f"Router '{router_key}' router_variant must be null or a string."
            )
        if target_suffix_value is not None and not isinstance(target_suffix_value, str):
            raise ValueError(
                f"Router '{router_key}' target_suffix must be null or a string."
            )

        router_root = artifact_root / "routers" / router_key
        router_summary_path = router_root / "summary.json"
        router_failures_path = router_root / "failures.jsonl"
        router_summary_payload = _load_json_object(
            router_summary_path,
            description=f"router summary for '{router_key}'",
        )
        failure_records = _load_jsonl_rows(
            router_failures_path,
            description=f"router failures for '{router_key}'",
        )
        router_summary_key = router_summary_payload.get("router_key")
        if router_summary_key != router_key:
            raise ValueError(
                f"Router summary for '{router_key}' reported router_key '{router_summary_key}'."
            )
        evaluation_status_value = router_summary_payload.get("evaluation_status")
        if (
            not isinstance(evaluation_status_value, str)
            or not evaluation_status_value.strip()
        ):
            raise ValueError(
                f"Router '{router_key}' summary evaluation_status must be a non-empty string."
            )

        router_selector = _build_router_selector(
            manifest_path=Path(manifest_path_value),
            target_suffix=target_suffix_value,
        )
        routers.append(
            PriorRouterRun(
                router_key=router_key,
                router_family=router_family_value.strip(),
                router_variant=router_variant_value.strip()
                if isinstance(router_variant_value, str)
                else None,
                manifest_path=Path(manifest_path_value),
                target_suffix=target_suffix_value.strip()
                if isinstance(target_suffix_value, str)
                else None,
                evaluation_status=evaluation_status_value.strip(),
                artifact_root=router_root,
                failure_records=failure_records,
                runtime_target_summary=router_target_summary_by_selector.get(
                    router_selector
                ),
            )
        )

    return PriorEvaluationRun(
        manifest_path=evaluation_manifest_path,
        summary_path=summary_path,
        artifact_root=artifact_root,
        dataset_manifest_path=Path(dataset_manifest_path_value),
        dataset_id=dataset_id_value.strip(),
        routers=routers,
    )


def select_retry_routers(
    prior_run: PriorEvaluationRun,
    *,
    include_statuses: set[str],
    router_keys: set[str],
) -> list[PriorRouterRun]:
    available_router_keys = {router.router_key for router in prior_run.routers}
    missing_router_keys = sorted(router_keys - available_router_keys)
    if missing_router_keys:
        raise ValueError(
            "Unknown --router-key values: " + ", ".join(missing_router_keys)
        )

    selected_routers = [
        router
        for router in prior_run.routers
        if router.evaluation_status in include_statuses
        and (not router_keys or router.router_key in router_keys)
    ]
    if not selected_routers:
        raise ValueError("No retryable router targets matched the requested filters.")
    return selected_routers


def load_router_targets_lookup(
    router_targets_json_file: Path | None,
) -> dict[tuple[str, str | None], RouterTargetInput]:
    if router_targets_json_file is None:
        return {}
    raw_payload = json.loads(router_targets_json_file.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, list):
        raise ValueError("--router-targets-json-file must decode to a JSON array.")
    normalized_targets = _normalize_router_targets(raw_payload)
    lookup: dict[tuple[str, str | None], RouterTargetInput] = {}
    for target in normalized_targets:
        selector = _build_router_selector(
            manifest_path=target.router_manifest_path,
            target_suffix=target.target_suffix,
        )
        if selector in lookup:
            raise ValueError(
                "--router-targets-json-file contains duplicate router target selectors."
            )
        lookup[selector] = target
    return lookup


def build_retry_router_targets(
    *,
    prior_run: PriorEvaluationRun,
    selected_routers: Sequence[PriorRouterRun],
    new_artifact_root: Path,
    router_targets_lookup: dict[tuple[str, str | None], RouterTargetInput],
    process_deferred_cache: bool,
) -> list[RouterTargetInput]:
    retry_targets: list[RouterTargetInput] = []
    for router in selected_routers:
        base_target = _resolve_retry_target(
            router,
            router_targets_lookup=router_targets_lookup,
        )
        if router.evaluation_status == "deferred_pending" and process_deferred_cache:
            retry_targets.append(
                _prepare_processed_deferred_retry_target(
                    router,
                    base_target=base_target,
                    prior_artifact_root=prior_run.artifact_root,
                    new_artifact_root=new_artifact_root,
                )
            )
            continue
        retry_targets.append(base_target)
    return retry_targets


def _resolve_retry_target(
    router: PriorRouterRun,
    *,
    router_targets_lookup: dict[tuple[str, str | None], RouterTargetInput],
) -> RouterTargetInput:
    selector = _build_router_selector(
        manifest_path=router.manifest_path,
        target_suffix=router.target_suffix,
    )
    matched_target = router_targets_lookup.get(selector)
    if matched_target is not None:
        return RouterTargetInput(
            router_manifest_path=matched_target.router_manifest_path,
            runtime_config=(
                dict(matched_target.runtime_config)
                if matched_target.runtime_config is not None
                else None
            ),
            target_suffix=matched_target.target_suffix,
            shared_cache_namespace=matched_target.shared_cache_namespace,
        )
    if router.used_runtime_config:
        raise ValueError(
            f"Router '{router.router_key}' used runtime_config in the prior run. "
            "Pass --router-targets-json-file with the original router target objects to replay it."
        )
    shared_cache_namespace = None
    if isinstance(router.runtime_target_summary, dict):
        shared_cache_namespace_value = router.runtime_target_summary.get(
            "shared_cache_namespace"
        )
        if (
            isinstance(shared_cache_namespace_value, str)
            and shared_cache_namespace_value.strip()
        ):
            shared_cache_namespace = shared_cache_namespace_value.strip()
    return RouterTargetInput(
        router_manifest_path=router.manifest_path,
        runtime_config=None,
        target_suffix=router.target_suffix,
        shared_cache_namespace=shared_cache_namespace,
    )


def _prepare_processed_deferred_retry_target(
    router: PriorRouterRun,
    *,
    base_target: RouterTargetInput,
    prior_artifact_root: Path,
    new_artifact_root: Path,
) -> RouterTargetInput:
    prior_runtime_config = (
        dict(base_target.runtime_config)
        if base_target.runtime_config is not None
        else None
    )
    prior_effective_runtime = _resolve_router_runtime_config(
        artifact_root=prior_artifact_root,
        router_artifact_root=router.manifest_path.parent,
        router_key=router.router_key,
        shared_cache_namespace=base_target.shared_cache_namespace,
        router_family=router.router_family,
        base_runtime_config=prior_runtime_config,
    )
    prior_cache_metadata = _build_router_cache_metadata(
        router_family=router.router_family,
        router_key=router.router_key,
        shared_cache_namespace=base_target.shared_cache_namespace,
        runtime_config=prior_effective_runtime,
    )
    if prior_cache_metadata is None:
        raise ValueError(
            f"Deferred router '{router.router_key}' did not resolve to reusable cache metadata."
        )

    _process_router_deferred_cache(
        router_key=router.router_key,
        cache_metadata=prior_cache_metadata,
    )
    refreshed_cache_metadata = _build_router_cache_metadata(
        router_family=router.router_family,
        router_key=router.router_key,
        shared_cache_namespace=base_target.shared_cache_namespace,
        runtime_config=prior_effective_runtime,
    )
    if refreshed_cache_metadata is None:
        raise ValueError(
            f"Deferred router '{router.router_key}' cache metadata disappeared after processing."
        )
    remaining_pending = refreshed_cache_metadata.get("unique_pending_count")
    if isinstance(remaining_pending, int) and remaining_pending > 0:
        raise RuntimeError(
            f"Deferred router '{router.router_key}' still has {remaining_pending} pending cache request(s) after processing."
        )

    prior_cache_dir_value = prior_effective_runtime.get("cache_dir")
    if not isinstance(prior_cache_dir_value, (str, Path)):
        raise ValueError(
            f"Deferred router '{router.router_key}' resolved an invalid cache_dir."
        )
    prior_cache_dir = Path(prior_cache_dir_value)

    retry_runtime_config = dict(base_target.runtime_config or {})
    retry_runtime_config["cache_mode"] = "replay_only"
    retry_effective_runtime = _resolve_router_runtime_config(
        artifact_root=new_artifact_root,
        router_artifact_root=router.manifest_path.parent,
        router_key=router.router_key,
        shared_cache_namespace=base_target.shared_cache_namespace,
        router_family=router.router_family,
        base_runtime_config=retry_runtime_config,
    )
    new_cache_dir_value = retry_effective_runtime.get("cache_dir")
    if not isinstance(new_cache_dir_value, (str, Path)):
        raise ValueError(
            f"Deferred router '{router.router_key}' retry target resolved an invalid cache_dir."
        )
    new_cache_dir = Path(new_cache_dir_value)
    _copy_cache_directory(source=prior_cache_dir, destination=new_cache_dir)
    return RouterTargetInput(
        router_manifest_path=base_target.router_manifest_path,
        runtime_config=retry_runtime_config,
        target_suffix=base_target.target_suffix,
        shared_cache_namespace=base_target.shared_cache_namespace,
    )


def _process_router_deferred_cache(
    *,
    router_key: str,
    cache_metadata: dict[str, Any],
) -> None:
    deferred_requests = cache_metadata.get("deferred_requests")
    if not isinstance(deferred_requests, dict):
        return
    cache_settings = get_cache_settings()
    for endpoint_name in sorted(deferred_requests):
        payload = deferred_requests[endpoint_name]
        if not isinstance(payload, dict):
            continue
        unique_pending_count = payload.get("unique_pending_count")
        requests_path_value = payload.get("requests_path")
        if (
            not isinstance(unique_pending_count, int)
            or unique_pending_count <= 0
            or not isinstance(requests_path_value, str)
            or not requests_path_value
        ):
            continue
        try:
            summary = process_deferred_requests(
                requests_path=requests_path_value,
                api_key=cache_settings.llm_api_key,
                base_url=cache_settings.llm_base_url,
            )
        except DeferredRequestProcessingError as exc:
            raise RuntimeError(
                f"Failed processing deferred cache for router '{router_key}' endpoint '{endpoint_name}'."
            ) from exc
        if summary.failed:
            raise RuntimeError(
                f"Deferred cache processing for router '{router_key}' endpoint '{endpoint_name}' reported {summary.failed} failure(s)."
            )


def _copy_cache_directory(*, source: Path, destination: Path) -> None:
    if not source.exists():
        raise ValueError(f"Deferred cache directory does not exist: {source}")
    if source.resolve(strict=False) == destination.resolve(strict=False):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _build_router_target_summary_lookup(
    manifest_inputs: dict[str, Any],
) -> dict[tuple[str, str | None], dict[str, Any]]:
    router_targets_summary = manifest_inputs.get("router_targets_summary")
    if router_targets_summary is None:
        return {}
    if not isinstance(router_targets_summary, list):
        raise ValueError(
            "Stage 3 evaluation manifest inputs.router_targets_summary must be a list when present."
        )
    lookup: dict[tuple[str, str | None], dict[str, Any]] = {}
    for index, entry in enumerate(router_targets_summary):
        if not isinstance(entry, dict):
            raise ValueError(
                f"inputs.router_targets_summary[{index}] must be an object."
            )
        manifest_path_value = entry.get("router_manifest_path")
        target_suffix_value = entry.get("target_suffix")
        if not isinstance(manifest_path_value, str) or not manifest_path_value.strip():
            raise ValueError(
                f"inputs.router_targets_summary[{index}].router_manifest_path must be a non-empty string."
            )
        if target_suffix_value is not None and not isinstance(target_suffix_value, str):
            raise ValueError(
                f"inputs.router_targets_summary[{index}].target_suffix must be null or a string."
            )
        selector = _build_router_selector(
            manifest_path=Path(manifest_path_value),
            target_suffix=target_suffix_value,
        )
        if selector in lookup:
            raise ValueError(
                "Stage 3 evaluation manifest contains duplicate router target summaries."
            )
        lookup[selector] = entry
    return lookup


def _build_router_selector(
    *, manifest_path: Path, target_suffix: str | None
) -> tuple[str, str | None]:
    return (str(manifest_path.resolve()), target_suffix)


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must decode to a JSON object.")
    return payload


def _load_jsonl_rows(path: Path, *, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not raw_line.strip():
            continue
        entry = json.loads(raw_line)
        if not isinstance(entry, dict):
            raise ValueError(f"{description}[{index}] must decode to a JSON object.")
        rows.append(entry)
    return rows


def _validate_new_artifact_root(
    *,
    prior_manifest_path: Path,
    prior_artifact_root: Path,
    new_artifact_root: Path,
) -> None:
    if new_artifact_root.resolve() == prior_artifact_root.resolve():
        raise ValueError(
            "Retry runs must use a new run label and may not overwrite the prior Stage 3 artifact root."
        )
    if new_artifact_root.exists():
        raise ValueError(
            f"Retry target artifact root already exists: {new_artifact_root}. "
            f"Choose a new --run-label instead of mutating {prior_manifest_path.parent}."
        )


def _validate_requested_statuses(include_statuses: set[str]) -> None:
    invalid_statuses = sorted(include_statuses - set(RETRYABLE_EVALUATION_STATUSES))
    if invalid_statuses:
        raise ValueError(
            "Unsupported retry status values: " + ", ".join(invalid_statuses)
        )


if __name__ == "__main__":
    raise SystemExit(main())

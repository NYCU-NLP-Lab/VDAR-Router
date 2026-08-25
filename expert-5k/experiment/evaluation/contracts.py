from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _jsonify_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonify_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify_value(item) for item in value]
    return value


_RUNTIME_SECRET_KEY_FRAGMENTS = {
    "api_key",
    "apikey",
    "authorization",
    "secret",
    "token",
    "password",
    "cookie",
}


def _is_secret_like_key(value: str) -> bool:
    normalized = value.strip().lower().replace("-", "_")
    return any(fragment in normalized for fragment in _RUNTIME_SECRET_KEY_FRAGMENTS)


def _sanitize_runtime_config(value: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(str(key) for key in value)
    summary: dict[str, Any] = {
        "keys": keys,
        "has_cache_dir": "cache_dir" in value,
        "has_request_options": "request_options" in value,
        "redacted_keys": sorted(
            str(key) for key in value if _is_secret_like_key(str(key))
        ),
    }
    cache_mode = value.get("cache_mode")
    if isinstance(cache_mode, str) and cache_mode.strip():
        summary["cache_mode"] = cache_mode.strip()
    request_options = value.get("request_options")
    if isinstance(request_options, dict):
        summary["request_option_keys"] = sorted(str(key) for key in request_options)
    return summary


def _sanitize_runtime_target_entry(value: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "router_manifest_path": _jsonify_value(value.get("router_manifest_path")),
        "target_suffix": value.get("target_suffix"),
    }
    shared_cache_namespace = value.get("shared_cache_namespace")
    if isinstance(shared_cache_namespace, str) and shared_cache_namespace.strip():
        summary["shared_cache_namespace"] = shared_cache_namespace.strip()
    runtime_config = value.get("runtime_config")
    if isinstance(runtime_config, dict):
        summary["runtime_config"] = _sanitize_runtime_config(runtime_config)
    return summary


def _sanitize_router_target_input(value: "RouterTargetInput") -> dict[str, Any]:
    return _sanitize_runtime_target_entry(
        {
            "router_manifest_path": value.router_manifest_path,
            "target_suffix": value.target_suffix,
            "shared_cache_namespace": value.shared_cache_namespace,
            "runtime_config": value.runtime_config,
        }
    )


def _sanitize_cache_payload(value: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    cache_mode = value.get("cache_mode")
    if cache_mode is not None:
        summary["cache_mode"] = cache_mode
    cache_scope = value.get("cache_scope")
    if cache_scope is not None:
        summary["cache_scope"] = cache_scope
    cache_variant = value.get("cache_variant")
    if cache_variant is not None:
        summary["cache_variant"] = cache_variant
    cache_namespace = value.get("cache_namespace")
    if cache_namespace is not None:
        summary["cache_namespace"] = cache_namespace
    for key in (
        "raw_pending_count",
        "unique_pending_count",
        "unique_request_count",
        "unique_response_count",
    ):
        if key in value:
            summary[key] = value[key]

    deferred_requests = value.get("deferred_requests")
    if isinstance(deferred_requests, dict):
        summary["deferred_requests"] = {
            str(endpoint): {
                subkey: payload[subkey]
                for subkey in (
                    "raw_request_count",
                    "raw_response_count",
                    "raw_pending_count",
                    "unique_request_count",
                    "unique_response_count",
                    "unique_pending_count",
                )
                if isinstance(payload, dict) and subkey in payload
            }
            for endpoint, payload in deferred_requests.items()
        }
    return summary


@dataclass(slots=True)
class CanonicalRow:
    id: str
    prompt_id: str
    input: str
    output: str
    score: float
    input_token: int
    output_tokken: int
    metadata: dict[str, Any]


@dataclass(slots=True)
class Stage1DatasetManifest:
    manifest_path: Path
    artifact_root: Path
    dataset_id: str
    dataset_version: str
    score_mode: str
    has_output: bool
    llm_config_path: Path
    full_path: Path
    train_path: Path
    test_path: Path


@dataclass(slots=True)
class BaselineProfile:
    name: str
    source_split: str
    semantic_role: str
    allowed_metrics: list[str]
    model_scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_split": self.source_split,
            "semantic_role": self.semantic_role,
            "allowed_metrics": list(self.allowed_metrics),
            "model_scores": dict(self.model_scores),
        }


@dataclass(slots=True)
class RouterTargetInput:
    router_manifest_path: Path
    runtime_config: dict[str, Any] | None = None
    target_suffix: str | None = None
    shared_cache_namespace: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "router_manifest_path": str(self.router_manifest_path),
            "runtime_config": _jsonify_value(self.runtime_config),
            "target_suffix": self.target_suffix,
            "shared_cache_namespace": self.shared_cache_namespace,
        }


@dataclass(slots=True)
class RouterEvaluationTarget:
    router_key: str
    router_family: str
    router_variant: str | None
    manifest_path: Path
    target_suffix: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "router_key": self.router_key,
            "router_family": self.router_family,
            "router_variant": self.router_variant,
            "manifest_path": str(self.manifest_path),
            "target_suffix": self.target_suffix,
        }


@dataclass(slots=True)
class RouterEvaluationCounts:
    total_prompts: int
    succeeded_prompts: int
    failed_prompts: int
    setup_failures: int
    skipped_prompts: int
    scored_pairs: int
    priced_prompts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_prompts": self.total_prompts,
            "succeeded_prompts": self.succeeded_prompts,
            "failed_prompts": self.failed_prompts,
            "setup_failures": self.setup_failures,
            "skipped_prompts": self.skipped_prompts,
            "scored_pairs": self.scored_pairs,
            "priced_prompts": self.priced_prompts,
        }


@dataclass(slots=True)
class RouterEvaluationMetrics:
    pairwise_accuracy: float | None
    average_train_elo_spearman: float | None
    average_full_elo_spearman: float | None
    average_prompt_cost: float | None
    total_prompt_cost: float | None
    total_actual_prompt_cost: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairwise_accuracy": self.pairwise_accuracy,
            "average_train_elo_spearman": self.average_train_elo_spearman,
            "average_full_elo_spearman": self.average_full_elo_spearman,
            "average_prompt_cost": self.average_prompt_cost,
            "total_prompt_cost": self.total_prompt_cost,
            "total_actual_prompt_cost": self.total_actual_prompt_cost,
        }


@dataclass(slots=True)
class RouterEvaluationSummary:
    router_key: str
    router_family: str
    router_variant: str | None
    manifest_path: Path
    target_suffix: str | None
    artifact_root: Path
    counts: RouterEvaluationCounts
    metrics: RouterEvaluationMetrics
    evaluation_status: str = "completed"
    runtime_config_effective: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] | None = None
    warnings: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "router_key": self.router_key,
            "router_family": self.router_family,
            "router_variant": self.router_variant,
            "manifest_path": str(self.manifest_path),
            "target_suffix": self.target_suffix,
            "artifact_root": str(self.artifact_root),
            "counts": self.counts.to_dict(),
            "metrics": self.metrics.to_dict(),
            "evaluation_status": self.evaluation_status,
            "runtime_config_effective": _sanitize_runtime_config(
                self.runtime_config_effective
            ),
        }
        if self.cache is not None:
            payload["cache"] = _sanitize_cache_payload(self.cache)
        if self.warnings is not None:
            payload["warnings"] = _jsonify_value(self.warnings)
        return payload


@dataclass(slots=True)
class EvaluationManifest:
    schema_version: str
    dataset_manifest_path: Path
    artifact_root: Path
    dataset_id: str
    dataset_version: str
    baselines: list[BaselineProfile]
    routers: list[RouterEvaluationTarget]
    router_targets: list[RouterTargetInput] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        inputs = {
            "dataset_manifest_path": str(self.dataset_manifest_path),
            "router_manifest_paths": [
                str(router.manifest_path) for router in self.routers
            ],
        }
        inputs["router_targets_summary"] = [
            _sanitize_router_target_input(router_target)
            for router_target in self.router_targets
        ]

        return {
            "schema_version": self.schema_version,
            "inputs": inputs,
            "outputs": {
                "artifact_root": str(self.artifact_root),
                "manifest": "manifest.json",
                "summary": "summary.json",
                "routers": "routers",
            },
            "dataset": {
                "dataset_id": self.dataset_id,
                "dataset_version": self.dataset_version,
            },
            "baselines": [baseline.to_dict() for baseline in self.baselines],
            "routers": [router.to_dict() for router in self.routers],
        }


@dataclass(slots=True)
class EvaluationRunResult:
    artifact_root: Path
    manifest_path: Path
    summary_path: Path
    manifest: EvaluationManifest
    router_summaries: list[RouterEvaluationSummary] = field(default_factory=list)

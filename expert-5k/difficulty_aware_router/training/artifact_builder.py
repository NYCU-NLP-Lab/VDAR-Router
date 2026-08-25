from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from cache._constants import SUPPORTED_CACHE_MODES
from cache._contracts import CacheMode
from difficulty_aware_router.agents.difficulty_analysis import PROMPT_REGISTRY
from difficulty_aware_router.agents.settings import get_analysis_settings
from experiment.generate_dataset.model_mapping import (
    load_llm_config,
    normalize_model_name,
)
from experiment.training.contracts import TrainArtifactRef

ROUTER_ARTIFACT_FILENAME = "router.json"
CANDIDATE_MODELS_FILENAME = "candidate_models.json"
MODEL_COST_STATS_FILENAME = "model_cost_stats.json"
TRAINING_METADATA_FILENAME = "difficulty_aware_artifact_manifest.json"
CHROMA_DIRNAME = "chroma"
TRAINING_METADATA_SCHEMA_VERSION = "difficulty-aware.training-artifact.v2"
SHARED_COLLECTION_NAME = "difficulty_aware_shared"
MODEL_COLLECTION_PREFIX = "difficulty_aware_model__"


@dataclass(slots=True)
class DifficultyAwareTrainingConfig:
    dry_run: bool = False
    limit: int = 0
    analysis_prompt_version: str = "v1"
    analysis_model: str = "gpt-5.2"
    analysis_embedding_model: str = "text-embedding-3-large"
    cache_mode: CacheMode | dict[str, CacheMode] = "record"
    chat_cache_mode: CacheMode = "record"
    embedding_cache_mode: CacheMode = "record"
    request_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    chat_request_options: dict[str, Any] = field(default_factory=dict)
    embedding_request_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DifficultyAwareArtifactBundle:
    router_artifact_path: Path
    candidate_models_path: Path
    model_cost_stats_path: Path
    training_metadata_path: Path
    chroma_dir: Path
    rows: list[dict[str, Any]]
    candidate_models: list[str]
    shared_collection_name: str
    model_collection_names: dict[str, str]


def normalize_training_config(config: dict[str, Any]) -> DifficultyAwareTrainingConfig:
    analysis_settings = get_analysis_settings()
    cache_mode, chat_cache_mode, embedding_cache_mode = _normalize_cache_mode_config(
        config.get("cache_mode", "record"),
        name="cache_mode",
    )
    (
        request_options,
        chat_request_options,
        embedding_request_options,
    ) = _normalize_request_options_config(
        config.get("request_options", {}),
        name="request_options",
    )
    limit = _normalize_non_negative_int(config.get("limit", 0), name="limit")
    dry_run = _normalize_bool(config.get("dry_run", False), name="dry_run")
    return DifficultyAwareTrainingConfig(
        dry_run=dry_run,
        limit=limit,
        analysis_prompt_version=_normalize_analysis_prompt_version(
            config.get("analysis_prompt_version", "v1")
        ),
        analysis_model=_normalize_non_empty_string(
            config.get("analysis_model", analysis_settings.llm_analysis_model),
            name="analysis_model",
        ),
        analysis_embedding_model=_normalize_non_empty_string(
            config.get(
                "analysis_embedding_model",
                config.get(
                    "embedding_model_name",
                    analysis_settings.llm_analysis_embedding_model,
                ),
            ),
            name="analysis_embedding_model",
        ),
        cache_mode=cache_mode,
        chat_cache_mode=chat_cache_mode,
        embedding_cache_mode=embedding_cache_mode,
        request_options=request_options,
        chat_request_options=chat_request_options,
        embedding_request_options=embedding_request_options,
    )


def build_training_artifacts(
    *,
    train_artifact: TrainArtifactRef,
    output_dir: Path,
    config: DifficultyAwareTrainingConfig,
) -> DifficultyAwareArtifactBundle:
    rows = load_stage1_rows(train_artifact.train_path)
    candidate_models = extract_candidate_models(rows)
    shared_collection_name = SHARED_COLLECTION_NAME
    model_collection_names = build_model_collection_names(candidate_models)

    chroma_dir = output_dir / CHROMA_DIRNAME

    router_artifact_path = output_dir / ROUTER_ARTIFACT_FILENAME
    candidate_models_path = output_dir / CANDIDATE_MODELS_FILENAME
    model_cost_stats_path = output_dir / MODEL_COST_STATS_FILENAME
    training_metadata_path = output_dir / TRAINING_METADATA_FILENAME

    return DifficultyAwareArtifactBundle(
        router_artifact_path=router_artifact_path,
        candidate_models_path=candidate_models_path,
        model_cost_stats_path=model_cost_stats_path,
        training_metadata_path=training_metadata_path,
        chroma_dir=chroma_dir,
        rows=rows,
        candidate_models=candidate_models,
        shared_collection_name=shared_collection_name,
        model_collection_names=model_collection_names,
    )


def materialize_training_artifacts(
    *,
    artifact_bundle: DifficultyAwareArtifactBundle,
    train_artifact: TrainArtifactRef,
    config: DifficultyAwareTrainingConfig,
    processed_row_count: int,
    indexed_row_count: int,
    missing_summary_skip_count: int,
    analysis_status: str,
    router_behavior: dict[str, Any] | None = None,
) -> None:
    candidate_models_payload = {
        "candidate_models": artifact_bundle.candidate_models,
    }
    write_json(artifact_bundle.candidate_models_path, candidate_models_payload)
    write_json(
        artifact_bundle.router_artifact_path,
        {
            **candidate_models_payload,
            "artifact_manifest": TRAINING_METADATA_FILENAME,
            "candidate_models_path": CANDIDATE_MODELS_FILENAME,
            "chroma_dir": CHROMA_DIRNAME,
            "collections": {
                "shared": artifact_bundle.shared_collection_name,
                "by_model": artifact_bundle.model_collection_names,
            },
            "model_cost_stats_path": MODEL_COST_STATS_FILENAME,
            "router_behavior": dict(router_behavior or {}),
        },
    )
    write_json(
        artifact_bundle.model_cost_stats_path,
        build_model_cost_stats(artifact_bundle.rows, train_artifact=train_artifact),
    )
    write_json(
        artifact_bundle.training_metadata_path,
        build_training_metadata(
            train_artifact=train_artifact,
            config=config,
            candidate_models=artifact_bundle.candidate_models,
            source_row_count=len(artifact_bundle.rows),
            processed_row_count=processed_row_count,
            indexed_row_count=indexed_row_count,
            missing_summary_skip_count=missing_summary_skip_count,
            shared_collection_name=artifact_bundle.shared_collection_name,
            model_collection_names=artifact_bundle.model_collection_names,
            analysis_status=analysis_status,
            router_behavior=router_behavior,
        ),
    )


def load_stage1_rows(train_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with train_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Stage 1 train artifact contains invalid JSON on line {line_number}."
                ) from exc

            if not isinstance(payload, dict):
                raise ValueError(
                    "Stage 1 train artifact must contain one JSON object per line "
                    f"(line {line_number})."
                )
            rows.append(payload)
    return rows


def extract_candidate_models(rows: list[dict[str, Any]]) -> list[str]:
    candidate_models: list[str] = []
    seen: set[str] = set()

    for row in rows:
        model_name = _extract_model_name(row)
        if model_name is None:
            continue
        if model_name in seen:
            continue
        seen.add(model_name)
        candidate_models.append(model_name)

    if not candidate_models:
        raise ValueError(
            "Stage 1 train artifact must contain at least one metadata.raw_model_name value."
        )

    return candidate_models


def prepare_training_row(
    *, row: dict[str, Any], dataset_id: str, dataset_version: str
) -> dict[str, Any]:
    row_id = _require_non_empty_string(row.get("id"), field_name="id")
    prompt_id = _require_non_empty_string(row.get("prompt_id"), field_name="prompt_id")
    question = _require_non_empty_string(row.get("input"), field_name="input")
    score = _require_number(row.get("score"), field_name="score")
    input_token = _require_int(row.get("input_token"), field_name="input_token")
    output_tokken = _require_int(row.get("output_tokken"), field_name="output_tokken")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Stage 1 row field metadata must be an object.")

    model_name = metadata.get("raw_model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(
            "Stage 1 row field metadata.raw_model_name must be a non-empty string."
        )

    normalized_model_name = model_name.strip()
    correctness = map_score_to_correctness(score)
    record_id = f"{dataset_id}:{dataset_version}:{row_id}:{normalized_model_name}"
    return {
        "record_id": record_id,
        "model_name": normalized_model_name,
        "question": question,
        "metadata": {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "question": question,
            "raw_model_name": normalized_model_name,
            "row_id": row_id,
            "prompt_id": prompt_id,
            "score": score,
            "correctness": correctness,
            "input_token": input_token,
            "output_tokken": output_tokken,
            "row_metadata_json": json.dumps(metadata, sort_keys=True),
        },
    }


def map_score_to_correctness(score: float) -> str:
    if score >= 1.0:
        return "Preffered"
    if score > 0:
        return "Tie"
    return "Not Preffered"


def build_model_cost_stats(
    rows: list[dict[str, Any]], *, train_artifact: TrainArtifactRef
) -> dict[str, dict[str, float]]:
    pricing_by_model = _load_model_pricing_by_raw_name(train_artifact.manifest_path)
    stats_by_model: dict[str, dict[str, float]] = {}
    for row in rows:
        model_name = _extract_model_name(row)
        if model_name is None:
            continue

        stats = stats_by_model.setdefault(
            model_name,
            {
                "response_tokens_sum": 0.0,
                "response_tokens_count": 0.0,
                "output_cost_per_token_sum": 0.0,
                "output_cost_per_token_count": 0.0,
            },
        )
        response_tokens = row.get("output_tokken")
        if isinstance(response_tokens, (int, float)) and not isinstance(
            response_tokens, bool
        ):
            stats["response_tokens_sum"] += float(response_tokens)
            stats["response_tokens_count"] += 1.0

        metadata = row.get("metadata")
        raw_cost = metadata.get("cost") if isinstance(metadata, dict) else None
        if (
            isinstance(raw_cost, (int, float))
            and not isinstance(raw_cost, bool)
            and isinstance(response_tokens, (int, float))
            and not isinstance(response_tokens, bool)
            and float(response_tokens) > 0
        ):
            stats["output_cost_per_token_sum"] += float(raw_cost) / float(
                response_tokens
            )
            stats["output_cost_per_token_count"] += 1.0

    model_cost_stats: dict[str, dict[str, float]] = {}
    for model_name, stats in stats_by_model.items():
        response_count = stats["response_tokens_count"]
        price_count = stats["output_cost_per_token_count"]
        pricing = pricing_by_model.get(model_name)
        model_cost_stats[model_name] = {
            "avg_response_tokens": (
                stats["response_tokens_sum"] / response_count
                if response_count > 0
                else 0.0
            ),
            "input_cost_per_token": (
                pricing["input_cost_per_token"] if pricing is not None else 0.0
            ),
            "output_cost_per_token": (
                pricing["output_cost_per_token"]
                if pricing is not None
                else (
                    stats["output_cost_per_token_sum"] / price_count
                    if price_count > 0
                    else 0.0
                )
            ),
        }
    return model_cost_stats


def build_training_metadata(
    *,
    train_artifact: TrainArtifactRef,
    config: DifficultyAwareTrainingConfig,
    candidate_models: list[str],
    source_row_count: int,
    processed_row_count: int,
    indexed_row_count: int,
    missing_summary_skip_count: int,
    shared_collection_name: str,
    model_collection_names: dict[str, str],
    analysis_status: str,
    router_behavior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_METADATA_SCHEMA_VERSION,
        "dataset": {
            "dataset_id": train_artifact.dataset_id,
            "dataset_version": train_artifact.dataset_version,
            "manifest_path": str(train_artifact.manifest_path.resolve()),
            "train_path": str(train_artifact.train_path.resolve()),
        },
        "outputs": {
            "router_artifact": ROUTER_ARTIFACT_FILENAME,
            "candidate_models": CANDIDATE_MODELS_FILENAME,
            "model_cost_stats": MODEL_COST_STATS_FILENAME,
            "chroma_dir": CHROMA_DIRNAME,
            "collections": {
                "shared": shared_collection_name,
                "by_model": model_collection_names,
            },
        },
        "counts": {
            "source_rows": source_row_count,
            "processed_rows": processed_row_count,
            "indexed_rows": indexed_row_count,
            "candidate_models": len(candidate_models),
        },
        "row_filtering": {
            "missing_summary": {
                "skipped_rows": missing_summary_skip_count,
            }
        },
        "config": _training_config_to_metadata_dict(config),
        "router_behavior": dict(router_behavior or {}),
        "analysis": {
            "status": analysis_status,
            "summary_source": (
                "DifficultyAnalysisAgent" if analysis_status == "completed" else "query"
            ),
            "analysis_model": config.analysis_model,
            "analysis_prompt_version": config.analysis_prompt_version,
            "analysis_embedding_model": config.analysis_embedding_model,
        },
    }


def _training_config_to_metadata_dict(
    config: DifficultyAwareTrainingConfig,
) -> dict[str, Any]:
    return {
        "dry_run": config.dry_run,
        "limit": config.limit,
        "analysis_prompt_version": config.analysis_prompt_version,
        "analysis_model": config.analysis_model,
        "analysis_embedding_model": config.analysis_embedding_model,
        "cache_mode": config.cache_mode,
        "request_options": config.request_options,
    }


def build_model_collection_names(candidate_models: list[str]) -> dict[str, str]:
    return {
        model_name: (
            f"{MODEL_COLLECTION_PREFIX}{_truncate_collection_token(model_name)}"
            f"__{_collection_suffix(model_name)}"
        )
        for model_name in candidate_models
    }


def normalize_candidate_models(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("candidate_models must be a non-empty list of strings.")

    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("candidate_models must contain only non-empty strings.")
        normalized.append(item.strip())
    return normalized


def _extract_model_name(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return None
    model_name = metadata.get("raw_model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        return None
    return model_name.strip()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sanitize_collection_token(value: str) -> str:
    collapsed = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return collapsed or "unknown_model"


def _truncate_collection_token(value: str) -> str:
    max_token_length = 63 - len(MODEL_COLLECTION_PREFIX) - len("__") - 8
    sanitized = _sanitize_collection_token(value)
    truncated = sanitized[:max_token_length].rstrip("_")
    return truncated or "unknown_model"


def _collection_suffix(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def _normalize_non_negative_int(value: object, *, name: str) -> int:
    normalized = _normalize_int(value, name=name)
    if normalized < 0:
        raise ValueError(f"config.{name} must be a non-negative integer.")
    return normalized


def _normalize_positive_int(value: object, *, name: str) -> int:
    normalized = _normalize_int(value, name=name)
    if normalized <= 0:
        raise ValueError(f"config.{name} must be a positive integer.")
    return normalized


def _normalize_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"config.{name} must be an integer.")
    return value


def _normalize_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"config.{name} must be a boolean.")
    return value


def _normalize_non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config.{name} must be a non-empty string.")
    return value.strip()


def _normalize_analysis_prompt_version(value: object) -> str:
    normalized = _normalize_non_empty_string(
        value,
        name="analysis_prompt_version",
    )
    if normalized not in PROMPT_REGISTRY:
        raise ValueError(
            "config.analysis_prompt_version must be one of: "
            f"{', '.join(sorted(PROMPT_REGISTRY))}."
        )
    return normalized


def _normalize_cache_mode(value: object, *, name: str) -> CacheMode:
    normalized = _normalize_non_empty_string(value, name=name)
    if normalized not in SUPPORTED_CACHE_MODES:
        raise ValueError(
            f"config.{name} must be one of: {', '.join(SUPPORTED_CACHE_MODES)}."
        )
    return cast(CacheMode, normalized)


def _normalize_cache_mode_config(
    value: object, *, name: str
) -> tuple[CacheMode | dict[str, CacheMode], CacheMode, CacheMode]:
    if not isinstance(value, dict):
        normalized = _normalize_cache_mode(value, name=name)
        return normalized, normalized, normalized

    expected_keys = {"chat_completions", "embeddings"}
    actual_keys = set(value.keys())
    missing_keys = sorted(expected_keys - actual_keys)
    extra_keys = sorted(actual_keys - expected_keys)
    if missing_keys or extra_keys:
        problems: list[str] = []
        if missing_keys:
            problems.append(f"missing keys: {', '.join(missing_keys)}")
        if extra_keys:
            problems.append(f"unexpected keys: {', '.join(extra_keys)}")
        raise ValueError(
            f"config.{name} must define exactly chat_completions and embeddings ({'; '.join(problems)})."
        )

    chat_cache_mode = _normalize_cache_mode(
        value.get("chat_completions"), name=f"{name}.chat_completions"
    )
    embedding_cache_mode = _normalize_cache_mode(
        value.get("embeddings"), name=f"{name}.embeddings"
    )
    normalized_config: dict[str, CacheMode] = {
        "chat_completions": chat_cache_mode,
        "embeddings": embedding_cache_mode,
    }
    return normalized_config, chat_cache_mode, embedding_cache_mode


def _normalize_request_options_config(
    value: object, *, name: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError(f"config.{name} must be an object.")

    if not value:
        request_options = {"chat_completions": {}, "embeddings": {}}
        return request_options, {}, {}

    expected_keys = {"chat_completions", "embeddings"}
    actual_keys = set(value.keys())
    missing_keys = sorted(expected_keys - actual_keys)
    extra_keys = sorted(actual_keys - expected_keys)
    if missing_keys or extra_keys:
        problems: list[str] = []
        if missing_keys:
            problems.append(f"missing keys: {', '.join(missing_keys)}")
        if extra_keys:
            problems.append(f"unexpected keys: {', '.join(extra_keys)}")
        raise ValueError(
            f"config.{name} must define exactly chat_completions and embeddings ({'; '.join(problems)})."
        )

    chat_request_options = _normalize_request_options_endpoint(
        value.get("chat_completions"),
        name=f"{name}.chat_completions",
    )
    embedding_request_options = _normalize_request_options_endpoint(
        value.get("embeddings"),
        name=f"{name}.embeddings",
    )
    request_options = {
        "chat_completions": chat_request_options,
        "embeddings": embedding_request_options,
    }
    return request_options, chat_request_options, embedding_request_options


def _normalize_request_options_endpoint(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"config.{name} must be an object.")
    return dict(value)


def _load_model_pricing_by_raw_name(
    manifest_path: Path,
) -> dict[str, dict[str, float]]:
    stage1_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(stage1_manifest, dict):
        return {}
    config_payload = stage1_manifest.get("config")
    if not isinstance(config_payload, dict):
        return {}
    source_normalization = config_payload.get("source_normalization")
    if not isinstance(source_normalization, dict):
        return {}
    raw_llm_config_path = source_normalization.get("llm_config_path")
    if not isinstance(raw_llm_config_path, str) or not raw_llm_config_path.strip():
        return {}

    llm_config_path = _resolve_stage1_llm_config_path(raw_llm_config_path)
    llm_config = load_llm_config(llm_config_path)
    pricing_by_model: dict[str, dict[str, float]] = {}
    for raw_model_name in llm_config.model_name_mapping:
        normalized_model_name = normalize_model_name(
            raw_model_name,
            llm_config.model_name_mapping,
        )
        pricing = llm_config.model_pricing.get(normalized_model_name)
        if pricing is None and llm_config.pricing_fallback_model_name is not None:
            pricing = llm_config.model_pricing.get(
                llm_config.pricing_fallback_model_name
            )
        if pricing is None:
            continue
        pricing_by_model[raw_model_name] = {
            "input_cost_per_token": pricing.input_cost_per_token,
            "output_cost_per_token": pricing.output_cost_per_token,
        }
    return pricing_by_model


def _resolve_stage1_llm_config_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[2] / candidate


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Stage 1 row field {field_name} must be a non-empty string.")
    return value.strip()


def _require_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Stage 1 row field {field_name} must be numeric.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"Stage 1 row field {field_name} must be finite.")
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"Stage 1 row field {field_name} must be between 0.0 and 1.0.")
    return normalized


def _require_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Stage 1 row field {field_name} must be an integer.")
    return value

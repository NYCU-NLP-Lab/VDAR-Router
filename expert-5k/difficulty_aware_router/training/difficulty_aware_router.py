from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from tqdm import tqdm

from cache import CacheDeferredRequest, CacheOpenAI
from difficulty_aware_router.agents import DifficultyAnalysisAgent
from difficulty_aware_router.difficulty_aware_router import (
    DEFAULT_TOP_K,
    DifficultyAwareRouter,
    NoRewardRankingRouter,
    QueryEmbeddingRouter,
)
from experiment.training.base import RouterTrainer, UnifiedRouterBase
from experiment.training.contracts import TrainArtifactRef, TrainingManifest

from .artifact_builder import (
    build_training_artifacts,
    extract_candidate_models,
    load_stage1_rows,
    materialize_training_artifacts,
    normalize_candidate_models,
    normalize_training_config,
    prepare_training_row,
)

TRAINER_PATH = "difficulty_aware_router.training:DifficultyAwareTrainer"
QUERY_EMBEDDING_TRAINER_PATH = "difficulty_aware_router.training:QueryEmbeddingTrainer"
NO_REWARD_RANKING_TRAINER_PATH = (
    "difficulty_aware_router.training:NoRewardRankingTrainer"
)


class _BaseDifficultyAwareTrainer(RouterTrainer):
    router_family = "difficulty-aware-router"
    trainer_path = TRAINER_PATH
    retrieval_text_source = "difficulty_summary"
    ranking_strategy = "reward_ranking"

    def should_write_artifacts(self, config: dict[str, Any]) -> bool:
        return not normalize_training_config(config).dry_run

    def get_trainer_path(self) -> str:
        return self.trainer_path

    def internal_train(
        self,
        *,
        train_artifact: TrainArtifactRef,
        output_dir: Path,
        config: dict[str, Any],
        variant: str | None,
        runtime_overrides: dict[str, Any],
    ) -> None:
        del variant, runtime_overrides
        normalized_config = normalize_training_config(config)
        cache_dir = output_dir / "cache"
        if normalized_config.dry_run:
            rows = load_stage1_rows(train_artifact.train_path)
            extract_candidate_models(rows)
            rows_to_process = _limit_rows(rows, normalized_config.limit)
            for row in rows_to_process:
                prepare_training_row(
                    row=row,
                    dataset_id=train_artifact.dataset_id,
                    dataset_version=train_artifact.dataset_version,
                )
            return

        artifact_bundle = build_training_artifacts(
            train_artifact=train_artifact,
            output_dir=output_dir,
            config=normalized_config,
        )
        rows_to_process = _limit_rows(artifact_bundle.rows, normalized_config.limit)
        analysis_agent = None
        if self._requires_analysis_for_indexing():
            analysis_agent = DifficultyAnalysisAgent(
                model=normalized_config.analysis_model,
                prompt_version=normalized_config.analysis_prompt_version,
                cache_dir=cache_dir,
                cache_mode=normalized_config.chat_cache_mode,
            )
        embedding_client = CacheOpenAI(
            cache_dir=cache_dir,
            cache_mode=normalized_config.embedding_cache_mode,
        )
        deferred_request: CacheDeferredRequest | None = None
        buffered_record_payloads: list[tuple[str, dict[str, Any]]] = []
        evaluated_rows: list[tuple[dict[str, Any], str]] = []
        missing_summary_skip_count = 0

        for row in rows_to_process:
            prepared_row = prepare_training_row(
                row=row,
                dataset_id=train_artifact.dataset_id,
                dataset_version=train_artifact.dataset_version,
            )

            if analysis_agent is None:
                summary = ""
            else:
                # Step1: Create difficulty analysis
                try:
                    analysis = analysis_agent.invoke(
                        question=prepared_row["question"],
                        request_options=normalized_config.chat_request_options,
                    )
                except CacheDeferredRequest as exc:
                    if deferred_request is None:
                        deferred_request = exc
                    continue
                summary = analysis.summary.strip()
            if not summary and self._requires_summary_for_indexing():
                missing_summary_skip_count += 1
                continue
            evaluated_rows.append((prepared_row, summary))

        for prepared_row, summary in evaluated_rows:
            retrieval_text = self._build_retrieval_text(
                prepared_row=prepared_row,
                summary=summary,
            ).strip()
            if not retrieval_text:
                raise ValueError(
                    "Difficulty-aware training retrieval text must not be empty."
                )
            # Step2: Create embedding of the analysis
            try:
                embedding_response = embedding_client.embeddings.create(
                    model=normalized_config.analysis_embedding_model,
                    input=retrieval_text,
                    **normalized_config.embedding_request_options,
                )
            except CacheDeferredRequest as exc:
                if deferred_request is None:
                    deferred_request = exc
                continue
            embedding = _extract_embedding(embedding_response)

            # Step3: Put the embedding into the database
            record_payload = {
                "ids": [prepared_row["record_id"]],
                "documents": [retrieval_text],
                "embeddings": [embedding],
                "metadatas": [prepared_row["metadata"]],
            }
            buffered_record_payloads.append(
                (prepared_row["model_name"], record_payload)
            )

        if deferred_request is not None:
            raise deferred_request

        if not evaluated_rows:
            raise ValueError(
                "Difficulty-aware training has no usable summaries after skipping empty analysis summaries."
            )

        chroma_client = PersistentClient(path=str(artifact_bundle.chroma_dir))
        shared_collection = chroma_client.get_or_create_collection(
            name=artifact_bundle.shared_collection_name
        )
        model_collections = {
            model_name: chroma_client.get_or_create_collection(name=collection_name)
            for model_name, collection_name in artifact_bundle.model_collection_names.items()
        }
        for model_name, record_payload in tqdm(
            buffered_record_payloads,
            total=len(buffered_record_payloads),
            desc="Upserting Chroma records",
            unit="record",
        ):
            shared_collection.upsert(**record_payload)
            model_collections[model_name].upsert(**record_payload)

        materialize_training_artifacts(
            artifact_bundle=artifact_bundle,
            train_artifact=train_artifact,
            config=normalized_config,
            processed_row_count=len(rows_to_process),
            indexed_row_count=len(buffered_record_payloads),
            missing_summary_skip_count=missing_summary_skip_count,
            analysis_status=(
                "completed" if self._requires_analysis_for_indexing() else "skipped"
            ),
            router_behavior={
                "retrieval_text_source": self.retrieval_text_source,
                "ranking_strategy": self.ranking_strategy,
            },
        )

        required_paths = (
            artifact_bundle.router_artifact_path,
            artifact_bundle.candidate_models_path,
            artifact_bundle.model_cost_stats_path,
            artifact_bundle.training_metadata_path,
            artifact_bundle.chroma_dir,
        )
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise ValueError(
                f"Difficulty-aware training did not produce expected artifacts: {missing_paths}"
            )

    def load_router(
        self,
        training_manifest: TrainingManifest,
        runtime_config: dict[str, Any] | None = None,
    ) -> UnifiedRouterBase:
        artifact_root = training_manifest.artifact_path
        router_payload = _load_json_payload(artifact_root / "router.json")
        candidate_models = normalize_candidate_models(
            router_payload.get("candidate_models")
        )

        chroma_dir_name = _require_non_empty_string(
            router_payload.get("chroma_dir"), field_name="router.json.chroma_dir"
        )
        chroma_path = artifact_root / chroma_dir_name
        if not chroma_path.exists() or not chroma_path.is_dir():
            raise ValueError(
                f"Difficulty-aware runtime chroma directory is missing: {chroma_path}"
            )

        artifact_manifest_name = _require_non_empty_string(
            router_payload.get("artifact_manifest"),
            field_name="router.json.artifact_manifest",
        )
        artifact_manifest = _load_json_payload(artifact_root / artifact_manifest_name)
        config_payload = artifact_manifest.get("config")
        if config_payload is None:
            normalized_config = normalize_training_config({})
        else:
            if not isinstance(config_payload, dict):
                raise ValueError(
                    "difficulty_aware_artifact_manifest.config must be an object."
                )
            normalized_config = normalize_training_config(config_payload)
        normalized_runtime_config = _normalize_runtime_config(runtime_config)

        raw_model_cost_stats_name = router_payload.get("model_cost_stats_path")
        if raw_model_cost_stats_name is None:
            model_cost_stats = {}
        else:
            model_cost_stats_name = _require_non_empty_string(
                raw_model_cost_stats_name,
                field_name="router.json.model_cost_stats_path",
            )
            model_cost_stats = _load_model_cost_stats(
                artifact_root / model_cost_stats_name
            )

        collections = router_payload.get("collections")
        if not isinstance(collections, dict):
            raise ValueError("router.json.collections must be an object.")

        shared_collection_name = _require_non_empty_string(
            collections.get("shared"), field_name="router.json.collections.shared"
        )
        model_collection_names = _normalize_model_collection_names(
            collections.get("by_model")
        )

        analysis_model = normalized_runtime_config.get("analysis_model")
        analysis_prompt_version = normalized_runtime_config.get(
            "analysis_prompt_version",
            normalized_config.analysis_prompt_version,
        )
        embedding_model = normalized_runtime_config.get("embedding_model")
        cache_mode = normalized_runtime_config.get(
            "cache_mode", normalized_config.cache_mode
        )
        request_options = normalized_runtime_config.get(
            "request_options", normalized_config.request_options
        )
        effective_top_k = normalized_runtime_config.get("top_k", DEFAULT_TOP_K)
        gamma = normalized_runtime_config.get("gamma", 0.8)
        cost_normalization_scale = normalized_runtime_config.get(
            "cost_normalization_scale", 0.0
        )
        router_kwargs: dict[str, Any] = {
            "candidate_models": candidate_models,
            "chroma_path": chroma_path,
            "model_collection_names": model_collection_names,
            "shared_collection_name": shared_collection_name,
            "analysis_model": analysis_model,
            "analysis_prompt_version": analysis_prompt_version,
            "embedding_model": embedding_model,
            "cache_mode": cache_mode,
            "request_options": request_options,
            "model_cost_stats": model_cost_stats,
            "gamma": gamma,
            "cost_normalization_scale": cost_normalization_scale,
        }
        if "cache_dir" in normalized_runtime_config:
            router_kwargs["cache_dir"] = normalized_runtime_config["cache_dir"]
        if "top_k" in normalized_runtime_config:
            router_kwargs["top_k"] = effective_top_k

        return self._get_router_class()(
            **router_kwargs,
        )

    def _requires_summary_for_indexing(self) -> bool:
        return True

    def _requires_analysis_for_indexing(self) -> bool:
        return True

    def _build_retrieval_text(
        self, *, prepared_row: dict[str, Any], summary: str
    ) -> str:
        del prepared_row
        return summary

    def _get_router_class(self) -> type[DifficultyAwareRouter]:
        return DifficultyAwareRouter


class DifficultyAwareTrainer(_BaseDifficultyAwareTrainer):
    router_family = "difficulty-aware-router"
    trainer_path = TRAINER_PATH
    retrieval_text_source = "difficulty_summary"
    ranking_strategy = "reward_ranking"


class QueryEmbeddingTrainer(_BaseDifficultyAwareTrainer):
    router_family = "difficulty-aware-query-embedding"
    trainer_path = QUERY_EMBEDDING_TRAINER_PATH
    retrieval_text_source = "query"
    ranking_strategy = "reward_ranking"

    def _requires_summary_for_indexing(self) -> bool:
        return False

    def _requires_analysis_for_indexing(self) -> bool:
        return False

    def _build_retrieval_text(
        self, *, prepared_row: dict[str, Any], summary: str
    ) -> str:
        del summary
        return str(prepared_row["question"])

    def _get_router_class(self) -> type[DifficultyAwareRouter]:
        return QueryEmbeddingRouter


class NoRewardRankingTrainer(_BaseDifficultyAwareTrainer):
    router_family = "difficulty-aware-no-reward-ranking"
    trainer_path = NO_REWARD_RANKING_TRAINER_PATH
    retrieval_text_source = "difficulty_summary"
    ranking_strategy = "direct_best_performance"

    def _get_router_class(self) -> type[DifficultyAwareRouter]:
        return NoRewardRankingRouter


def _limit_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return list(rows)
    return list(rows[:limit])


def _extract_embedding(response: object) -> list[float]:
    data = getattr(response, "data", None)
    if not isinstance(data, list) or not data:
        raise ValueError("Embedding response is missing data entries.")

    first_item = data[0]
    embedding = getattr(first_item, "embedding", None)
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("Embedding response is missing a numeric embedding vector.")

    normalized_embedding: list[float] = []
    for value in embedding:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Embedding vectors must contain only numeric values.")
        normalized_embedding.append(float(value))
    return normalized_embedding


def _normalize_runtime_config(
    value: dict[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("difficulty-aware runtime_config must be an object.")

    allowed_keys = {
        "alpha",
        "gamma",
        "cost_normalization_scale",
        "analysis_model",
        "analysis_prompt_version",
        "embedding_model",
        "cache_dir",
        "cache_mode",
        "request_options",
        "top_k",
    }
    unexpected_keys = sorted(set(value) - allowed_keys)
    if unexpected_keys:
        raise ValueError(
            "difficulty-aware runtime_config contains unsupported keys: "
            f"{unexpected_keys}"
        )

    normalized: dict[str, Any] = {}
    if "alpha" in value:
        raise ValueError(
            "runtime_config.alpha is no longer supported; use runtime_config.gamma instead."
        )
    if "gamma" in value:
        gamma = _require_float(value.get("gamma"), field_name="runtime_config.gamma")
        if gamma < 0.0 or gamma > 1.0:
            raise ValueError("runtime_config.gamma must be between 0.0 and 1.0.")
        normalized["gamma"] = gamma
    if "cost_normalization_scale" in value:
        normalized["cost_normalization_scale"] = _require_non_negative_float(
            value.get("cost_normalization_scale"),
            field_name="runtime_config.cost_normalization_scale",
        )
    if "analysis_model" in value:
        normalized["analysis_model"] = _require_non_empty_string(
            value.get("analysis_model"), field_name="runtime_config.analysis_model"
        )
    if "analysis_prompt_version" in value:
        normalized["analysis_prompt_version"] = _require_non_empty_string(
            value.get("analysis_prompt_version"),
            field_name="runtime_config.analysis_prompt_version",
        )
    if "embedding_model" in value:
        normalized["embedding_model"] = _require_non_empty_string(
            value.get("embedding_model"),
            field_name="runtime_config.embedding_model",
        )
    if "cache_dir" in value:
        cache_dir = value.get("cache_dir")
        if not isinstance(cache_dir, (str, Path)) or (
            isinstance(cache_dir, str) and not cache_dir.strip()
        ):
            raise ValueError(
                "runtime_config.cache_dir must be a non-empty string or Path."
            )
        normalized["cache_dir"] = cache_dir
    if "cache_mode" in value:
        normalized["cache_mode"] = value.get("cache_mode")
    if "request_options" in value:
        normalized["request_options"] = value.get("request_options")
    if "top_k" in value:
        normalized["top_k"] = value.get("top_k")
    return normalized


def _load_json_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Difficulty-aware runtime artifact is missing: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Difficulty-aware runtime artifact must be a JSON object: {path}"
        )
    return payload


def _load_model_cost_stats(path: Path) -> dict[str, dict[str, float]]:
    payload = _load_json_payload(path)
    normalized: dict[str, dict[str, float]] = {}
    for model_name, stats in payload.items():
        if not isinstance(stats, dict):
            raise ValueError(
                f"model_cost_stats entries must be JSON objects for model: {model_name}"
            )
        normalized[
            _require_non_empty_string(model_name, field_name="model_cost_stats key")
        ] = {
            key: _require_float(
                value,
                field_name=f"model_cost_stats[{model_name!r}].{key}",
            )
            for key, value in stats.items()
        }
    return normalized


def _normalize_model_collection_names(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError("router.json.collections.by_model must be a non-empty object.")

    normalized: dict[str, str] = {}
    for model_name, collection_name in value.items():
        normalized[
            _require_non_empty_string(
                model_name,
                field_name="router.json.collections.by_model key",
            )
        ] = _require_non_empty_string(
            collection_name,
            field_name="router.json.collections.by_model value",
        )
    return normalized


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _require_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")
    return float(value)


def _require_non_negative_float(value: object, *, field_name: str) -> float:
    normalized = _require_float(value, field_name=field_name)
    if not normalized >= 0.0:
        raise ValueError(f"{field_name} must be non-negative.")
    return normalized

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from chromadb import PersistentClient

from cache import CacheOpenAI
from cache._constants import SUPPORTED_CACHE_MODES
from cache._contracts import CacheMode
from difficulty_aware_router.agents import DifficultyAnalysisAgent
from difficulty_aware_router.agents.settings import get_analysis_settings
from experiment.training.base import RankedModel, RouterResult, UnifiedRouterBase

DEFAULT_SHARED_COLLECTION_NAME = "difficulty_aware_shared"
DEFAULT_TOP_K = 3
PREFERRED_LABEL = "Preffered"
TIE_LABEL = "Tie"
NOT_PREFERRED_LABEL = "Not Preffered"


class DifficultyAwareRouter(UnifiedRouterBase):
    retrieval_text_source = "difficulty_summary"
    ranking_strategy = "reward_ranking"
    strategy_name = "difficulty_aware_retrieval"

    def __init__(
        self,
        *,
        candidate_models: list[str],
        chroma_path: str | Path,
        model_collection_names: dict[str, str],
        shared_collection_name: str | None = None,
        analysis_model: str | None = None,
        analysis_prompt_version: str = "v1",
        embedding_model: str | None = None,
        cache_dir: str | Path | None = None,
        cache_mode: CacheMode | dict[str, CacheMode] = "record",
        request_options: dict[str, dict[str, Any]] | None = None,
        top_k: int = DEFAULT_TOP_K,
        seed: int | None = None,
        model_cost_stats: dict[str, dict[str, float]] | None = None,
        gamma: float = 0.8,
        cost_normalization_scale: float = 0.0,
    ) -> None:
        del seed

        if not candidate_models:
            raise ValueError("candidate_models must not be empty.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
            raise ValueError("gamma must be a number between 0.0 and 1.0.")
        normalized_gamma = float(gamma)
        if (
            not math.isfinite(normalized_gamma)
            or normalized_gamma < 0.0
            or normalized_gamma > 1.0
        ):
            raise ValueError("gamma must be a number between 0.0 and 1.0.")
        if isinstance(cost_normalization_scale, bool) or not isinstance(
            cost_normalization_scale, (int, float)
        ):
            raise ValueError("cost_normalization_scale must be a non-negative number.")
        normalized_cost_normalization_scale = float(cost_normalization_scale)
        if (
            not math.isfinite(normalized_cost_normalization_scale)
            or normalized_cost_normalization_scale < 0.0
        ):
            raise ValueError("cost_normalization_scale must be a non-negative number.")

        normalized_models = _normalize_candidate_models(candidate_models)
        normalized_collection_names = _normalize_collection_names(
            model_collection_names
        )
        missing_collections = [
            model_name
            for model_name in normalized_models
            if model_name not in normalized_collection_names
        ]
        if missing_collections:
            raise ValueError(
                "model_collection_names is missing mappings for candidate models: "
                f"{missing_collections}"
            )

        resolved_chroma_path = Path(chroma_path)
        if not resolved_chroma_path.exists() or not resolved_chroma_path.is_dir():
            raise ValueError(
                f"chroma_path must point to an existing directory: {resolved_chroma_path}"
            )

        settings = get_analysis_settings()
        (
            self.cache_mode,
            self.chat_cache_mode,
            self.embedding_cache_mode,
        ) = _normalize_cache_mode_config(cache_mode)
        (
            self.request_options,
            self.chat_request_options,
            self.embedding_request_options,
        ) = _normalize_request_options_config(request_options)
        resolved_shared_collection_name = _require_non_empty_string(
            shared_collection_name or DEFAULT_SHARED_COLLECTION_NAME,
            field_name="shared_collection_name",
        )

        self.candidate_models = normalized_models
        self.chroma_path = resolved_chroma_path
        self.model_collection_names = normalized_collection_names
        self.shared_collection_name = resolved_shared_collection_name
        self.analysis_model = analysis_model or settings.llm_analysis_model
        self.analysis_prompt_version = _require_non_empty_string(
            analysis_prompt_version,
            field_name="analysis_prompt_version",
        )
        self.embedding_model = embedding_model or settings.llm_analysis_embedding_model
        self.cache_dir = (
            _resolve_cache_dir(cache_dir)
            if cache_dir is not None
            else self.chroma_path.parent / "cache"
        )
        self.top_k = top_k
        self.gamma = normalized_gamma
        self.cost_normalization_scale = normalized_cost_normalization_scale
        self.model_cost_stats = dict(model_cost_stats or {})
        self.analysis_agent = DifficultyAnalysisAgent(
            model=self.analysis_model,
            prompt_version=self.analysis_prompt_version,
            cache_dir=self.cache_dir,
            cache_mode=self.chat_cache_mode,
        )
        self.embedding_client = (
            self.analysis_agent.client
            if self.embedding_cache_mode == self.chat_cache_mode
            else CacheOpenAI(
                cache_dir=self.cache_dir,
                cache_mode=self.embedding_cache_mode,
            )
        )
        self.chroma_client = PersistentClient(path=str(self.chroma_path))
        self.shared_collection = self.chroma_client.get_collection(
            self.shared_collection_name
        )
        self.model_collections = {
            model_name: self.chroma_client.get_collection(
                self.model_collection_names[model_name]
            )
            for model_name in self.candidate_models
        }

        super().__init__(model=None, yaml_path=None, resources=self.candidate_models)

    def route_single_ranked(self, query_input: dict[str, Any]) -> RouterResult:
        query = _extract_query(query_input)
        analysis_result = self.analysis_agent.invoke(
            question=query,
            request_options=self.chat_request_options,
        )
        query_summary = analysis_result.summary.strip()
        if not query_summary and self._requires_query_summary():
            raise ValueError("DifficultyAnalysisAgent returned an empty summary.")

        retrieval_text = self._build_retrieval_text(
            query=query,
            analysis_result=analysis_result,
            query_summary=query_summary,
        ).strip()
        if not retrieval_text:
            raise ValueError("Difficulty-aware retrieval text must not be empty.")
        embedding_response = self.embedding_client.embeddings.create(
            model=self.embedding_model,
            input=retrieval_text,
            **self.embedding_request_options,
        )
        query_embedding = _extract_embedding(embedding_response)
        input_token_count = analysis_result.input_token_count
        if input_token_count is None:
            input_token_count = _estimate_input_tokens(query)
        ranking_payload = self._build_ranking_payload(
            query_embedding=query_embedding,
            input_token_count=input_token_count,
        )
        return RouterResult(
            ranked_models=ranking_payload["ranked_models"],
            metadata={
                "strategy": self.strategy_name,
                "candidate_count": len(self.candidate_models),
                "top_k": self.top_k,
                "analysis_text": analysis_result.response_text,
                "query_summary": query_summary,
                "retrieval_text_source": self.retrieval_text_source,
                "ranking_strategy": self.ranking_strategy,
                "analysis_model": self.analysis_model,
                "analysis_prompt_version": self.analysis_prompt_version,
                "embedding_model": self.embedding_model,
                "gamma": self.gamma,
                "cost_normalization_scale": self.cost_normalization_scale,
                "input_token_count": input_token_count,
                "model_rewards": ranking_payload["reward_by_model"],
                "model_estimated_costs": ranking_payload["estimated_cost_by_model"],
                "model_normalized_costs": ranking_payload["normalized_cost_by_model"],
                "model_scores": ranking_payload["score_by_model"],
                "model_neighbor_counts": ranking_payload["neighbor_counts"],
                "retrieved_evidence": ranking_payload["retrieved_evidence_by_model"],
            },
        )

    def _requires_query_summary(self) -> bool:
        return True

    def _build_retrieval_text(
        self,
        *,
        query: str,
        analysis_result: Any,
        query_summary: str,
    ) -> str:
        del analysis_result, query
        return query_summary

    def _build_ranking_payload(
        self,
        *,
        query_embedding: list[float],
        input_token_count: int,
    ) -> dict[str, Any]:
        reward_by_model: dict[str, float] = {}
        estimated_cost_by_model: dict[str, float] = {}
        normalized_cost_by_model: dict[str, float] = {}
        score_by_model: dict[str, float] = {}
        neighbor_counts: dict[str, int] = {}
        retrieved_evidence_by_model: dict[str, list[dict[str, Any]]] = {}
        for model_name in self.candidate_models:
            query_result = self.model_collections[model_name].query(
                query_embeddings=[query_embedding],
                n_results=self.top_k,
                include=["distances", "documents", "metadatas"],
            )
            reward, neighbor_count = _compute_model_reward(
                query_result=query_result,
                top_k=self.top_k,
            )
            reward_by_model[model_name] = reward
            estimated_cost = _estimate_model_cost(
                model_stats=self.model_cost_stats.get(model_name),
                input_token_count=input_token_count,
            )
            estimated_cost_by_model[model_name] = estimated_cost
            normalized_cost = _normalize_estimated_cost(
                estimated_cost=estimated_cost,
                cost_normalization_scale=self.cost_normalization_scale,
            )
            normalized_cost_by_model[model_name] = normalized_cost
            if self.cost_normalization_scale <= 0.0:
                score_by_model[model_name] = reward
            else:
                score_by_model[model_name] = (
                    self.gamma * reward - (1.0 - self.gamma) * normalized_cost
                )
            neighbor_counts[model_name] = neighbor_count
            retrieved_evidence_by_model[model_name] = _build_retrieved_evidence(
                query_result=query_result
            )
        return _build_ranked_payload(
            candidate_models=self.candidate_models,
            reward_by_model=reward_by_model,
            estimated_cost_by_model=estimated_cost_by_model,
            normalized_cost_by_model=normalized_cost_by_model,
            score_by_model=score_by_model,
            neighbor_counts=neighbor_counts,
            retrieved_evidence_by_model=retrieved_evidence_by_model,
        )


class QueryEmbeddingRouter(DifficultyAwareRouter):
    retrieval_text_source = "query"
    strategy_name = "difficulty_aware_query_embedding_retrieval"

    def route_single_ranked(self, query_input: dict[str, Any]) -> RouterResult:
        query = _extract_query(query_input)
        embedding_response = self.embedding_client.embeddings.create(
            model=self.embedding_model,
            input=query,
            **self.embedding_request_options,
        )
        query_embedding = _extract_embedding(embedding_response)
        input_token_count = _estimate_input_tokens(query)
        ranking_payload = self._build_ranking_payload(
            query_embedding=query_embedding,
            input_token_count=input_token_count,
        )
        return RouterResult(
            ranked_models=ranking_payload["ranked_models"],
            metadata={
                "strategy": self.strategy_name,
                "candidate_count": len(self.candidate_models),
                "top_k": self.top_k,
                "analysis_text": None,
                "query_summary": None,
                "retrieval_text_source": self.retrieval_text_source,
                "ranking_strategy": self.ranking_strategy,
                "analysis_model": None,
                "analysis_prompt_version": None,
                "embedding_model": self.embedding_model,
                "gamma": self.gamma,
                "cost_normalization_scale": self.cost_normalization_scale,
                "input_token_count": input_token_count,
                "model_rewards": ranking_payload["reward_by_model"],
                "model_estimated_costs": ranking_payload["estimated_cost_by_model"],
                "model_normalized_costs": ranking_payload["normalized_cost_by_model"],
                "model_scores": ranking_payload["score_by_model"],
                "model_neighbor_counts": ranking_payload["neighbor_counts"],
                "retrieved_evidence": ranking_payload["retrieved_evidence_by_model"],
            },
        )


class NoRewardRankingRouter(DifficultyAwareRouter):
    ranking_strategy = "direct_best_performance"
    strategy_name = "difficulty_aware_no_reward_ranking"

    def _build_ranking_payload(
        self,
        *,
        query_embedding: list[float],
        input_token_count: int,
    ) -> dict[str, Any]:
        query_result = self.shared_collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["distances", "documents", "metadatas"],
        )
        score_by_model, neighbor_counts, retrieved_evidence_by_model = (
            _compute_direct_model_scores(
                query_result=query_result,
                candidate_models=self.candidate_models,
                top_k=self.top_k,
            )
        )
        estimated_cost_by_model: dict[str, float] = {}
        normalized_cost_by_model: dict[str, float] = {}
        for model_name in self.candidate_models:
            estimated_cost = _estimate_model_cost(
                model_stats=self.model_cost_stats.get(model_name),
                input_token_count=input_token_count,
            )
            estimated_cost_by_model[model_name] = estimated_cost
            normalized_cost_by_model[model_name] = _normalize_estimated_cost(
                estimated_cost=estimated_cost,
                cost_normalization_scale=self.cost_normalization_scale,
            )
        return _build_ranked_payload(
            candidate_models=self.candidate_models,
            reward_by_model={},
            estimated_cost_by_model=estimated_cost_by_model,
            normalized_cost_by_model=normalized_cost_by_model,
            score_by_model=score_by_model,
            neighbor_counts=neighbor_counts,
            retrieved_evidence_by_model=retrieved_evidence_by_model,
        )


def _extract_query(query_input: dict[str, Any]) -> str:
    if not isinstance(query_input, dict):
        raise ValueError(
            "DifficultyAwareRouter expects query_input to be a mapping with a non-empty 'query' string."
        )

    query = query_input.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "DifficultyAwareRouter expects query_input['query'] to be a non-empty string."
        )
    return query.strip()


def _normalize_candidate_models(candidate_models: list[str]) -> list[str]:
    normalized_models: list[str] = []
    for model_name in candidate_models:
        normalized_models.append(
            _require_non_empty_string(model_name, field_name="candidate_models[]")
        )
    return normalized_models


def _normalize_collection_names(value: dict[str, str]) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(
            "model_collection_names must be a non-empty mapping of model name to collection name."
        )

    normalized: dict[str, str] = {}
    for model_name, collection_name in value.items():
        normalized[
            _require_non_empty_string(
                model_name, field_name="model_collection_names key"
            )
        ] = _require_non_empty_string(
            collection_name,
            field_name="model_collection_names value",
        )
    return normalized


def _normalize_cache_mode(value: object, *, field_name: str) -> CacheMode:
    normalized = _require_non_empty_string(value, field_name=field_name)
    if normalized not in SUPPORTED_CACHE_MODES:
        raise ValueError(
            f"{field_name} must be one of: {', '.join(SUPPORTED_CACHE_MODES)}."
        )
    return normalized


def _normalize_cache_mode_config(
    value: object,
) -> tuple[CacheMode | dict[str, CacheMode], CacheMode, CacheMode]:
    if not isinstance(value, dict):
        normalized = _normalize_cache_mode(value, field_name="cache_mode")
        return normalized, normalized, normalized

    if not value:
        return "record", "record", "record"

    expected_keys = {"chat_completions", "embeddings"}
    actual_keys = set(value.keys())
    if actual_keys != expected_keys:
        raise ValueError(
            "cache_mode must define exactly chat_completions and embeddings."
        )

    chat_cache_mode = _normalize_cache_mode(
        value.get("chat_completions"),
        field_name="cache_mode.chat_completions",
    )
    embedding_cache_mode = _normalize_cache_mode(
        value.get("embeddings"),
        field_name="cache_mode.embeddings",
    )
    normalized_config: dict[str, CacheMode] = {
        "chat_completions": chat_cache_mode,
        "embeddings": embedding_cache_mode,
    }
    return normalized_config, chat_cache_mode, embedding_cache_mode


def _normalize_request_options_config(
    value: object,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if value is None:
        request_options = {"chat_completions": {}, "embeddings": {}}
        return request_options, {}, {}
    if not isinstance(value, dict):
        raise ValueError("request_options must be an object.")
    if not value:
        request_options = {"chat_completions": {}, "embeddings": {}}
        return request_options, {}, {}

    expected_keys = {"chat_completions", "embeddings"}
    actual_keys = set(value.keys())
    if actual_keys != expected_keys:
        raise ValueError(
            "request_options must define exactly chat_completions and embeddings."
        )

    chat_request_options = _normalize_request_options_endpoint(
        value.get("chat_completions"),
        field_name="request_options.chat_completions",
    )
    embedding_request_options = _normalize_request_options_endpoint(
        value.get("embeddings"),
        field_name="request_options.embeddings",
    )
    request_options = {
        "chat_completions": chat_request_options,
        "embeddings": embedding_request_options,
    }
    return request_options, chat_request_options, embedding_request_options


def _normalize_request_options_endpoint(
    value: object,
    *,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object.")
    return dict(value)


def _resolve_cache_dir(value: str | Path) -> Path:
    if isinstance(value, Path):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("cache_dir must be a non-empty string or Path.")
    return Path(value.strip())


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


def _compute_model_reward(
    *, query_result: dict[str, Any], top_k: int
) -> tuple[float, int]:
    distances = _extract_query_rows(query_result, key="distances")
    metadatas = _extract_query_rows(query_result, key="metadatas")
    if len(distances) != len(metadatas):
        raise ValueError(
            "Chroma query result distances and metadatas must have equal length."
        )

    contribution_sum = 0.0
    for distance, metadata in zip(distances, metadatas, strict=True):
        similarity = _distance_to_similarity(distance)
        performance_score = _performance_score(metadata)
        contribution_sum += similarity * performance_score
    return contribution_sum / float(top_k), len(distances)


def _compute_direct_model_scores(
    *, query_result: dict[str, Any], candidate_models: list[str], top_k: int
) -> tuple[dict[str, float], dict[str, int], dict[str, list[dict[str, Any]]]]:
    distances = _extract_query_rows(query_result, key="distances")
    metadatas = _extract_query_rows(query_result, key="metadatas")
    if len(distances) != len(metadatas):
        raise ValueError(
            "Chroma query result distances and metadatas must have equal length."
        )
    score_by_model = {model_name: 0.0 for model_name in candidate_models}
    neighbor_counts = {model_name: 0 for model_name in candidate_models}
    retrieved_evidence_by_model = {model_name: [] for model_name in candidate_models}
    evidence_rows = _build_retrieved_evidence(query_result)
    for distance, metadata, evidence_row in zip(
        distances, metadatas, evidence_rows, strict=True
    ):
        if not isinstance(metadata, dict):
            raise ValueError("Chroma query metadatas must contain object entries.")
        model_name = metadata.get("raw_model_name")
        if not isinstance(model_name, str) or model_name not in score_by_model:
            continue
        del distance
        score_by_model[model_name] += _performance_score(metadata)
        neighbor_counts[model_name] += 1
        retrieved_evidence_by_model[model_name].append(evidence_row)
    for model_name in candidate_models:
        count = neighbor_counts[model_name]
        score_by_model[model_name] = (
            score_by_model[model_name] / float(count) if count > 0 else 0.0
        )
    return score_by_model, neighbor_counts, retrieved_evidence_by_model


def _build_ranked_payload(
    *,
    candidate_models: list[str],
    reward_by_model: dict[str, float],
    estimated_cost_by_model: dict[str, float],
    normalized_cost_by_model: dict[str, float],
    score_by_model: dict[str, float],
    neighbor_counts: dict[str, int],
    retrieved_evidence_by_model: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    ranked_names = sorted(
        candidate_models,
        key=lambda model_name: (-score_by_model.get(model_name, 0.0), model_name),
    )
    ranked_models = [
        RankedModel(model_name=model_name, score=score_by_model.get(model_name, 0.0))
        for model_name in ranked_names
    ]
    return {
        "ranked_models": ranked_models,
        "reward_by_model": reward_by_model,
        "estimated_cost_by_model": estimated_cost_by_model,
        "normalized_cost_by_model": normalized_cost_by_model,
        "score_by_model": score_by_model,
        "neighbor_counts": neighbor_counts,
        "retrieved_evidence_by_model": retrieved_evidence_by_model,
    }


def _build_retrieved_evidence(query_result: dict[str, Any]) -> list[dict[str, Any]]:
    distances = _extract_query_rows(query_result, key="distances")
    metadatas = _extract_query_rows(query_result, key="metadatas")
    documents = _extract_optional_query_rows(query_result, key="documents")
    if len(distances) != len(metadatas):
        raise ValueError(
            "Chroma query result distances and metadatas must have equal length."
        )
    if documents is not None and len(distances) != len(documents):
        raise ValueError(
            "Chroma query result distances and documents must have equal length."
        )

    evidence_rows: list[dict[str, Any]] = []
    for rank, (distance, metadata, document) in enumerate(
        zip(
            distances,
            metadatas,
            documents if documents is not None else [None] * len(distances),
            strict=True,
        ),
        start=1,
    ):
        normalized_distance = _normalize_distance(distance)
        if not isinstance(metadata, dict):
            raise ValueError("Chroma query metadatas must contain object entries.")
        evidence_row: dict[str, Any] = {
            "rank": rank,
            "distance": normalized_distance,
            "similarity": _distance_to_similarity(normalized_distance),
        }
        for key in (
            "correctness",
            "prompt_id",
            "question",
            "raw_model_name",
            "row_id",
            "score",
            "input_token",
            "output_tokken",
        ):
            if key in metadata:
                evidence_row[key] = metadata[key]
        if isinstance(document, str) and document.strip():
            evidence_row["retrieved_document"] = document.strip()
        row_metadata_json = metadata.get("row_metadata_json")
        if isinstance(row_metadata_json, str) and row_metadata_json.strip():
            try:
                row_metadata = json.loads(row_metadata_json)
            except json.JSONDecodeError:
                evidence_row["row_metadata_json"] = row_metadata_json
            else:
                if isinstance(row_metadata, dict):
                    evidence_row["row_metadata"] = row_metadata
        evidence_rows.append(evidence_row)
    return evidence_rows


def _estimate_input_tokens(query: str) -> int:
    return len(query.split())


def _normalize_estimated_cost(
    *, estimated_cost: float, cost_normalization_scale: float
) -> float:
    if cost_normalization_scale <= 0.0:
        return 0.0
    return min(max(estimated_cost / cost_normalization_scale, 0.0), 1.0)


def _estimate_model_cost(
    *, model_stats: dict[str, float] | None, input_token_count: int
) -> float:
    if not isinstance(model_stats, dict):
        return 0.0
    input_cost_per_token = _coerce_non_negative_float(
        model_stats.get("input_cost_per_token")
    )
    output_cost_per_token = _coerce_non_negative_float(
        model_stats.get("output_cost_per_token")
    )
    avg_response_tokens = _coerce_non_negative_float(
        model_stats.get("avg_response_tokens")
    )
    return (
        float(input_token_count) * input_cost_per_token
        + avg_response_tokens * output_cost_per_token
    )


def _extract_query_rows(query_result: dict[str, Any], *, key: str) -> list[Any]:
    value = query_result.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Chroma query result is missing '{key}' rows.")
    if not value:
        return []

    first_query_rows = value[0]
    if not isinstance(first_query_rows, list):
        raise ValueError(f"Chroma query result '{key}' must be a nested list.")
    return first_query_rows


def _extract_optional_query_rows(
    query_result: dict[str, Any], *, key: str
) -> list[Any] | None:
    value = query_result.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"Chroma query result '{key}' must be a nested list.")
    if not value:
        return []

    first_query_rows = value[0]
    if not isinstance(first_query_rows, list):
        raise ValueError(f"Chroma query result '{key}' must be a nested list.")
    return first_query_rows


def _distance_to_similarity(distance: object) -> float:
    normalized_distance = _normalize_distance(distance)
    return 1.0 / (1.0 + normalized_distance)


def _normalize_distance(distance: object) -> float:
    if isinstance(distance, bool) or not isinstance(distance, (int, float)):
        raise ValueError("Chroma query distances must be numeric.")

    normalized_distance = float(distance)
    if not math.isfinite(normalized_distance) or normalized_distance < 0.0:
        raise ValueError("Chroma query distances must be finite, non-negative numbers.")
    return normalized_distance


def _performance_score(metadata: object) -> float:
    if not isinstance(metadata, dict):
        raise ValueError("Chroma query metadatas must contain object entries.")

    score = metadata.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("Chroma query metadata score must be numeric.")

    normalized_score = float(score)
    if not math.isfinite(normalized_score) or not 0.0 <= normalized_score <= 1.0:
        raise ValueError(
            "Chroma query metadata score must be a finite number between 0.0 and 1.0."
        )
    return normalized_score


def _coerce_non_negative_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        return 0.0
    return normalized


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()

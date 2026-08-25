from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import write_training_artifacts
from .contracts import (
    DatasetRef,
    TrainArtifactRef,
    TrainingManifest,
    TrainingPipelineResult,
)

__all__ = [
    "RankedModel",
    "RouterResult",
    "UnifiedRouterBase",
    "RouterTrainer",
]


class MetaRouterBase(ABC):
    def __init__(
        self,
        model: object | None = None,
        yaml_path: str | Path | None = None,
        resources: list[str] | None = None,
    ) -> None:
        self.model = model
        self.yaml_path = yaml_path
        self.resources = resources or []


try:
    MetaRouterBase = getattr(
        importlib.import_module("llmrouter.models.meta_router"),
        "MetaRouter",
    )
except ImportError:
    pass


@dataclass(slots=True)
class RankedModel:
    model_name: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "score": self.score}


@dataclass(slots=True)
class RouterResult:
    ranked_models: list[RankedModel]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_llmrouter_dict(self) -> dict[str, Any]:
        if not self.ranked_models:
            raise ValueError("ranked_models must not be empty.")
        return {
            "ranked_models": [item.to_dict() for item in self.ranked_models],
            "metadata": self.metadata,
            "model_name": self.ranked_models[0].model_name,
        }


class UnifiedRouterBase(MetaRouterBase, ABC):
    def route_single_ranked(self, query_input: dict[str, Any]) -> RouterResult:
        raise NotImplementedError

    def route_single(self, query_input: dict[str, Any]) -> dict[str, Any]:
        return self.route_single_ranked(query_input).to_llmrouter_dict()

    def route_batch(
        self,
        batch: list[dict[str, Any]],
        task_name: str | None = None,
    ) -> list[dict[str, Any]]:
        return [self.route_single(item) for item in batch]


class RouterTrainer(ABC):
    schema_version = "stage2.training.v1"
    router_family: str = ""

    def train(
        self,
        train_artifact: TrainArtifactRef,
        output_dir: Path,
        config: dict[str, Any],
        variant: str | None = None,
        runtime_overrides: dict[str, Any] | None = None,
    ) -> TrainingPipelineResult:
        runtime_overrides = dict(runtime_overrides or {})
        resolved_variant = _normalize_variant(variant)
        resolved_router_family = self.resolve_router_family(config)
        should_write_artifacts = self.should_write_artifacts(config)
        if should_write_artifacts:
            output_dir.mkdir(parents=True, exist_ok=True)
        self.internal_train(
            train_artifact=train_artifact,
            output_dir=output_dir,
            config=config,
            variant=resolved_variant,
            runtime_overrides=runtime_overrides,
        )
        manifest = TrainingManifest(
            schema_version=self.schema_version,
            dataset=DatasetRef(manifest_path=train_artifact.manifest_path),
            trainer_path=self.get_trainer_path(),
            router_family=resolved_router_family,
            router_variant=resolved_variant,
            manifest_path=output_dir / "manifest.json",
            artifact_path=output_dir,
        )
        if not should_write_artifacts:
            return TrainingPipelineResult(
                artifact_root=output_dir,
                manifest_path=manifest.manifest_path,
                training_manifest=manifest,
            )
        return write_training_artifacts(manifest)

    def get_trainer_path(self) -> str:
        return f"{self.__class__.__module__}:{self.__class__.__name__}"

    def resolve_router_family(self, config: dict[str, Any]) -> str:
        if "router_family" in config:
            raise ValueError(
                "config.router_family is not supported; define router_family on the trainer class."
            )
        router_family = self.router_family
        if not isinstance(router_family, str) or not router_family.strip():
            raise ValueError("trainer.router_family must be a non-empty string.")
        return router_family.strip()

    def should_write_artifacts(self, config: dict[str, Any]) -> bool:
        del config
        return True

    @abstractmethod
    def internal_train(
        self,
        *,
        train_artifact: TrainArtifactRef,
        output_dir: Path,
        config: dict[str, Any],
        variant: str | None,
        runtime_overrides: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_router(
        self,
        training_manifest: TrainingManifest,
        runtime_config: dict[str, Any] | None = None,
    ) -> UnifiedRouterBase:
        raise NotImplementedError


def _normalize_variant(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("variant must be null or a non-empty string.")
    return value.strip()

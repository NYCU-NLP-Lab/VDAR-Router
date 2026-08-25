from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from difficulty_aware_router.example_router import ExampleRouter
from experiment.training.base import RouterTrainer, UnifiedRouterBase
from experiment.training.contracts import TrainArtifactRef, TrainingManifest

from .artifact_builder import normalize_candidate_models, write_json

TRAINER_PATH = "difficulty_aware_router.training.example_trainer:ExampleTrainer"
ROUTER_CLASS_PATH = "difficulty_aware_router.example_router:ExampleRouter"


class ExampleTrainer(RouterTrainer):
    router_family = "difficulty-aware-router"

    def get_trainer_path(self) -> str:
        return TRAINER_PATH

    def internal_train(
        self,
        *,
        train_artifact: TrainArtifactRef,
        output_dir: Path,
        config: dict[str, Any],
        variant: str | None = None,
        runtime_overrides: dict[str, Any],
    ) -> None:
        del variant, runtime_overrides
        candidate_models = _extract_candidate_models(train_artifact.train_path)
        seed = _normalize_seed(config.get("seed"))

        artifact_path = output_dir / "example_router.json"
        write_json(
            artifact_path,
            {
                "candidate_models": candidate_models,
                "seed": seed,
                "ranking_algorithm": "random_shuffle",
            },
        )

    def load_router(
        self,
        training_manifest: TrainingManifest,
        runtime_config: dict[str, Any] | None = None,
    ) -> UnifiedRouterBase:
        del runtime_config
        artifact_path = training_manifest.artifact_path / "example_router.json"
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        candidate_models = normalize_candidate_models(payload.get("candidate_models"))
        seed = _normalize_seed(payload.get("seed"))
        return ExampleRouter(candidate_models=candidate_models, seed=seed)


def _extract_candidate_models(train_path: Path) -> list[str]:
    candidate_models: list[str] = []
    seen: set[str] = set()

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
                continue

            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                continue

            model_name = metadata.get("raw_model_name")
            if not isinstance(model_name, str) or not model_name.strip():
                continue

            normalized_name = model_name.strip()
            if normalized_name in seen:
                continue

            seen.add(normalized_name)
            candidate_models.append(normalized_name)

    if not candidate_models:
        raise ValueError(
            "Stage 1 train artifact must contain at least one metadata.raw_model_name value."
        )

    return candidate_models


def _normalize_seed(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("config.seed must be an integer when provided.")
    return value

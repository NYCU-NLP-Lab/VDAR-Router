from __future__ import annotations

import importlib
import json
from pathlib import Path

from .artifacts import build_artifact_root
from .base import RouterTrainer, UnifiedRouterBase
from .contracts import (
    DatasetRef,
    TrainArtifactRef,
    TrainingManifest,
    TrainingPipelineResult,
)
from .settings import Settings, get_settings


def run_training_pipeline(
    *,
    train_manifest_path: Path,
    trainer_path: str,
    variant: str,
    config: dict[str, object] | None = None,
    runtime_overrides: dict[str, object] | None = None,
    settings: Settings | None = None,
) -> TrainingPipelineResult:
    train_artifact = load_train_artifact_reference(train_manifest_path)
    trainer = load_trainer(trainer_path)
    resolved_config = dict(config or {})
    resolved_runtime_overrides = dict(runtime_overrides or {})
    router_family = trainer.resolve_router_family(resolved_config)
    runtime_settings = settings or get_settings()
    artifact_root = build_artifact_root(
        runtime_settings.data_dir, router_family, variant
    )
    result = trainer.train(
        train_artifact=train_artifact,
        output_dir=artifact_root,
        config=resolved_config,
        variant=variant,
        runtime_overrides=resolved_runtime_overrides,
    )
    if _is_dry_run_config(config):
        return result
    _validate_training_manifest(
        training_manifest=result.training_manifest,
        train_artifact=train_artifact,
        artifact_root=artifact_root,
        router_family=router_family,
        variant=variant,
        trainer_path=trainer_path,
    )
    return result


def load_train_artifact_reference(train_manifest_path: Path) -> TrainArtifactRef:
    manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    required_sections = {"inputs", "outputs", "config", "counts"}
    missing_sections = sorted(required_sections - set(manifest))
    if missing_sections:
        raise ValueError(
            f"Stage 1 manifest is missing required sections: {missing_sections}"
        )

    outputs = manifest["outputs"]
    required_outputs = {"artifact_root", "train", "manifest"}
    missing_outputs = sorted(required_outputs - set(outputs))
    if missing_outputs:
        raise ValueError(
            f"Stage 1 manifest outputs are missing required keys: {missing_outputs}"
        )

    artifact_root = train_manifest_path.parent
    train_path = artifact_root / outputs["train"]
    if not train_path.exists():
        raise ValueError(f"Canonical train artifact is missing: {train_path}")

    return TrainArtifactRef(
        artifact_root=artifact_root,
        train_path=train_path,
        manifest_path=train_manifest_path,
        dataset_id=manifest["config"]["dataset_id"],
        dataset_version=manifest["config"]["version"],
    )


def load_trainer(trainer_path: str) -> RouterTrainer:
    module_name, separator, class_name = trainer_path.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError("trainer_path must be in the form 'package.module:ClassName'.")

    module = importlib.import_module(module_name)
    trainer_class = getattr(module, class_name)
    trainer = trainer_class()
    if not isinstance(trainer, RouterTrainer):
        raise TypeError(f"{trainer_path} is not a RouterTrainer implementation.")
    return trainer


def load_router_from_manifest(
    training_manifest: TrainingManifest,
    runtime_config: dict[str, object] | None = None,
) -> UnifiedRouterBase:
    trainer = load_trainer(training_manifest.trainer_path)
    return trainer.load_router(training_manifest, runtime_config=runtime_config)


def load_training_manifest(training_manifest_path: Path) -> TrainingManifest:
    manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version")
    if schema_version is None:
        raise ValueError(
            "Stage 2 training manifest is missing required section: schema_version"
        )
    if schema_version == "stage2.training.v1":
        required_sections = {
            "schema_version",
            "dataset",
            "trainer_path",
            "router_family",
            "router_variant",
            "manifest_path",
            "artifact_path",
        }
        missing_sections = sorted(required_sections - set(manifest))
        if missing_sections:
            raise ValueError(
                f"Stage 2 training manifest is missing required sections: {missing_sections}"
            )

        dataset = manifest["dataset"]
        dataset_manifest_path = (
            dataset.get("manifest_path") if isinstance(dataset, dict) else None
        )
        if (
            not isinstance(dataset_manifest_path, str)
            or not dataset_manifest_path.strip()
        ):
            raise ValueError(
                "Stage 2 training manifest dataset.manifest_path must be a non-empty string."
            )

        trainer_path = manifest["trainer_path"]
        if not isinstance(trainer_path, str) or ":" not in trainer_path:
            raise ValueError(
                "Stage 2 training manifest trainer_path must be a 'package.module:ClassName' string."
            )

        router_family = manifest["router_family"]
        if not isinstance(router_family, str) or not router_family.strip():
            raise ValueError(
                "Stage 2 training manifest router_family must be a non-empty string."
            )

        router_variant = manifest["router_variant"]
        if router_variant is not None and (
            not isinstance(router_variant, str) or not router_variant.strip()
        ):
            raise ValueError(
                "Stage 2 training manifest router_variant must be null or a non-empty string."
            )

        manifest_path = manifest["manifest_path"]
        if not isinstance(manifest_path, str) or not manifest_path.strip():
            raise ValueError(
                "Stage 2 training manifest manifest_path must be a non-empty string."
            )

        artifact_path = manifest["artifact_path"]
        if not isinstance(artifact_path, str) or not artifact_path.strip():
            raise ValueError(
                "Stage 2 training manifest artifact_path must be a non-empty string."
            )

        return TrainingManifest(
            schema_version=schema_version,
            dataset=DatasetRef(manifest_path=Path(dataset_manifest_path)),
            trainer_path=trainer_path,
            router_family=router_family.strip(),
            router_variant=(
                router_variant.strip() if isinstance(router_variant, str) else None
            ),
            manifest_path=Path(manifest_path),
            artifact_path=Path(artifact_path),
        )

    raise ValueError(
        f"Unsupported Stage 2 training manifest schema_version: {schema_version}"
    )


def _is_dry_run_config(config: dict[str, object] | None) -> bool:
    return isinstance(config, dict) and config.get("dry_run") is True


def _validate_training_manifest(
    *,
    training_manifest: TrainingManifest,
    train_artifact: TrainArtifactRef,
    artifact_root: Path,
    router_family: str,
    variant: str,
    trainer_path: str,
) -> None:
    if training_manifest.dataset.manifest_path != train_artifact.manifest_path:
        raise ValueError(
            "training manifest dataset.manifest_path does not match Stage 1 input."
        )
    if training_manifest.trainer_path != trainer_path:
        raise ValueError("training manifest trainer_path does not match request.")
    if training_manifest.router_family != router_family:
        raise ValueError("training manifest router_family does not match request.")
    if training_manifest.router_variant != variant:
        raise ValueError("training manifest router_variant does not match request.")
    if training_manifest.artifact_path != artifact_root:
        raise ValueError("training manifest artifact_path does not match output root.")
    if training_manifest.manifest_path != artifact_root / "manifest.json":
        raise ValueError("training manifest manifest_path does not match output root.")

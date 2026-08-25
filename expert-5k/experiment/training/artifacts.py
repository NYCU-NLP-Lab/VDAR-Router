from __future__ import annotations

import json
from pathlib import Path

from .contracts import TrainingManifest, TrainingPipelineResult


def build_artifact_root(data_dir: Path, router_family: str, run_label: str) -> Path:
    return data_dir / router_family / run_label


def write_training_artifacts(
    training_manifest: TrainingManifest,
) -> TrainingPipelineResult:
    artifact_root = training_manifest.artifact_path
    artifact_root.mkdir(parents=True, exist_ok=True)
    training_manifest.manifest_path.write_text(
        json.dumps(training_manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = TrainingPipelineResult(
        artifact_root=artifact_root,
        manifest_path=training_manifest.manifest_path,
        training_manifest=training_manifest,
    )
    validate_training_artifact_contract(result)
    return result


def validate_training_artifact_contract(result: TrainingPipelineResult) -> None:
    if not result.artifact_root.exists():
        raise ValueError("Training artifact root does not exist.")
    if not result.manifest_path.exists():
        raise ValueError("Training artifact manifest is missing.")

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
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
            f"training manifest is missing required sections: {missing_sections}"
        )

    if manifest.get("schema_version") != "stage2.training.v1":
        raise ValueError(
            "training manifest schema_version must be 'stage2.training.v1'."
        )

    trainer_path = manifest.get("trainer_path")
    if not isinstance(trainer_path, str) or ":" not in trainer_path:
        raise ValueError(
            "training manifest trainer_path must be a 'package.module:ClassName' string."
        )

    router_family = manifest.get("router_family")
    if not isinstance(router_family, str) or not router_family.strip():
        raise ValueError("training manifest router_family must be a non-empty string.")

    router_variant = manifest.get("router_variant")
    if router_variant is not None and (
        not isinstance(router_variant, str) or not router_variant.strip()
    ):
        raise ValueError(
            "training manifest router_variant must be null or a non-empty string."
        )

    dataset = manifest.get("dataset")
    dataset_manifest_path = (
        dataset.get("manifest_path") if isinstance(dataset, dict) else None
    )
    if (
        not isinstance(dataset_manifest_path, str)
        or not Path(dataset_manifest_path).exists()
    ):
        raise ValueError("training manifest dataset.manifest_path must exist.")

    manifest_path = manifest.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        raise ValueError("training manifest manifest_path must be a non-empty string.")

    artifact_path = Path(manifest["artifact_path"])
    if not artifact_path.exists():
        raise ValueError(
            f"training manifest artifact_path does not exist: {artifact_path}"
        )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DatasetRef:
    manifest_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {"manifest_path": str(self.manifest_path)}


@dataclass(slots=True)
class TrainArtifactRef:
    artifact_root: Path
    train_path: Path
    manifest_path: Path
    dataset_id: str
    dataset_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_root": str(self.artifact_root),
            "train_path": str(self.train_path),
            "manifest_path": str(self.manifest_path),
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
        }


@dataclass(slots=True)
class TrainingManifest:
    schema_version: str
    dataset: DatasetRef
    trainer_path: str
    router_family: str
    router_variant: str | None
    manifest_path: Path
    artifact_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset.to_dict(),
            "trainer_path": self.trainer_path,
            "router_family": self.router_family,
            "router_variant": self.router_variant,
            "manifest_path": str(self.manifest_path),
            "artifact_path": str(self.artifact_path),
        }


@dataclass(slots=True)
class TrainingPipelineResult:
    artifact_root: Path
    manifest_path: Path
    training_manifest: TrainingManifest

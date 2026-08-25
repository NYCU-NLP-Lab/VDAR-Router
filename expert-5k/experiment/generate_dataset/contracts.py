from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

STABLE_ROW_FIELDS = (
    "id",
    "prompt_id",
    "input",
    "output",
    "score",
    "input_token",
    "output_tokken",
    "metadata",
)


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BuildInputs:
    source_adapter: str
    source_location: str
    source_config_summary: dict[str, Any]


@dataclass(slots=True)
class SplitConfig:
    strategy: str = "prompt_group_train_test"
    test_fraction: float = 0.2
    seed: int = 0
    occupational_tags: tuple[str, ...] | None = None


@dataclass(slots=True)
class DatasetBuildConfig:
    dataset_id: str
    version: str
    score_mode: str
    split: SplitConfig
    output_root: Path
    llm_config_path: Path
    max_model_groups: int | None = None
    max_prompt_groups: int | None = None
    max_evaluation_order: int = 1


@dataclass(slots=True)
class DatasetCounts:
    total_rows: int
    train_rows: int
    test_rows: int
    total_prompt_groups: int
    train_prompt_groups: int
    test_prompt_groups: int


@dataclass(slots=True)
class DatasetMetrics:
    average_models_per_prompt: float


@dataclass(slots=True)
class DatasetBuildResult:
    artifact_root: Path
    full_path: Path
    train_path: Path
    test_path: Path
    manifest_path: Path
    counts: DatasetCounts
    metrics: DatasetMetrics

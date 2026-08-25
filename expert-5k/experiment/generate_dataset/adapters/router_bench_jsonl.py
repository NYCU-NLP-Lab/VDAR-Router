from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..contracts import CanonicalRow
from .router_bench import RouterBenchAdapter


class RouterBenchJsonlAdapter(RouterBenchAdapter):
    adapter_name = "router-bench-jsonl"
    train_dataset_name = "data/routerbench/data/routing_data_train.jsonl"
    test_dataset_name = "data/routerbench/data/routing_data_test.jsonl"
    train_jsonl_config_key = "train_jsonl_path"
    test_jsonl_config_key = "test_jsonl_path"

    def __init__(
        self,
        *,
        split: str = "train",
        source_path: Path | None = None,
        train_jsonl_path: str | Path | None = None,
        test_jsonl_path: str | Path | None = None,
    ) -> None:
        super().__init__(split=split)
        self._source_path = self._resolve_source_path(
            source_path=source_path,
            train_jsonl_path=train_jsonl_path,
            test_jsonl_path=test_jsonl_path,
        )
        self.source_location = str(self._source_path)

    def load_rows(self) -> list[CanonicalRow]:
        rows: list[CanonicalRow] = []
        for raw_index, payload in enumerate(self._load_records()):
            if isinstance(payload, dict):
                normalized = self._normalize_jsonl_row(payload, raw_index=raw_index)
                if normalized is not None:
                    rows.append(normalized)
        return rows

    def _load_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line in self._source_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
        return records

    def _normalize_jsonl_row(
        self, payload: dict[str, Any], *, raw_index: int
    ) -> CanonicalRow | None:
        embedding_id = payload.get("embedding_id")
        if isinstance(embedding_id, bool) or not isinstance(embedding_id, int):
            raise ValueError("Missing required integer field 'embedding_id'.")

        raw_model_name = self._require_string(payload, "model_name")
        prompt_text = self._stringify_text_sequence(
            payload.get("query"), field_name="query"
        )
        score = self._read_score(payload.get("performance"))
        if score is None:
            return None

        prompt_id = self._build_prompt_id(
            embedding_id=embedding_id, prompt_text=prompt_text
        )
        row_id = f"{self._split}:{prompt_id}:{raw_index}:{raw_model_name}"
        response_text = ""
        return CanonicalRow(
            id=row_id,
            prompt_id=prompt_id,
            input=prompt_text,
            output=response_text,
            score=score,
            input_token=self._count_tokens(prompt_text),
            output_tokken=0,
            metadata={
                "source_adapter": self.adapter_name,
                "source_dataset": self.source_location,
                "source_split": self._split,
                "source_row_id": row_id,
                "embedding_id": embedding_id,
                "raw_index": raw_index,
                "raw_model_name": raw_model_name,
                "cost": self._read_score(payload.get("cost")),
            },
        )

    def _build_prompt_id(self, *, embedding_id: int, prompt_text: str) -> str:
        prompt_hash = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:12]
        return f"{embedding_id}:{prompt_hash}"

    def _resolve_source_path(
        self,
        *,
        source_path: Path | None,
        train_jsonl_path: str | Path | None,
        test_jsonl_path: str | Path | None,
    ) -> Path:
        if source_path is not None:
            return source_path
        if self._split == "train":
            return self._resolve_configured_or_default_path(
                train_jsonl_path,
                config_key=self.train_jsonl_config_key,
                default_relative_path=self.train_dataset_name,
            )
        if self._split == "test":
            return self._resolve_configured_or_default_path(
                test_jsonl_path,
                config_key=self.test_jsonl_config_key,
                default_relative_path=self.test_dataset_name,
            )
        raise ValueError(
            "router-bench-jsonl only supports split='train' or split='test'."
        )

    def _resolve_configured_or_default_path(
        self,
        value: str | Path | None,
        *,
        config_key: str,
        default_relative_path: str,
    ) -> Path:
        if value is None:
            return Path(__file__).resolve().parents[3] / default_relative_path
        if isinstance(value, Path):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                f"router-bench-jsonl requires '{config_key}' in source_config_summary when source_split='{self._split}'."
            )
        return Path(stripped)

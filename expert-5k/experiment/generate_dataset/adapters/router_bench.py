from __future__ import annotations

import ast
import math
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from ..contracts import CanonicalRow
from .base import SourceAdapter


class RouterBenchAdapter(SourceAdapter):
    adapter_name = "router-bench"
    dataset_name = "withmartian/routerbench"
    raw_data_file = "routerbench_raw.pkl"

    def __init__(self, *, split: str = "raw") -> None:
        self._split = split

    def load_rows(self) -> list[CanonicalRow]:
        rows: list[CanonicalRow] = []
        for payload in self._load_records():
            if isinstance(payload, dict):
                normalized = self._normalize_row(payload)
                if normalized is not None:
                    rows.append(normalized)
        return rows

    def _load_records(self) -> list[dict[str, Any]]:
        path = hf_hub_download(
            repo_id=self.dataset_name, filename=self.raw_data_file, repo_type="dataset"
        )
        frame = pd.read_pickle(path)
        columns = [str(column) for column in frame.columns]
        return [
            dict(zip(columns, row, strict=False))
            for row in frame.itertuples(index=False, name=None)
        ]

    def _normalize_row(self, payload: dict[str, Any]) -> CanonicalRow | None:
        sample_id = self._require_string(payload, "sample_id")
        raw_model_name = self._require_string(payload, "model_name")
        row_id = f"{sample_id}:{raw_model_name}"
        prompt_text = self._stringify_text_sequence(
            payload.get("prompt"), field_name="prompt"
        )
        response_text = self._stringify_text_sequence(
            payload.get("model_response"),
            field_name="model_response",
            allow_empty=True,
        )
        score = self._read_score(payload.get("performance"))
        if score is None:
            return None

        return CanonicalRow(
            id=row_id,
            prompt_id=sample_id,
            input=prompt_text,
            output=response_text,
            score=score,
            input_token=self._count_tokens(prompt_text),
            output_tokken=self._count_tokens(response_text),
            metadata={
                "source_adapter": self.adapter_name,
                "source_dataset": self.dataset_name,
                "source_split": self._split,
                "source_row_id": row_id,
                "sample_id": sample_id,
                "raw_index": payload.get("index"),
                "eval_name": self._require_string(payload, "eval_name"),
                "raw_model_name": raw_model_name,
                "cost": payload.get("cost"),
            },
        )

    def _stringify_text_sequence(
        self, value: Any, *, field_name: str, allow_empty: bool = False
    ) -> str:
        parsed = value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                if allow_empty:
                    return ""
                raise ValueError(f"Missing required text field '{field_name}'.")
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                return stripped

        if isinstance(parsed, str):
            text = parsed.strip()
            if text:
                return text
            if allow_empty:
                return ""
            raise ValueError(f"Missing required text field '{field_name}'.")

        if isinstance(parsed, (list, tuple)):
            text_parts = [
                part.strip()
                for part in parsed
                if isinstance(part, str) and part.strip()
            ]
            if text_parts:
                return "\n".join(text_parts)
            if allow_empty:
                return ""

        raise ValueError(
            f"Field '{field_name}' did not contain a usable text sequence."
        )

    def _read_score(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(score):
            return None
        return score

    def _count_tokens(self, text: str) -> int:
        return len(text.split())

    def _require_string(self, payload: dict[str, Any], field_name: str) -> str:
        value = payload.get(field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing required string field '{field_name}'.")
        return value

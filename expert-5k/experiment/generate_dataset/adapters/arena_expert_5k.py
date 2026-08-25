from __future__ import annotations

import hashlib
from typing import Any

from datasets import load_dataset

from ..contracts import CanonicalRow
from ..util import parse_numpy_repr_payload
from .base import SourceAdapter


class ArenaExpert5KAdapter(SourceAdapter):
    adapter_name = "arena-expert-5k"
    dataset_name = "lmarena-ai/arena-expert-5k"

    def __init__(self, *, split: str = "train") -> None:
        self._split = split

    def load_rows(self) -> list[CanonicalRow]:
        dataset = load_dataset(self.dataset_name, split=self._split)
        rows: list[CanonicalRow] = []
        for payload in dataset:
            if isinstance(payload, dict):
                rows.extend(self._normalize_battle(payload))
        return rows

    def _normalize_battle(self, payload: dict[str, Any]) -> list[CanonicalRow]:
        winner = payload.get("winner")
        if winner not in {"model_a", "model_b"}:
            return []

        battle_id = self._require_string(payload, "id")
        model_a_name = self._read_model_name(
            payload, preferred_keys=("model_a", "model_a_name")
        )
        model_b_name = self._read_model_name(
            payload, preferred_keys=("model_b", "model_b_name")
        )
        prompt_text = self._build_prompt_text(payload)
        prompt_id = self._build_prompt_id(payload, prompt_text)
        occupational_tags = payload.get("occupational_tags")

        return [
            self._build_response_row(
                battle_id=battle_id,
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                source_payload=payload,
                response_key="conversation_a",
                raw_model_name=model_a_name,
                score=1.0 if winner == "model_a" else 0.0,
                row_suffix="model_a",
                occupational_tags=occupational_tags,
            ),
            self._build_response_row(
                battle_id=battle_id,
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                source_payload=payload,
                response_key="conversation_b",
                raw_model_name=model_b_name,
                score=1.0 if winner == "model_b" else 0.0,
                row_suffix="model_b",
                occupational_tags=occupational_tags,
            ),
        ]

    def _build_response_row(
        self,
        *,
        battle_id: str,
        prompt_id: str,
        prompt_text: str,
        source_payload: dict[str, Any],
        response_key: str,
        raw_model_name: str,
        score: float,
        row_suffix: str,
        occupational_tags: Any,
    ) -> CanonicalRow:
        conversation = source_payload.get(response_key)
        response_text = self._extract_response_text(conversation, response_key)

        return CanonicalRow(
            id=f"{battle_id}:{row_suffix}",
            prompt_id=prompt_id,
            input=prompt_text,
            output=response_text,
            score=score,
            input_token=self._count_tokens(prompt_text),
            output_tokken=self._count_tokens(response_text),
            metadata={
                "source_adapter": self.adapter_name,
                "source_dataset": self.dataset_name,
                "source_split": self._split,
                "source_row_id": battle_id,
                "battle_id": battle_id,
                "pair_id": battle_id,
                "evaluation_order": source_payload.get("evaluation_order"),
                "winner": source_payload.get("winner"),
                "query_model_id": raw_model_name,
                "raw_model_name": raw_model_name,
                "occupational_tags": occupational_tags,
                "source_response_key": response_key,
            },
        )

    def _build_prompt_text(self, payload: dict[str, Any]) -> str:
        for key in ("conversation_a", "conversation_b"):
            messages = self._conversation_to_messages(payload.get(key))
            prompt_only = [
                message for message in messages if message["role"] != "assistant"
            ]
            if prompt_only:
                return self._stringify_messages(prompt_only, include_assistant=False)

        full_conversation = payload.get("full_conversation")
        messages = self._conversation_to_messages(full_conversation)
        prompt_only = [
            message for message in messages if message["role"] != "assistant"
        ]
        if prompt_only:
            return self._stringify_messages(prompt_only, include_assistant=False)

        raise ValueError("Arena Expert 5K row is missing usable conversation context.")

    def _build_prompt_id(self, payload: dict[str, Any], prompt_text: str) -> str:
        prompt_id = payload.get("prompt_id")
        if isinstance(prompt_id, str) and prompt_id:
            return prompt_id

        digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]
        return f"prompt-{digest}"

    def _extract_response_text(self, conversation: Any, response_key: str) -> str:
        messages = self._conversation_to_messages(conversation)
        if not messages:
            return ""

        for message in reversed(messages):
            if message["role"] == "assistant":
                content = message["content"]
                if content:
                    return content

        return ""

    def _stringify_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        include_assistant: bool,
    ) -> str:
        rendered: list[str] = []
        for message in messages:
            role = self._read_role(message)
            if role == "assistant" and not include_assistant:
                continue
            content = self._read_content(message)
            if content:
                rendered.append(f"{role}: {content}")
        if not rendered:
            raise ValueError("Conversation messages did not contain any usable text.")
        return "\n".join(rendered)

    def _conversation_to_messages(self, conversation: Any) -> list[dict[str, str]]:
        if isinstance(conversation, list):
            return self._extract_messages(conversation)

        if isinstance(conversation, str) and conversation.strip():
            parsed = parse_numpy_repr_payload(conversation)
            return self._extract_messages(parsed)

        return []

    def _extract_messages(self, payload: Any) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        if isinstance(payload, dict):
            role = self._read_role(payload)
            content = self._read_content(payload)
            if content:
                messages.append({"role": role, "content": content})
                return messages

            for value in payload.values():
                messages.extend(self._extract_messages(value))
            return messages

        if isinstance(payload, list):
            for item in payload:
                messages.extend(self._extract_messages(item))

        return messages

    def _read_model_name(
        self,
        payload: dict[str, Any],
        *,
        preferred_keys: tuple[str, ...],
    ) -> str:
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        raise ValueError(f"Missing model name in fields {preferred_keys}.")

    def _read_role(self, message: Any) -> str:
        if not isinstance(message, dict):
            return "unknown"
        role = message.get("role")
        return role if isinstance(role, str) else "unknown"

    def _read_content(self, message: Any) -> str:
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if isinstance(content, list):
            return self._read_content_parts(content)
        return content.strip() if isinstance(content, str) else ""

    def _read_content_parts(self, parts: list[Any]) -> str:
        text_parts: list[str] = []
        for part in parts:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
        return "\n".join(text_parts)

    def _count_tokens(self, text: str) -> int:
        return len(text.split())

    def _require_string(self, payload: dict[str, Any], field_name: str) -> str:
        value = payload.get(field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing required string field '{field_name}'.")
        return value

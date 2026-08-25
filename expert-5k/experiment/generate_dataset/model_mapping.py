from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


class ModelMappingError(ValueError):
    """Raised when llm model mapping config is invalid."""


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_cost_per_token: float
    output_cost_per_token: float


@dataclass(frozen=True, slots=True)
class LLMConfig:
    model_name_mapping: dict[str, str]
    model_pricing: dict[str, ModelPricing]
    pricing_fallback_model_name: str | None = None


def load_model_mapping(config_path: Path) -> dict[str, str]:
    return load_llm_config(config_path).model_name_mapping


def load_llm_config(config_path: Path) -> LLMConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    mappings = payload.get("model_name_mapping")
    pricing_payload = payload.get("model_pricing")
    pricing_fallback_payload = payload.get("pricing_fallback")

    if not isinstance(mappings, dict) or not mappings:
        raise ModelMappingError(
            "config/llm.json must define a non-empty 'model_name_mapping' object."
        )
    if not isinstance(pricing_payload, dict) or not pricing_payload:
        raise ModelMappingError(
            "config/llm.json must define a non-empty 'model_pricing' object."
        )

    normalized: dict[str, str] = {}
    for raw_name, openrouter_name in mappings.items():
        if not isinstance(raw_name, str) or not isinstance(openrouter_name, str):
            raise ModelMappingError(
                "model_name_mapping entries must map string names to string names."
            )
        normalized[raw_name] = openrouter_name

    model_pricing: dict[str, ModelPricing] = {}
    for model_name, pricing in pricing_payload.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise ModelMappingError(
                "model_pricing entries must use non-empty string model names."
            )
        if not isinstance(pricing, dict):
            raise ModelMappingError(
                "model_pricing entries must be objects with per-token price fields."
            )
        input_cost_per_token = _require_non_negative_float(
            pricing.get("input_cost_per_token"),
            field_name=(f"model_pricing.{model_name}.input_cost_per_token"),
        )
        output_cost_per_token = _require_non_negative_float(
            pricing.get("output_cost_per_token"),
            field_name=(f"model_pricing.{model_name}.output_cost_per_token"),
        )
        model_pricing[model_name.strip()] = ModelPricing(
            input_cost_per_token=input_cost_per_token,
            output_cost_per_token=output_cost_per_token,
        )

    pricing_fallback_model_name: str | None = None
    if pricing_fallback_payload is not None:
        if not isinstance(pricing_fallback_payload, dict):
            raise ModelMappingError("pricing_fallback must be an object when provided.")
        raw_default_model_name = pricing_fallback_payload.get("default_model_name")
        if (
            not isinstance(raw_default_model_name, str)
            or not raw_default_model_name.strip()
        ):
            raise ModelMappingError(
                "pricing_fallback.default_model_name must be a non-empty string."
            )
        pricing_fallback_model_name = raw_default_model_name.strip()
        if pricing_fallback_model_name not in model_pricing:
            raise ModelMappingError(
                "pricing_fallback.default_model_name must reference a model_pricing entry."
            )

    return LLMConfig(
        model_name_mapping=normalized,
        model_pricing=model_pricing,
        pricing_fallback_model_name=pricing_fallback_model_name,
    )


def normalize_model_name(raw_name: str, mapping: dict[str, str]) -> str:
    return mapping.get(raw_name, raw_name)


def _require_non_negative_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelMappingError(f"{field_name} must be a non-negative number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ModelMappingError(f"{field_name} must be a non-negative number.")
    return normalized

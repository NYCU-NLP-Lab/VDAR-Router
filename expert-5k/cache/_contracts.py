from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

CacheMode = Literal["record", "refresh", "deferred", "replay_only"]


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _require_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data[key]
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object")
    return value


def _require_list_of_dicts(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError(f"{key} items must be objects")
        normalized.append(item)
    return normalized


@dataclass(slots=True)
class CacheConfig:
    cache_dir: Path
    cache_mode: CacheMode
    provider_name: str
    provider_base_url: str


@dataclass(slots=True)
class ManifestPaths:
    root_dir: Path
    requests_path: Path
    responses_path: Path


@dataclass(slots=True)
class ReplayKey:
    endpoint: str
    request_hash: str


@dataclass(slots=True)
class CanonicalRequestRow:
    schema_version: str
    canonicalization_version: str
    endpoint: str
    request_hash: str
    request_hash_algorithm: str
    request_body_normalized: dict[str, Any]
    created_at: str
    provider_name: str
    provider_base_url: str
    legacy_aliases: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "canonicalization_version": self.canonicalization_version,
            "endpoint": self.endpoint,
            "request_hash": self.request_hash,
            "request_hash_algorithm": self.request_hash_algorithm,
            "request_body_normalized": self.request_body_normalized,
            "created_at": self.created_at,
            "provider_name": self.provider_name,
            "provider_base_url": self.provider_base_url,
            "legacy_aliases": list(self.legacy_aliases),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanonicalRequestRow":
        return cls(
            schema_version=_require_str(data, "schema_version"),
            canonicalization_version=_require_str(data, "canonicalization_version"),
            endpoint=_require_str(data, "endpoint"),
            request_hash=_require_str(data, "request_hash"),
            request_hash_algorithm=_require_str(data, "request_hash_algorithm"),
            request_body_normalized=_require_dict(data, "request_body_normalized"),
            created_at=_require_str(data, "created_at"),
            provider_name=_require_str(data, "provider_name"),
            provider_base_url=_require_str(data, "provider_base_url"),
            legacy_aliases=_require_list_of_dicts(data, "legacy_aliases"),
        )


@dataclass(slots=True)
class CanonicalResponseRow:
    schema_version: str
    canonicalization_version: str
    endpoint: str
    request_hash: str
    response_body_canonical: dict[str, Any]
    provider_response_metadata: dict[str, Any]
    created_at: str
    source_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "canonicalization_version": self.canonicalization_version,
            "endpoint": self.endpoint,
            "request_hash": self.request_hash,
            "response_body_canonical": self.response_body_canonical,
            "provider_response_metadata": self.provider_response_metadata,
            "created_at": self.created_at,
            "source_mode": self.source_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanonicalResponseRow":
        return cls(
            schema_version=_require_str(data, "schema_version"),
            canonicalization_version=_require_str(data, "canonicalization_version"),
            endpoint=_require_str(data, "endpoint"),
            request_hash=_require_str(data, "request_hash"),
            response_body_canonical=_require_dict(data, "response_body_canonical"),
            provider_response_metadata=_require_dict(
                data, "provider_response_metadata"
            ),
            created_at=_require_str(data, "created_at"),
            source_mode=_require_str(data, "source_mode"),
        )

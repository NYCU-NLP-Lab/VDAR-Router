from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from ._constants import (
    CANONICALIZATION_VERSION,
    REDACTED_VALUE,
    REQUEST_HASH_ALGORITHM,
    SCHEMA_VERSION,
    SENSITIVE_KEY_NAMES,
    TRANSPORT_ONLY_REQUEST_FIELDS,
)
from ._contracts import CanonicalRequestRow


def normalize_base_url(base_url: str | None) -> str:
    if not base_url:
        return "https://api.openai.com/v1"
    return base_url.rstrip("/")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_json_value(item) for item in value]
    return value


def scrub_sensitive_data(value: Any) -> Any:
    if isinstance(value, dict):
        scrubbed: dict[str, Any] = {}
        for key, item in value.items():
            if key.lower() in SENSITIVE_KEY_NAMES:
                scrubbed[key] = REDACTED_VALUE
            else:
                scrubbed[key] = scrub_sensitive_data(item)
        return scrubbed
    if isinstance(value, list):
        return [scrub_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return [scrub_sensitive_data(item) for item in value]
    return value


def canonicalize_request_kwargs(request_kwargs: dict[str, Any]) -> dict[str, Any]:
    semantic_kwargs = {
        key: value
        for key, value in request_kwargs.items()
        if key not in TRANSPORT_ONLY_REQUEST_FIELDS
    }
    return normalize_json_value(scrub_sensitive_data(semantic_kwargs))


def canonical_json_dumps(payload: Any) -> str:
    return json.dumps(
        normalize_json_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def build_request_hash(
    *,
    endpoint: str,
    provider_name: str,
    provider_base_url: str,
    request_body_normalized: dict[str, Any],
) -> str:
    hash_material = {
        "endpoint": endpoint,
        "provider_name": provider_name,
        "provider_base_url": provider_base_url,
        "request_body_normalized": request_body_normalized,
    }
    return hashlib.sha256(
        canonical_json_dumps(hash_material).encode("utf-8")
    ).hexdigest()


def build_canonical_request_row(
    *,
    endpoint: str,
    request_kwargs: dict[str, Any],
    provider_name: str,
    provider_base_url: str,
) -> CanonicalRequestRow:
    request_body_normalized = canonicalize_request_kwargs(request_kwargs)
    request_hash = build_request_hash(
        endpoint=endpoint,
        provider_name=provider_name,
        provider_base_url=provider_base_url,
        request_body_normalized=request_body_normalized,
    )
    return CanonicalRequestRow(
        schema_version=SCHEMA_VERSION,
        canonicalization_version=CANONICALIZATION_VERSION,
        endpoint=endpoint,
        request_hash=request_hash,
        request_hash_algorithm=REQUEST_HASH_ALGORITHM,
        request_body_normalized=request_body_normalized,
        created_at=utc_timestamp(),
        provider_name=provider_name,
        provider_base_url=provider_base_url,
        legacy_aliases=[],
    )

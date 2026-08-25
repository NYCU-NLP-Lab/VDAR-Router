from __future__ import annotations

from typing import Any

from openai._models import construct_type
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.create_embedding_response import CreateEmbeddingResponse

from ._canonical import utc_timestamp
from ._constants import CANONICALIZATION_VERSION, SCHEMA_VERSION
from ._contracts import CanonicalRequestRow, CanonicalResponseRow


def _model_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False)
    if hasattr(value, "dict"):
        return value.dict()
    raise TypeError(f"Unsupported OpenAI model object: {type(value)!r}")


def runtime_object_to_response_row(
    *,
    request_row: CanonicalRequestRow,
    runtime_object: Any,
    source_mode: str,
    provider_response_metadata_updates: dict[str, Any] | None = None,
) -> CanonicalResponseRow:
    response_id = getattr(runtime_object, "id", None)
    response_object = getattr(runtime_object, "object", None)
    provider_response_metadata = {
        key: value
        for key, value in {
            "id": response_id,
            "object": response_object,
        }.items()
        if value is not None
    }
    if provider_response_metadata_updates:
        provider_response_metadata.update(provider_response_metadata_updates)
    return CanonicalResponseRow(
        schema_version=SCHEMA_VERSION,
        canonicalization_version=CANONICALIZATION_VERSION,
        endpoint=request_row.endpoint,
        request_hash=request_row.request_hash,
        response_body_canonical=_model_to_dict(runtime_object),
        provider_response_metadata=provider_response_metadata,
        created_at=utc_timestamp(),
        source_mode=source_mode,
    )


def reconstruct_runtime_object(
    *, endpoint: str, response_body_canonical: dict[str, Any]
) -> Any:
    if endpoint == "chat.completions.create":
        return construct_type(value=response_body_canonical, type_=ChatCompletion)
    if endpoint == "embeddings.create":
        return construct_type(
            value=response_body_canonical, type_=CreateEmbeddingResponse
        )
    raise ValueError(f"Unsupported endpoint: {endpoint}")

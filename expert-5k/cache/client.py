from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from openai import AsyncOpenAI, OpenAI

from ._canonical import build_canonical_request_row, normalize_base_url
from ._constants import DEFAULT_PROVIDER_NAME, SUPPORTED_CACHE_MODES
from ._contracts import CacheConfig
from ._openai import reconstruct_runtime_object, runtime_object_to_response_row
from ._storage import CacheStorage
from .exceptions import CacheDeferredRequest, CacheModeError, CacheReplayMiss
from .settings import get_settings


class _SyncResponsesResource:
    def __init__(self, client: "CacheOpenAI") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._request_sync(
            endpoint="chat.completions.create", request_kwargs=kwargs
        )


class _SyncEmbeddingsResource:
    def __init__(self, client: "CacheOpenAI") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._request_sync(
            endpoint="embeddings.create", request_kwargs=kwargs
        )


class _AsyncResponsesResource:
    def __init__(self, client: "CacheAsyncOpenAI") -> None:
        self._client = client

    async def create(self, **kwargs: Any) -> Any:
        return await self._client._request_async(
            endpoint="chat.completions.create", request_kwargs=kwargs
        )


class _SyncChatResource:
    def __init__(self, client: "CacheOpenAI") -> None:
        self.completions = _SyncResponsesResource(client)


class _AsyncChatResource:
    def __init__(self, client: "CacheAsyncOpenAI") -> None:
        self.completions = _AsyncResponsesResource(client)


class _AsyncEmbeddingsResource:
    def __init__(self, client: "CacheAsyncOpenAI") -> None:
        self._client = client

    async def create(self, **kwargs: Any) -> Any:
        return await self._client._request_async(
            endpoint="embeddings.create", request_kwargs=kwargs
        )


class _CacheClientBase:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        timeout: Any = 600,
        max_retries: int | None = 3,
        default_headers: dict[str, str] | None = None,
        http_client: Any = None,
        cache_dir: str | Path | None = None,
        cache_mode: Literal["record", "refresh", "deferred", "replay_only"] = "record",
    ) -> None:
        if cache_mode not in SUPPORTED_CACHE_MODES:
            raise CacheModeError(f"Unsupported cache mode: {cache_mode}")
        settings = get_settings()
        resolved_api_key = api_key if api_key is not None else settings.llm_api_key
        resolved_base_url = base_url if base_url is not None else settings.llm_base_url
        self._client_kwargs = {
            "api_key": resolved_api_key,
            "base_url": resolved_base_url,
            "organization": organization,
            "project": project,
            "timeout": timeout,
            "max_retries": max_retries,
            "default_headers": default_headers,
        }
        if http_client:
            self._client_kwargs["http_client"] = http_client
        resolved_cache_dir = (
            Path(cache_dir) if cache_dir is not None else settings.cache_dir
        )
        self._config = CacheConfig(
            cache_dir=resolved_cache_dir,
            cache_mode=cache_mode,
            provider_name=DEFAULT_PROVIDER_NAME,
            provider_base_url=normalize_base_url(resolved_base_url),
        )
        self._storage = CacheStorage(self._config.cache_dir)
        if cache_mode != "replay_only":
            self._storage.ensure_base_layout()

    def _build_request_row(
        self, *, endpoint: str, request_kwargs: dict[str, Any]
    ) -> Any:
        return build_canonical_request_row(
            endpoint=endpoint,
            request_kwargs=request_kwargs,
            provider_name=self._config.provider_name,
            provider_base_url=self._config.provider_base_url,
        )

    def _replay(self, *, endpoint: str, request_hash: str) -> Any:
        cached_row = self._lookup_cached_response(
            endpoint=endpoint, request_hash=request_hash
        )
        if cached_row is None:
            raise CacheReplayMiss(
                endpoint=endpoint,
                request_hash=request_hash,
                manifest_path=self._storage.manifest_paths(endpoint).responses_path,
            )
        return reconstruct_runtime_object(
            endpoint=endpoint,
            response_body_canonical=cached_row.response_body_canonical,
        )

    def _lookup_cached_response(
        self, *, endpoint: str, request_hash: str
    ) -> Any | None:
        return self._storage.find_response_row(
            endpoint=endpoint,
            request_hash=request_hash,
        )


class CacheOpenAI(_CacheClientBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: Any | None = None
        self.chat = _SyncChatResource(self)
        self.embeddings = _SyncEmbeddingsResource(self)

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = OpenAI(**self._client_kwargs)
        return self._client

    def _call_network(self, *, endpoint: str, request_kwargs: dict[str, Any]) -> Any:
        client = self._get_client()
        if endpoint == "chat.completions.create":
            return client.chat.completions.create(**request_kwargs)
        if endpoint == "embeddings.create":
            return client.embeddings.create(**request_kwargs)
        raise ValueError(f"Unsupported endpoint: {endpoint}")

    def _request_sync(self, *, endpoint: str, request_kwargs: dict[str, Any]) -> Any:
        request_row = self._build_request_row(
            endpoint=endpoint, request_kwargs=request_kwargs
        )
        if self._config.cache_mode == "record":
            cached_row = self._lookup_cached_response(
                endpoint=endpoint, request_hash=request_row.request_hash
            )
            if cached_row is not None:
                return reconstruct_runtime_object(
                    endpoint=endpoint,
                    response_body_canonical=cached_row.response_body_canonical,
                )
            runtime_object = self._call_network(
                endpoint=endpoint, request_kwargs=request_kwargs
            )
            response_row = runtime_object_to_response_row(
                request_row=request_row,
                runtime_object=runtime_object,
                source_mode="record",
            )
            self._storage.append_exchange(request_row, response_row)
            return runtime_object
        if self._config.cache_mode == "refresh":
            runtime_object = self._call_network(
                endpoint=endpoint, request_kwargs=request_kwargs
            )
            response_row = runtime_object_to_response_row(
                request_row=request_row,
                runtime_object=runtime_object,
                source_mode="refresh",
            )
            self._storage.append_exchange(request_row, response_row)
            return runtime_object
        if self._config.cache_mode == "deferred":
            self._storage.append_request(request_row)
            raise CacheDeferredRequest(
                endpoint=endpoint,
                request_hash=request_row.request_hash,
                manifest_path=self._storage.manifest_paths(endpoint).requests_path,
            )
        if self._config.cache_mode == "replay_only":
            return self._replay(
                endpoint=endpoint, request_hash=request_row.request_hash
            )
        raise CacheModeError(f"Unsupported cache mode: {self._config.cache_mode}")


class CacheAsyncOpenAI(_CacheClientBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: Any | None = None
        self.chat = _AsyncChatResource(self)
        self.embeddings = _AsyncEmbeddingsResource(self)

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = AsyncOpenAI(**self._client_kwargs)
        return self._client

    async def _call_network(
        self, *, endpoint: str, request_kwargs: dict[str, Any]
    ) -> Any:
        client = self._get_client()
        if endpoint == "chat.completions.create":
            return await client.chat.completions.create(**request_kwargs)
        if endpoint == "embeddings.create":
            return await client.embeddings.create(**request_kwargs)
        raise ValueError(f"Unsupported endpoint: {endpoint}")

    async def _request_async(
        self, *, endpoint: str, request_kwargs: dict[str, Any]
    ) -> Any:
        request_row = self._build_request_row(
            endpoint=endpoint, request_kwargs=request_kwargs
        )
        if self._config.cache_mode == "record":
            cached_row = self._lookup_cached_response(
                endpoint=endpoint, request_hash=request_row.request_hash
            )
            if cached_row is not None:
                return reconstruct_runtime_object(
                    endpoint=endpoint,
                    response_body_canonical=cached_row.response_body_canonical,
                )
            runtime_object = await self._call_network(
                endpoint=endpoint, request_kwargs=request_kwargs
            )
            response_row = runtime_object_to_response_row(
                request_row=request_row,
                runtime_object=runtime_object,
                source_mode="record",
            )
            self._storage.append_exchange(request_row, response_row)
            return runtime_object
        if self._config.cache_mode == "refresh":
            runtime_object = await self._call_network(
                endpoint=endpoint, request_kwargs=request_kwargs
            )
            response_row = runtime_object_to_response_row(
                request_row=request_row,
                runtime_object=runtime_object,
                source_mode="refresh",
            )
            self._storage.append_exchange(request_row, response_row)
            return runtime_object
        if self._config.cache_mode == "deferred":
            self._storage.append_request(request_row)
            raise CacheDeferredRequest(
                endpoint=endpoint,
                request_hash=request_row.request_hash,
                manifest_path=self._storage.manifest_paths(endpoint).requests_path,
            )
        if self._config.cache_mode == "replay_only":
            return self._replay(
                endpoint=endpoint, request_hash=request_row.request_hash
            )
        raise CacheModeError(f"Unsupported cache mode: {self._config.cache_mode}")

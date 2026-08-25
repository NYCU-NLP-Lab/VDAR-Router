from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from ._canonical import normalize_base_url
from ._constants import (
    DEFAULT_PROVIDER_NAME,
    ENDPOINT_DIRECTORY_NAMES,
    REQUESTS_FILE_NAME,
    RESPONSES_FILE_NAME,
)
from ._contracts import CanonicalRequestRow
from ._openai import runtime_object_to_response_row
from ._storage import CacheStorage


@dataclass(slots=True)
class DeferredRequestProcessingSummary:
    processed: int = 0
    skipped_existing: int = 0
    skipped_invalid: int = 0
    skipped_provider_mismatch: int = 0
    failed: int = 0
    failures: list[dict[str, str]] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "processed": self.processed,
            "skipped_existing": self.skipped_existing,
            "skipped_invalid": self.skipped_invalid,
            "skipped_provider_mismatch": self.skipped_provider_mismatch,
            "failed": self.failed,
        }
        if self.failures:
            payload["failures"] = list(self.failures)
        return payload


@dataclass(slots=True)
class _PreparedDeferredRequest:
    request_row: CanonicalRequestRow
    runtime_base_url: str
    runtime_request_kwargs: dict[str, Any]
    runtime_extra_body: dict[str, Any] | None


class DeferredRequestProcessingError(RuntimeError):
    def __init__(
        self,
        *,
        summary: DeferredRequestProcessingSummary,
        endpoint: str,
        request_hash: str,
        cause: Exception,
    ) -> None:
        super().__init__(
            f"Failed processing deferred request {endpoint} {request_hash}: {cause}"
        )
        self.summary = summary
        self.endpoint = endpoint
        self.request_hash = request_hash
        self.cause = cause


def _build_openai_client(**kwargs: Any) -> Any:
    return OpenAI(**kwargs)


def _compact_openai_client_kwargs(**kwargs: Any) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def _build_runtime_request(
    *,
    request_body_normalized: dict[str, Any],
    model: str | None,
    extra_body: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    request_kwargs = dict(request_body_normalized)
    recorded_extra_body = request_kwargs.pop("extra_body", None)
    if model is not None:
        request_kwargs["model"] = model
    runtime_extra_body = extra_body if extra_body is not None else recorded_extra_body
    return request_kwargs, runtime_extra_body


def _truncate_embedding_input_value(value: Any, *, input_length: int) -> Any:
    if input_length < 0:
        return value
    if isinstance(value, str):
        return value[:input_length]
    if isinstance(value, list):
        if all(isinstance(item, int) for item in value):
            return value[:input_length]
        return [
            _truncate_embedding_input_value(item, input_length=input_length)
            for item in value
        ]
    return value


def _apply_embedding_input_length_limit(
    request_kwargs: dict[str, Any], *, input_length: int
) -> dict[str, Any]:
    if input_length < 0 or "input" not in request_kwargs:
        return request_kwargs
    limited_request_kwargs = dict(request_kwargs)
    limited_request_kwargs["input"] = _truncate_embedding_input_value(
        limited_request_kwargs["input"],
        input_length=input_length,
    )
    return limited_request_kwargs


def _call_endpoint(
    *,
    client: Any,
    endpoint: str,
    request_kwargs: dict[str, Any],
    extra_body: dict[str, Any] | None,
) -> Any:
    create = None
    if endpoint == "chat.completions.create":
        create = client.chat.completions.create
    elif endpoint == "embeddings.create":
        create = client.embeddings.create
    else:
        raise ValueError(f"Unsupported endpoint: {endpoint}")
    if extra_body is None:
        return create(**request_kwargs)
    return create(**request_kwargs, extra_body=extra_body)


def _build_fulfillment_metadata(
    *,
    request_hash: str,
    runtime_base_url: str,
    base_url_override: str | None,
    model_override: str | None,
    extra_body_override: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "fulfilled_mode": "deferred",
        "fulfilled_from_request_hash": request_hash,
    }
    if base_url_override is not None:
        metadata["fulfilled_via_base_url"] = runtime_base_url
    if model_override is not None:
        metadata["fulfilled_via_model"] = model_override
    if extra_body_override is not None:
        metadata["fulfilled_via_extra_body"] = extra_body_override
    return metadata


def _build_request_rows_progress(
    *,
    storage: CacheStorage,
    requests_path: Path,
    endpoint: str,
    existing_request_hashes: set[str],
    limit: int | None,
    dry_run: bool,
) -> Any:
    total = _count_eligible_request_rows(
        storage=storage,
        requests_path=requests_path,
        endpoint=endpoint,
        existing_request_hashes=existing_request_hashes,
        limit=limit,
        simulate_successful_writes=not dry_run,
    )
    return tqdm(
        desc=f"Processing {endpoint}",
        unit="request",
        total=total,
        disable=None,
    )


def _prepare_deferred_request(
    *,
    request_row: CanonicalRequestRow,
    base_url: str | None,
    model: str | None,
    extra_body: dict[str, Any] | None,
    input_length: int,
) -> _PreparedDeferredRequest:
    runtime_base_url = normalize_base_url(base_url or request_row.provider_base_url)
    runtime_request_kwargs, runtime_extra_body = _build_runtime_request(
        request_body_normalized=request_row.request_body_normalized,
        model=model,
        extra_body=extra_body,
    )
    if request_row.endpoint == "embeddings.create":
        runtime_request_kwargs = _apply_embedding_input_length_limit(
            runtime_request_kwargs,
            input_length=input_length,
        )
    return _PreparedDeferredRequest(
        request_row=request_row,
        runtime_base_url=runtime_base_url,
        runtime_request_kwargs=runtime_request_kwargs,
        runtime_extra_body=runtime_extra_body,
    )


def _execute_deferred_request(
    *,
    prepared_request: _PreparedDeferredRequest,
    api_key: str | None,
    organization: str | None,
    project: str | None,
    timeout: Any,
    max_retries: int | None,
    default_headers: dict[str, str] | None,
    http_client: Any,
) -> Any:
    client = _build_openai_client(
        **_compact_openai_client_kwargs(
            api_key=api_key,
            base_url=prepared_request.runtime_base_url,
            organization=organization,
            project=project,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
        )
    )
    return _call_endpoint(
        client=client,
        endpoint=prepared_request.request_row.endpoint,
        request_kwargs=prepared_request.runtime_request_kwargs,
        extra_body=prepared_request.runtime_extra_body,
    )


def _append_runtime_response(
    *,
    storage: CacheStorage,
    responses_path: Path,
    prepared_request: _PreparedDeferredRequest,
    runtime_object: Any,
    base_url_override: str | None,
    model_override: str | None,
    extra_body_override: dict[str, Any] | None,
) -> None:
    storage.append_response_to_path(
        responses_path,
        runtime_object_to_response_row(
            request_row=prepared_request.request_row,
            runtime_object=runtime_object,
            source_mode="deferred",
            provider_response_metadata_updates=_build_fulfillment_metadata(
                request_hash=prepared_request.request_row.request_hash,
                runtime_base_url=prepared_request.runtime_base_url,
                base_url_override=base_url_override,
                model_override=model_override,
                extra_body_override=extra_body_override,
            ),
        ),
    )


def _record_failure(
    *,
    summary: DeferredRequestProcessingSummary,
    endpoint: str,
    request_hash: str,
    cause: Exception,
) -> None:
    summary.failed += 1
    if summary.failures is None:
        summary.failures = []
    summary.failures.append(
        {
            "endpoint": endpoint,
            "request_hash": request_hash,
            "error": str(cause),
        }
    )


def _classify_request_row(
    *,
    request_row: CanonicalRequestRow | None,
    endpoint: str,
    summary: DeferredRequestProcessingSummary,
    existing_request_hashes: set[str],
) -> CanonicalRequestRow | None:
    if request_row is None:
        summary.skipped_invalid += 1
        return None
    if request_row.endpoint != endpoint:
        summary.skipped_invalid += 1
        return None
    if request_row.provider_name != DEFAULT_PROVIDER_NAME:
        summary.skipped_provider_mismatch += 1
        return None
    if request_row.request_hash in existing_request_hashes:
        summary.skipped_existing += 1
        return None
    return request_row


def _count_eligible_request_rows(
    *,
    storage: CacheStorage,
    requests_path: Path,
    endpoint: str,
    existing_request_hashes: set[str],
    limit: int | None,
    simulate_successful_writes: bool,
) -> int:
    if limit is not None and limit <= 0:
        return 0

    eligible_count = 0
    count_summary = DeferredRequestProcessingSummary()
    seen_request_hashes = set(existing_request_hashes)

    for request_row in storage.iter_request_row_records_from_path(requests_path):
        eligible_row = _classify_request_row(
            request_row=request_row,
            endpoint=endpoint,
            summary=count_summary,
            existing_request_hashes=seen_request_hashes,
        )
        if eligible_row is None:
            continue
        eligible_count += 1
        if simulate_successful_writes:
            seen_request_hashes.add(eligible_row.request_hash)
        if limit is not None and eligible_count >= limit:
            break

    return eligible_count


def _advance_request_rows_progress(progress: Any) -> None:
    total = getattr(progress, "total", None)
    current = getattr(progress, "n", None)
    if isinstance(total, int) and isinstance(current, int) and current >= total:
        progress.total = current + 1
        refresh = getattr(progress, "refresh", None)
        if callable(refresh):
            refresh()
    progress.update(1)


def _consume_skips_until_next_eligible(
    *,
    request_rows_iter: Any,
    pending_rows: deque[CanonicalRequestRow],
    endpoint: str,
    summary: DeferredRequestProcessingSummary,
    existing_request_hashes: set[str],
    exhausted: bool,
) -> bool:
    while True:
        request_row: CanonicalRequestRow | None
        if pending_rows:
            request_row = pending_rows.popleft()
        else:
            if exhausted:
                return True
            try:
                request_row = next(request_rows_iter)
            except StopIteration:
                return True

        eligible_row = _classify_request_row(
            request_row=request_row,
            endpoint=endpoint,
            summary=summary,
            existing_request_hashes=existing_request_hashes,
        )
        if eligible_row is None:
            continue
        pending_rows.appendleft(eligible_row)
        return False


def _process_deferred_requests_serial(
    *,
    request_rows: Any,
    storage: CacheStorage,
    responses_path: Path,
    endpoint: str,
    summary: DeferredRequestProcessingSummary,
    api_key: str | None,
    base_url: str | None,
    organization: str | None,
    project: str | None,
    timeout: Any,
    max_retries: int | None,
    default_headers: dict[str, str] | None,
    http_client: Any,
    model: str | None,
    extra_body: dict[str, Any] | None,
    existing_request_hashes: set[str],
    progress: Any,
    dry_run: bool,
    limit: int | None,
    keep_going: bool,
    input_length: int,
) -> DeferredRequestProcessingSummary:
    for request_row in request_rows:
        eligible_row = _classify_request_row(
            request_row=request_row,
            endpoint=endpoint,
            summary=summary,
            existing_request_hashes=existing_request_hashes,
        )
        if eligible_row is None:
            continue
        if limit is not None and summary.processed >= limit:
            return summary
        prepared_request = _prepare_deferred_request(
            request_row=eligible_row,
            base_url=base_url,
            model=model,
            extra_body=extra_body,
            input_length=input_length,
        )
        _advance_request_rows_progress(progress)
        if dry_run:
            summary.processed += 1
            continue
        try:
            runtime_object = _execute_deferred_request(
                prepared_request=prepared_request,
                api_key=api_key,
                organization=organization,
                project=project,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
                http_client=http_client,
            )
            _append_runtime_response(
                storage=storage,
                responses_path=responses_path,
                prepared_request=prepared_request,
                runtime_object=runtime_object,
                base_url_override=base_url,
                model_override=model,
                extra_body_override=extra_body,
            )
            existing_request_hashes.add(eligible_row.request_hash)
        except Exception as exc:
            _record_failure(
                summary=summary,
                endpoint=endpoint,
                request_hash=eligible_row.request_hash,
                cause=exc,
            )
            if not keep_going:
                raise DeferredRequestProcessingError(
                    summary=summary,
                    endpoint=endpoint,
                    request_hash=eligible_row.request_hash,
                    cause=exc,
                ) from exc
            continue
        summary.processed += 1
    return summary


def _process_deferred_requests_concurrently(
    *,
    request_rows: Any,
    storage: CacheStorage,
    responses_path: Path,
    endpoint: str,
    summary: DeferredRequestProcessingSummary,
    api_key: str | None,
    base_url: str | None,
    organization: str | None,
    project: str | None,
    timeout: Any,
    max_retries: int | None,
    default_headers: dict[str, str] | None,
    http_client: Any,
    model: str | None,
    extra_body: dict[str, Any] | None,
    existing_request_hashes: set[str],
    progress: Any,
    limit: int | None,
    workers: int,
    input_length: int,
) -> DeferredRequestProcessingSummary:
    request_rows_iter = iter(request_rows)
    pending_rows: deque[CanonicalRequestRow] = deque()
    exhausted = False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            remaining_capacity = workers
            if limit is not None:
                remaining_capacity = min(remaining_capacity, limit - summary.processed)
            if remaining_capacity <= 0:
                exhausted = _consume_skips_until_next_eligible(
                    request_rows_iter=request_rows_iter,
                    pending_rows=pending_rows,
                    endpoint=endpoint,
                    summary=summary,
                    existing_request_hashes=existing_request_hashes,
                    exhausted=exhausted,
                )
                return summary

            prepared_batch: list[_PreparedDeferredRequest] = []
            scheduled_hashes: set[str] = set()
            deferred_rows: list[CanonicalRequestRow] = []

            while len(prepared_batch) < remaining_capacity:
                request_row: CanonicalRequestRow | None
                if pending_rows:
                    request_row = pending_rows.popleft()
                else:
                    if exhausted:
                        break
                    try:
                        request_row = next(request_rows_iter)
                    except StopIteration:
                        exhausted = True
                        break

                eligible_row = _classify_request_row(
                    request_row=request_row,
                    endpoint=endpoint,
                    summary=summary,
                    existing_request_hashes=existing_request_hashes,
                )
                if eligible_row is None:
                    continue
                if eligible_row.request_hash in scheduled_hashes:
                    deferred_rows.append(eligible_row)
                    continue

                prepared_batch.append(
                    _prepare_deferred_request(
                        request_row=eligible_row,
                        base_url=base_url,
                        model=model,
                        extra_body=extra_body,
                        input_length=input_length,
                    )
                )
                scheduled_hashes.add(eligible_row.request_hash)

            pending_rows.extend(deferred_rows)

            if not prepared_batch:
                if exhausted and not pending_rows:
                    return summary
                continue

            for _ in prepared_batch:
                _advance_request_rows_progress(progress)

            future_to_request = {
                executor.submit(
                    _execute_deferred_request,
                    prepared_request=prepared_request,
                    api_key=api_key,
                    organization=organization,
                    project=project,
                    timeout=timeout,
                    max_retries=max_retries,
                    default_headers=default_headers,
                    http_client=http_client,
                ): prepared_request
                for prepared_request in prepared_batch
            }
            for future in as_completed(future_to_request):
                prepared_request = future_to_request[future]
                try:
                    runtime_object = future.result()
                except Exception as exc:
                    _record_failure(
                        summary=summary,
                        endpoint=endpoint,
                        request_hash=prepared_request.request_row.request_hash,
                        cause=exc,
                    )
                    continue

                _append_runtime_response(
                    storage=storage,
                    responses_path=responses_path,
                    prepared_request=prepared_request,
                    runtime_object=runtime_object,
                    base_url_override=base_url,
                    model_override=model,
                    extra_body_override=extra_body,
                )
                existing_request_hashes.add(prepared_request.request_row.request_hash)
                summary.processed += 1

    return summary


def _endpoint_from_requests_path(requests_path: Path) -> str:
    if requests_path.name != REQUESTS_FILE_NAME:
        raise ValueError(
            f"Deferred request manifest must be named {REQUESTS_FILE_NAME}."
        )
    endpoint_dir_name = requests_path.parent.name
    for endpoint, directory_name in ENDPOINT_DIRECTORY_NAMES.items():
        if directory_name == endpoint_dir_name:
            return endpoint
    raise ValueError(
        "Deferred request manifest must live in a supported endpoint directory."
    )


def process_deferred_requests(
    *,
    requests_path: str | Path,
    api_key: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
    project: str | None = None,
    timeout: Any = None,
    max_retries: int | None = None,
    default_headers: dict[str, str] | None = None,
    http_client: Any = None,
    model: str | None = None,
    extra_body: dict[str, Any] | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    keep_going: bool = False,
    workers: int = 1,
    input_length: int = -1,
) -> DeferredRequestProcessingSummary:
    if workers <= 0:
        raise ValueError("workers must be a positive integer.")
    if input_length < -1:
        raise ValueError("input_length must be -1 or a non-negative integer.")
    requests_file_path = Path(requests_path)
    responses_path = requests_file_path.with_name(RESPONSES_FILE_NAME)
    endpoint = _endpoint_from_requests_path(requests_file_path)
    storage = CacheStorage(requests_file_path.parent.parent)
    summary = DeferredRequestProcessingSummary()
    existing_request_hashes = storage.response_request_hashes_from_path(
        responses_path,
        endpoint=endpoint,
    )

    request_rows = _build_request_rows_progress(
        storage=storage,
        requests_path=requests_file_path,
        endpoint=endpoint,
        existing_request_hashes=existing_request_hashes,
        limit=limit,
        dry_run=dry_run,
    )
    raw_request_rows = storage.iter_request_row_records_from_path(requests_file_path)
    try:
        if dry_run or workers == 1 or not keep_going:
            return _process_deferred_requests_serial(
                request_rows=raw_request_rows,
                storage=storage,
                responses_path=responses_path,
                endpoint=endpoint,
                summary=summary,
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                project=project,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
                http_client=http_client,
                model=model,
                extra_body=extra_body,
                existing_request_hashes=set(existing_request_hashes),
                progress=request_rows,
                dry_run=dry_run,
                limit=limit,
                keep_going=keep_going,
                input_length=input_length,
            )
        return _process_deferred_requests_concurrently(
            request_rows=raw_request_rows,
            storage=storage,
            responses_path=responses_path,
            endpoint=endpoint,
            summary=summary,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
            model=model,
            extra_body=extra_body,
            existing_request_hashes=set(existing_request_hashes),
            progress=request_rows,
            limit=limit,
            workers=workers,
            input_length=input_length,
        )
    finally:
        request_rows.close()

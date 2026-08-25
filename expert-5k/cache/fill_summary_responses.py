from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._canonical import normalize_base_url
from ._constants import DEFAULT_PROVIDER_NAME, REQUESTS_FILE_NAME, RESPONSES_FILE_NAME
from ._contracts import CanonicalRequestRow
from ._openai import runtime_object_to_response_row
from ._processor import (
    _build_openai_client,
    _build_runtime_request,
    _call_endpoint,
    _compact_openai_client_kwargs,
)
from ._storage import CacheStorage
from .settings import get_settings

RELEVANT_ENDPOINT = "chat.completions.create"
SUMMARY_PATTERN = re.compile(r"<summary>\s*(.*?)\s*</summary>", re.DOTALL)


@dataclass(slots=True)
class SummaryRefillSummary:
    total_rows: int = 0
    checked_rows: int = 0
    processed: int = 0
    skipped_existing: int = 0
    skipped_duplicate: int = 0
    skipped_non_chat: int = 0
    skipped_invalid: int = 0
    skipped_provider_mismatch: int = 0
    invalid_response_attempts: int = 0
    failed: int = 0
    invalid_entries: list[dict[str, Any]] | None = None
    failures: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "total_rows": self.total_rows,
            "checked_rows": self.checked_rows,
            "processed": self.processed,
            "skipped_existing": self.skipped_existing,
            "skipped_duplicate": self.skipped_duplicate,
            "skipped_non_chat": self.skipped_non_chat,
            "skipped_invalid": self.skipped_invalid,
            "skipped_provider_mismatch": self.skipped_provider_mismatch,
            "invalid_response_attempts": self.invalid_response_attempts,
            "failed": self.failed,
        }
        if self.invalid_entries:
            payload["invalid_entries"] = list(self.invalid_entries)
        if self.failures:
            payload["failures"] = list(self.failures)
        return payload


class SummaryRefillError(RuntimeError):
    def __init__(
        self,
        *,
        summary: SummaryRefillSummary,
        request_hash: str,
        cause: Exception,
    ) -> None:
        super().__init__(
            f"Failed refilling summary response for {request_hash}: {cause}"
        )
        self.summary = summary
        self.request_hash = request_hash
        self.cause = cause


class _RequestFillFailure(RuntimeError):
    def __init__(self, message: str, *, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cache.fill_summary_responses",
        description=(
            "Fill only chat-completions cache responses whose latest cached "
            "response is missing a valid <summary>...</summary> block"
        ),
    )
    parser.add_argument("requests_path")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--extra-body",
        default=None,
        help="JSON object string passed to the provider as extra_body.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--summary-attempt-limit",
        type=int,
        default=0,
        help="Maximum API calls per request hash. Use 0 for unlimited retries.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _parse_json_object(raw: str, *, flag_name: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{flag_name} must decode to a JSON object.")
    return value


def _validate_requests_path(raw: str) -> Path:
    path = Path(raw)
    if path.name != REQUESTS_FILE_NAME:
        raise ValueError("requests_path must point to a requests.jsonl file.")
    if not path.exists():
        raise ValueError("requests_path does not exist.")
    if not path.is_file():
        raise ValueError("requests_path must point to a file.")
    return path


def _print_summary(
    *, requests_path: str, dry_run: bool, summary: dict[str, Any]
) -> None:
    print(
        json.dumps(
            {
                "requests_path": requests_path,
                "dry_run": dry_run,
                **summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _record_invalid_entry(
    *,
    summary: SummaryRefillSummary,
    line_number: int,
    request_hash: str | None,
    error: str,
) -> None:
    summary.skipped_invalid += 1
    if summary.invalid_entries is None:
        summary.invalid_entries = []
    summary.invalid_entries.append(
        {
            "line_number": line_number,
            "request_hash": request_hash,
            "error": error,
        }
    )


def _record_failure(
    *,
    summary: SummaryRefillSummary,
    line_number: int,
    request_hash: str,
    attempts: int,
    error: str,
) -> None:
    summary.failed += 1
    if summary.failures is None:
        summary.failures = []
    summary.failures.append(
        {
            "line_number": line_number,
            "endpoint": RELEVANT_ENDPOINT,
            "request_hash": request_hash,
            "attempts": attempts,
            "error": error,
        }
    )


def _response_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        return "\n".join(text_parts)
    return ""


def _response_has_valid_summary(response_body_canonical: dict[str, Any]) -> bool:
    choices = response_body_canonical.get("choices")
    if not isinstance(choices, list):
        return False

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        content = _response_content_to_text(message.get("content"))
        for match in SUMMARY_PATTERN.findall(content):
            if match.strip():
                return True
    return False


def _build_summary_refill_metadata(
    *,
    request_hash: str,
    runtime_base_url: str,
    base_url_override: str | None,
    model_override: str | None,
    extra_body_override: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "fulfilled_mode": "summary_refill",
        "fulfilled_from_request_hash": request_hash,
    }
    if base_url_override is not None:
        metadata["fulfilled_via_base_url"] = runtime_base_url
    if model_override is not None:
        metadata["fulfilled_via_model"] = model_override
    if extra_body_override is not None:
        metadata["fulfilled_via_extra_body"] = extra_body_override
    return metadata


def _try_fill_request_hash(
    *,
    storage: CacheStorage,
    responses_path: Path,
    request_row: CanonicalRequestRow,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
    extra_body: dict[str, Any] | None,
    dry_run: bool,
    summary: SummaryRefillSummary,
    summary_attempt_limit: int | None,
) -> int:
    runtime_base_url = normalize_base_url(base_url or request_row.provider_base_url)
    runtime_request_kwargs, runtime_extra_body = _build_runtime_request(
        request_body_normalized=request_row.request_body_normalized,
        model=model,
        extra_body=extra_body,
    )

    if dry_run:
        return 0

    attempts = 0
    while summary_attempt_limit is None or attempts < summary_attempt_limit:
        attempts += 1
        try:
            client = _build_openai_client(
                **_compact_openai_client_kwargs(
                    api_key=api_key,
                    base_url=runtime_base_url,
                )
            )
            runtime_object = _call_endpoint(
                client=client,
                endpoint=RELEVANT_ENDPOINT,
                request_kwargs=runtime_request_kwargs,
                extra_body=runtime_extra_body,
            )
            response_row = runtime_object_to_response_row(
                request_row=request_row,
                runtime_object=runtime_object,
                source_mode="summary_refill",
                provider_response_metadata_updates=_build_summary_refill_metadata(
                    request_hash=request_row.request_hash,
                    runtime_base_url=runtime_base_url,
                    base_url_override=base_url,
                    model_override=model,
                    extra_body_override=extra_body,
                ),
            )
        except Exception as exc:
            if summary_attempt_limit is None or attempts < summary_attempt_limit:
                continue
            raise _RequestFillFailure(str(exc), attempts=attempts) from exc

        if not _response_has_valid_summary(response_row.response_body_canonical):
            summary.invalid_response_attempts += 1
            continue

        storage.append_response_to_path(responses_path, response_row)
        return attempts

    raise _RequestFillFailure(
        "summary attempt limit reached without a valid non-empty "
        "<summary>...</summary> block",
        attempts=attempts,
    )


def fill_summary_responses(
    *,
    requests_path: str | Path,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    extra_body: dict[str, Any] | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    keep_going: bool = False,
    summary_attempt_limit: int | None = None,
) -> SummaryRefillSummary:
    requests_file_path = Path(requests_path)
    responses_path = requests_file_path.with_name(RESPONSES_FILE_NAME)
    storage = CacheStorage(requests_file_path.parent.parent)
    summary = SummaryRefillSummary()
    handled_request_hashes: set[str] = set()

    with requests_file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw = raw_line.strip()
            if not raw:
                continue

            summary.total_rows += 1

            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                _record_invalid_entry(
                    summary=summary,
                    line_number=line_number,
                    request_hash=None,
                    error=f"invalid JSON: {exc.msg}",
                )
                continue

            if not isinstance(data, dict):
                _record_invalid_entry(
                    summary=summary,
                    line_number=line_number,
                    request_hash=None,
                    error="row must be an object",
                )
                continue

            request_hash = data.get("request_hash")
            try:
                request_row = CanonicalRequestRow.from_dict(data)
            except (KeyError, TypeError, ValueError) as exc:
                _record_invalid_entry(
                    summary=summary,
                    line_number=line_number,
                    request_hash=request_hash
                    if isinstance(request_hash, str)
                    else None,
                    error=str(exc),
                )
                continue

            if request_row.endpoint != RELEVANT_ENDPOINT:
                summary.skipped_non_chat += 1
                continue

            summary.checked_rows += 1

            if request_row.request_hash in handled_request_hashes:
                summary.skipped_duplicate += 1
                continue
            handled_request_hashes.add(request_row.request_hash)

            if request_row.provider_name != DEFAULT_PROVIDER_NAME:
                summary.skipped_provider_mismatch += 1
                continue

            latest_response = storage.find_response_row_from_path(
                responses_path,
                endpoint=RELEVANT_ENDPOINT,
                request_hash=request_row.request_hash,
            )
            if latest_response is not None and _response_has_valid_summary(
                latest_response.response_body_canonical
            ):
                summary.skipped_existing += 1
                continue

            if limit is not None and summary.processed >= limit:
                break

            attempts = 0
            try:
                attempts = _try_fill_request_hash(
                    storage=storage,
                    responses_path=responses_path,
                    request_row=request_row,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    extra_body=extra_body,
                    dry_run=dry_run,
                    summary=summary,
                    summary_attempt_limit=summary_attempt_limit,
                )
                summary.processed += 1
            except Exception as exc:
                _record_failure(
                    summary=summary,
                    line_number=line_number,
                    request_hash=request_row.request_hash,
                    attempts=(
                        exc.attempts
                        if isinstance(exc, _RequestFillFailure)
                        else attempts or (summary_attempt_limit or 0)
                    ),
                    error=str(exc),
                )
                if not keep_going:
                    raise SummaryRefillError(
                        summary=summary,
                        request_hash=request_row.request_hash,
                        cause=exc,
                    ) from exc

    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    requests_path_raw = args.requests_path
    try:
        requests_path = _validate_requests_path(requests_path_raw)
        extra_body = None
        if args.extra_body is not None:
            extra_body = _parse_json_object(args.extra_body, flag_name="--extra-body")
        if args.summary_attempt_limit < 0:
            raise ValueError(
                "--summary-attempt-limit must be zero or a positive integer."
            )
    except (json.JSONDecodeError, ValueError) as exc:
        _print_summary(
            requests_path=requests_path_raw,
            dry_run=args.dry_run,
            summary={
                **SummaryRefillSummary().to_dict(),
                "error": str(exc),
            },
        )
        return 1

    try:
        summary = fill_summary_responses(
            requests_path=requests_path,
            api_key=args.api_key or settings.llm_api_key,
            base_url=args.base_url or settings.llm_base_url,
            model=args.model,
            extra_body=extra_body,
            dry_run=args.dry_run,
            limit=args.limit,
            keep_going=args.keep_going,
            summary_attempt_limit=(
                None if args.summary_attempt_limit == 0 else args.summary_attempt_limit
            ),
        )
    except SummaryRefillError as exc:
        _print_summary(
            requests_path=str(requests_path),
            dry_run=args.dry_run,
            summary={
                **exc.summary.to_dict(),
                "error": str(exc),
            },
        )
        return 1

    _print_summary(
        requests_path=str(requests_path),
        dry_run=args.dry_run,
        summary=summary.to_dict(),
    )
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

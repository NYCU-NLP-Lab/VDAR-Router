from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ._processor import (
    DeferredRequestProcessingError,
    DeferredRequestProcessingSummary,
    process_deferred_requests,
)
from .settings import get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cache.process_requests",
        description="Fulfill a deferred requests.jsonl manifest into sibling responses.jsonl",
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--input-length",
        type=int,
        default=-1,
        help=(
            "Limit deferred embeddings input length before fulfillment. "
            "Use -1 to disable truncation."
        ),
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
    if path.name != "requests.jsonl":
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    requests_path_raw = args.requests_path
    try:
        requests_path = _validate_requests_path(requests_path_raw)
        extra_body = None
        if args.extra_body is not None:
            extra_body = _parse_json_object(args.extra_body, flag_name="--extra-body")
        if args.workers <= 0:
            raise ValueError("--workers must be a positive integer.")
        if args.input_length < -1:
            raise ValueError("--input-length must be -1 or a non-negative integer.")
    except (json.JSONDecodeError, ValueError) as exc:
        _print_summary(
            requests_path=requests_path_raw,
            dry_run=args.dry_run,
            summary={
                **DeferredRequestProcessingSummary().to_dict(),
                "error": str(exc),
            },
        )
        return 1
    base_url = args.base_url or settings.llm_base_url
    api_key = args.api_key or settings.llm_api_key
    try:
        summary = process_deferred_requests(
            requests_path=requests_path,
            api_key=api_key,
            base_url=base_url,
            model=args.model,
            extra_body=extra_body,
            dry_run=args.dry_run,
            limit=args.limit,
            keep_going=args.keep_going,
            workers=args.workers,
            input_length=args.input_length,
        )
    except DeferredRequestProcessingError as exc:
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

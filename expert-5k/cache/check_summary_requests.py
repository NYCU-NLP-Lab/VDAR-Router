from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from ._contracts import CanonicalRequestRow

SUMMARY_START = "<summary>"
SUMMARY_END = "</summary>"
RELEVANT_ENDPOINT = "chat.completions.create"
SUMMARY_BLOCK_PATTERN = re.compile(r"<summary>.*?</summary>", re.DOTALL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cache.check_summary_requests",
        description="Check requests.jsonl rows for summary-block instructions",
    )
    parser.add_argument("requests_path")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _validate_requests_path(raw: str) -> Path:
    path = Path(raw)
    if path.name != "requests.jsonl":
        raise ValueError("requests_path must point to a requests.jsonl file.")
    if not path.exists():
        raise ValueError("requests_path does not exist.")
    if not path.is_file():
        raise ValueError("requests_path must point to a file.")
    return path


def _print_summary(*, requests_path: str, summary: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "requests_path": requests_path,
                **summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _iter_request_rows(
    requests_path: Path,
) -> tuple[int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    total_rows = 0
    invalid_rows = 0
    invalid_entries: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []

    with requests_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw = raw_line.strip()
            if not raw:
                continue
            total_rows += 1
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                invalid_rows += 1
                invalid_entries.append(
                    {
                        "line_number": line_number,
                        "request_hash": None,
                        "error": f"invalid JSON: {exc.msg}",
                    }
                )
                continue
            if not isinstance(data, dict):
                invalid_rows += 1
                invalid_entries.append(
                    {
                        "line_number": line_number,
                        "request_hash": None,
                        "error": "row must be an object",
                    }
                )
                continue
            try:
                row = CanonicalRequestRow.from_dict(data)
            except (KeyError, TypeError, ValueError) as exc:
                invalid_rows += 1
                invalid_entries.append(
                    {
                        "line_number": line_number,
                        "request_hash": data.get("request_hash"),
                        "error": str(exc),
                    }
                )
                continue
            request_rows.append(
                {
                    "line_number": line_number,
                    "row": row,
                }
            )

    return total_rows, invalid_rows, invalid_entries, request_rows


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def _row_has_summary_block(row: CanonicalRequestRow) -> bool:
    messages = row.request_body_normalized.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("messages items must be objects")
        if SUMMARY_BLOCK_PATTERN.search(_message_text(message)):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    requests_path_raw = args.requests_path
    try:
        requests_path = _validate_requests_path(requests_path_raw)
    except ValueError as exc:
        _print_summary(
            requests_path=requests_path_raw,
            summary={
                "total_rows": 0,
                "checked_rows": 0,
                "invalid_rows": 0,
                "missing_summary_rows": 0,
                "invalid_entries": [],
                "missing_summary_entries": [],
                "error": str(exc),
            },
        )
        return 1

    total_rows, invalid_rows, invalid_entries, request_rows = _iter_request_rows(
        requests_path
    )

    checked_rows = 0
    missing_summary_entries: list[dict[str, Any]] = []
    missing_summary_rows = 0

    for entry in request_rows:
        row = entry["row"]
        if row.endpoint != RELEVANT_ENDPOINT:
            continue
        checked_rows += 1
        try:
            has_summary_block = _row_has_summary_block(row)
        except ValueError as exc:
            invalid_rows += 1
            invalid_entries.append(
                {
                    "line_number": entry["line_number"],
                    "request_hash": row.request_hash,
                    "error": str(exc),
                }
            )
            continue
        if has_summary_block:
            continue
        missing_summary_rows += 1
        missing_summary_entries.append(
            {
                "line_number": entry["line_number"],
                "request_hash": row.request_hash,
                "endpoint": row.endpoint,
                "error": "missing <summary>...</summary> block",
            }
        )

    summary = {
        "total_rows": total_rows,
        "checked_rows": checked_rows,
        "invalid_rows": invalid_rows,
        "missing_summary_rows": missing_summary_rows,
        "invalid_entries": invalid_entries,
        "missing_summary_entries": missing_summary_entries,
    }
    if invalid_rows or missing_summary_rows:
        summary["error"] = "summary block check failed"
        _print_summary(requests_path=str(requests_path), summary=summary)
        return 1

    _print_summary(requests_path=str(requests_path), summary=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

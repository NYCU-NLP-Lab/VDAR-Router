from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ._constants import (
    ENDPOINT_DIRECTORY_NAMES,
    REQUESTS_FILE_NAME,
    RESPONSES_FILE_NAME,
    SUPPORTED_ENDPOINTS,
)
from ._contracts import CanonicalRequestRow, CanonicalResponseRow, ManifestPaths

_ResponseIndexKey = tuple[str, str]


def build_manifest_paths(root_dir: Path, endpoint: str) -> ManifestPaths:
    endpoint_dir = root_dir / ENDPOINT_DIRECTORY_NAMES[endpoint]
    return ManifestPaths(
        root_dir=endpoint_dir,
        requests_path=endpoint_dir / REQUESTS_FILE_NAME,
        responses_path=endpoint_dir / RESPONSES_FILE_NAME,
    )


class CacheStorage:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self._response_indexes_by_path: dict[
            Path, dict[_ResponseIndexKey, CanonicalResponseRow]
        ] = {}

    def manifest_paths(self, endpoint: str) -> ManifestPaths:
        return build_manifest_paths(self.root_dir, endpoint)

    def ensure_base_layout(self) -> None:
        for endpoint in SUPPORTED_ENDPOINTS:
            manifest_paths = self.manifest_paths(endpoint)
            manifest_paths.root_dir.mkdir(parents=True, exist_ok=True)
            manifest_paths.requests_path.touch(exist_ok=True)
            manifest_paths.responses_path.touch(exist_ok=True)

    def append_request(self, row: CanonicalRequestRow) -> None:
        self._append_jsonl(
            self.manifest_paths(row.endpoint).requests_path, row.to_dict()
        )

    def append_response(self, row: CanonicalResponseRow) -> None:
        self.append_response_to_path(
            self.manifest_paths(row.endpoint).responses_path, row
        )

    def append_exchange(
        self, request_row: CanonicalRequestRow, response_row: CanonicalResponseRow
    ) -> None:
        self.append_request(request_row)
        self.append_response(response_row)

    def iter_request_rows(self, endpoint: str) -> Iterator[CanonicalRequestRow]:
        for row in self.iter_request_row_records(endpoint):
            if row is None:
                continue
            yield row

    def iter_request_row_records(
        self, endpoint: str
    ) -> Iterator[CanonicalRequestRow | None]:
        yield from self.iter_request_row_records_from_path(
            self.manifest_paths(endpoint).requests_path
        )

    def find_response_row(
        self, *, endpoint: str, request_hash: str
    ) -> CanonicalResponseRow | None:
        return self.find_response_row_from_path(
            self.manifest_paths(endpoint).responses_path,
            endpoint=endpoint,
            request_hash=request_hash,
        )

    def iter_request_row_records_from_path(
        self, requests_path: Path
    ) -> Iterator[CanonicalRequestRow | None]:
        for data in self._iter_jsonl_entries(requests_path):
            if data is None:
                yield None
                continue
            try:
                yield CanonicalRequestRow.from_dict(data)
            except (KeyError, TypeError, ValueError):
                yield None

    def iter_response_row_records_from_path(
        self, responses_path: Path
    ) -> Iterator[CanonicalResponseRow | None]:
        for data in self._iter_jsonl_entries(responses_path):
            if data is None:
                yield None
                continue
            try:
                yield CanonicalResponseRow.from_dict(data)
            except (KeyError, TypeError, ValueError):
                yield None

    def append_response_to_path(
        self, responses_path: Path, row: CanonicalResponseRow
    ) -> None:
        self._append_jsonl(responses_path, row.to_dict())
        response_index = self._response_indexes_by_path.get(responses_path)
        if response_index is not None:
            response_index[self._build_replay_key(row.endpoint, row.request_hash)] = row

    def response_index_from_path(
        self, responses_path: Path
    ) -> dict[_ResponseIndexKey, CanonicalResponseRow]:
        response_index = self._response_indexes_by_path.get(responses_path)
        if response_index is None:
            response_index = self._build_response_index_from_path(responses_path)
            self._response_indexes_by_path[responses_path] = response_index
        return response_index

    def response_request_hashes_from_path(
        self, responses_path: Path, *, endpoint: str
    ) -> set[str]:
        return {
            request_hash
            for replay_key in self.response_index_from_path(responses_path)
            for replay_endpoint, request_hash in [replay_key]
            if replay_endpoint == endpoint
        }

    def count_jsonl_rows_from_path(self, path: Path) -> int:
        if not path.exists():
            return 0
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count

    def find_response_row_from_path(
        self, responses_path: Path, *, endpoint: str, request_hash: str
    ) -> CanonicalResponseRow | None:
        return self.response_index_from_path(responses_path).get(
            self._build_replay_key(endpoint, request_hash)
        )

    def _build_response_index_from_path(
        self, responses_path: Path
    ) -> dict[_ResponseIndexKey, CanonicalResponseRow]:
        response_index: dict[_ResponseIndexKey, CanonicalResponseRow] = {}
        for row in self.iter_response_row_records_from_path(responses_path):
            if row is None:
                continue
            response_index[self._build_replay_key(row.endpoint, row.request_hash)] = row
        return response_index

    def _build_replay_key(self, endpoint: str, request_hash: str) -> _ResponseIndexKey:
        return (endpoint, request_hash)

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with path.open("ab") as handle:
            handle.write(serialized.encode("utf-8"))

    def _iter_jsonl_entries(self, path: Path) -> Iterator[dict[str, Any] | None]:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    yield None
                    continue
                if isinstance(data, dict):
                    yield data
                    continue
                yield None

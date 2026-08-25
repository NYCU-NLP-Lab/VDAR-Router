from __future__ import annotations

from pathlib import Path


class CacheModeError(ValueError):
    pass


class CacheDeferredRequest(RuntimeError):
    def __init__(
        self, *, endpoint: str, request_hash: str, manifest_path: Path
    ) -> None:
        self.endpoint = endpoint
        self.request_hash = request_hash
        self.manifest_path = manifest_path
        super().__init__(
            f"Deferred cache request recorded for {endpoint} ({request_hash}) at {manifest_path}"
        )


class CacheReplayMiss(LookupError):
    def __init__(
        self, *, endpoint: str, request_hash: str, manifest_path: Path
    ) -> None:
        self.endpoint = endpoint
        self.request_hash = request_hash
        self.manifest_path = manifest_path
        super().__init__(
            f"No cached response found for {endpoint} ({request_hash}) in {manifest_path}"
        )

from __future__ import annotations

from pathlib import Path


def ensure_safe_router_key(router_key: str) -> str:
    normalized = router_key.strip()
    if not normalized:
        raise ValueError("router_key must be a non-empty string.")
    if normalized in {".", ".."}:
        raise ValueError("router_key must not be '.' or '..'.")
    if Path(normalized).is_absolute():
        raise ValueError("router_key must not be an absolute path.")
    if any(separator in normalized for separator in ("/", "\\")):
        raise ValueError("router_key must not contain path separators.")
    return normalized


def resolve_under_root(root: Path, *parts: str) -> Path:
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Resolved path '{candidate}' escapes expected root '{resolved_root}'."
        ) from exc
    return candidate

from __future__ import annotations

SCHEMA_VERSION = "cache.v1"
CANONICALIZATION_VERSION = "v1"
REQUEST_HASH_ALGORITHM = "sha256"
DEFAULT_CACHE_DIR = ".cache"
DEFAULT_PROVIDER_NAME = "openai"
DEFAULT_PROVIDER_BASE_URL = "https://api.openai.com/v1"
REQUESTS_FILE_NAME = "requests.jsonl"
RESPONSES_FILE_NAME = "responses.jsonl"

SUPPORTED_CACHE_MODES = (
    "record",
    "refresh",
    "deferred",
    "replay_only",
)

SUPPORTED_ENDPOINTS = (
    "chat.completions.create",
    "embeddings.create",
)

TRANSPORT_ONLY_REQUEST_FIELDS = frozenset(
    {
        "timeout",
        "extra_headers",
        "extra_query",
        "stream",
        "stream_options",
    }
)

SENSITIVE_KEY_NAMES = frozenset(
    {
        "api_key",
        "authorization",
        "x-api-key",
        "proxy-authorization",
    }
)

REDACTED_VALUE = "[REDACTED]"

ENDPOINT_DIRECTORY_NAMES = {
    "chat.completions.create": "chat_completions",
    "embeddings.create": "embeddings",
}

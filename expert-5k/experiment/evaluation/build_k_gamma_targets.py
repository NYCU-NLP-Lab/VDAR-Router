from __future__ import annotations

from scripts.build_k_gamma_targets import (
    DEFAULT_GAMMA_VALUES,
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TOP_K_VALUES,
    _format_value_token,
    _require_json_object,
    _resolve_base_suffix,
    build_parser,
    generate_targets,
    main,
)

__all__ = [
    "DEFAULT_GAMMA_VALUES",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_TOP_K_VALUES",
    "build_parser",
    "generate_targets",
    "main",
    "_format_value_token",
    "_require_json_object",
    "_resolve_base_suffix",
]


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from scripts.plot_k_gamma_relationship import (
    DEFAULT_EVALUATION_DIR,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TARGETS_JSON_PATH,
    METRIC_SPECS,
    RouterMetrics,
    RouterPoint,
    build_metric_output_path,
    build_parser,
    collect_router_points,
    extract_target_suffix,
    load_router_summary_lookup,
    load_target_lookup,
    main,
    render_combined_plot,
    render_pairwise_reference_plot,
    render_plot,
    render_plots,
    require_json_object,
    require_numeric_metric,
)

__all__ = [
    "DEFAULT_EVALUATION_DIR",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_TARGETS_JSON_PATH",
    "METRIC_SPECS",
    "RouterMetrics",
    "RouterPoint",
    "build_metric_output_path",
    "build_parser",
    "collect_router_points",
    "extract_target_suffix",
    "load_router_summary_lookup",
    "load_target_lookup",
    "main",
    "render_combined_plot",
    "render_pairwise_reference_plot",
    "render_plot",
    "render_plots",
    "require_json_object",
    "require_numeric_metric",
]


if __name__ == "__main__":
    raise SystemExit(main())

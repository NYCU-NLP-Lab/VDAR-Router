from __future__ import annotations

from .contracts import RouterEvaluationSummary


def build_warning_text(router_summaries: list[RouterEvaluationSummary]) -> str:
    warning_blocks = [
        _render_deferred_cache_warning(router_summaries),
        _render_unmapped_model_warning(router_summaries),
        _render_missing_pricing_warning(router_summaries),
        _render_fallback_pricing_warning(router_summaries),
    ]
    return "\n".join(block for block in warning_blocks if block)


def _render_deferred_cache_warning(
    router_summaries: list[RouterEvaluationSummary],
) -> str | None:
    lines: list[str] = []
    for summary in router_summaries:
        if summary.evaluation_status != "deferred_pending":
            continue
        cache = summary.cache or {}
        unique_pending_count = cache.get("unique_pending_count")
        if not isinstance(unique_pending_count, int) or unique_pending_count <= 0:
            continue
        cache_dir = cache.get("cache_dir")
        lines.append(
            (
                f"WARNING: Deferred cache work remains for router "
                f"'{summary.router_key}' ({unique_pending_count} unique pending requests)."
            )
        )
        if isinstance(cache_dir, str) and cache_dir:
            lines.append(f"  Fill the cache under: {cache_dir}")
        deferred_requests = cache.get("deferred_requests")
        if isinstance(deferred_requests, dict):
            pending_paths: list[str] = []
            for endpoint_name in sorted(deferred_requests):
                payload = deferred_requests[endpoint_name]
                if not isinstance(payload, dict):
                    continue
                endpoint_pending_count = payload.get("unique_pending_count")
                requests_path = payload.get("requests_path")
                if (
                    isinstance(endpoint_pending_count, int)
                    and endpoint_pending_count > 0
                    and isinstance(requests_path, str)
                    and requests_path
                ):
                    pending_paths.append(
                        f"  Pending {endpoint_name} requests: {requests_path}"
                    )
            lines.extend(pending_paths)
    if not lines:
        return None
    return "\n".join(lines)


def _render_unmapped_model_warning(
    router_summaries: list[RouterEvaluationSummary],
) -> str | None:
    lines: list[str] = []
    for summary in router_summaries:
        warnings = summary.warnings or {}
        unmapped_raw_model_names = warnings.get("unmapped_raw_model_names")
        llm_config_path = warnings.get("llm_config_path")
        if (
            not isinstance(unmapped_raw_model_names, list)
            or not unmapped_raw_model_names
        ):
            continue
        lines.append(
            (
                f"WARNING: Prompt cost lookup could not map raw model names for router "
                f"'{summary.router_key}'."
            )
        )
        lines.append(
            "  Unmapped raw model names: "
            + ", ".join(str(name) for name in unmapped_raw_model_names)
        )
        if isinstance(llm_config_path, str) and llm_config_path:
            lines.append(f"  Add mappings to: {llm_config_path}")
    if not lines:
        return None
    return "\n".join(lines)


def _render_missing_pricing_warning(
    router_summaries: list[RouterEvaluationSummary],
) -> str | None:
    lines: list[str] = []
    for summary in router_summaries:
        warnings = summary.warnings or {}
        missing_priced_model_names = warnings.get("missing_priced_model_names")
        llm_config_path = warnings.get("llm_config_path")
        if (
            not isinstance(missing_priced_model_names, list)
            or not missing_priced_model_names
        ):
            continue
        lines.append(
            (
                f"WARNING: Prompt cost pricing is missing mapped normalized model names for router "
                f"'{summary.router_key}'."
            )
        )
        lines.append(
            "  Missing priced model names: "
            + ", ".join(str(name) for name in missing_priced_model_names)
        )
        if isinstance(llm_config_path, str) and llm_config_path:
            lines.append(f"  Add pricing to: {llm_config_path}")
    if not lines:
        return None
    return "\n".join(lines)


def _render_fallback_pricing_warning(
    router_summaries: list[RouterEvaluationSummary],
) -> str | None:
    lines: list[str] = []
    for summary in router_summaries:
        warnings = summary.warnings or {}
        fallback_priced_model_names = warnings.get("fallback_priced_model_names")
        fallback_priced_prompt_count = warnings.get("fallback_priced_prompt_count")
        fallback_pricing_model_name = warnings.get("fallback_pricing_model_name")
        llm_config_path = warnings.get("llm_config_path")
        if (
            not isinstance(fallback_priced_model_names, list)
            or not fallback_priced_model_names
        ):
            continue
        prompt_count_text = ""
        if (
            isinstance(fallback_priced_prompt_count, int)
            and fallback_priced_prompt_count > 0
        ):
            prompt_count_text = f" ({fallback_priced_prompt_count} prompt(s))"
        lines.append(
            (
                f"WARNING: Prompt cost used fallback pricing for router "
                f"'{summary.router_key}'{prompt_count_text}."
            )
        )
        lines.append(
            "  Fallback-priced raw model names: "
            + ", ".join(str(name) for name in fallback_priced_model_names)
        )
        if isinstance(fallback_pricing_model_name, str) and fallback_pricing_model_name:
            lines.append(f"  Fallback pricing source: {fallback_pricing_model_name}")
        if isinstance(llm_config_path, str) and llm_config_path:
            lines.append(f"  Review pricing config: {llm_config_path}")
    if not lines:
        return None
    return "\n".join(lines)


__all__ = [
    "build_warning_text",
    "_render_deferred_cache_warning",
    "_render_fallback_pricing_warning",
    "_render_missing_pricing_warning",
    "_render_unmapped_model_warning",
]

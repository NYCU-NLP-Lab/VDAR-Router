from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .contracts import (
    EvaluationManifest,
    EvaluationRunResult,
    RouterEvaluationSummary,
    _is_secret_like_key,
)

REPORT_DIRNAME = "report"
REPORT_INDEX_FILENAME = "index.md"
REPORT_SUBDIR_BY_CASE_KIND = {
    "error": "error",
    "pairwise_incorrect": "pairwise_incorrect",
    "pairwise_correct": "pairwise_correct",
}


def build_artifact_root(data_dir: Path, dataset_id: str, run_label: str) -> Path:
    return data_dir / "evaluations" / dataset_id / run_label


def write_evaluation_artifacts(
    *,
    artifact_root: Path,
    manifest: EvaluationManifest,
    router_payloads: list[dict[str, Any]],
    report_enabled: bool = True,
) -> EvaluationRunResult:
    router_summaries: list[RouterEvaluationSummary] = []
    for payload in router_payloads:
        router_summary = write_router_evaluation_artifacts(
            artifact_root=artifact_root,
            payload=payload,
            report_enabled=report_enabled,
        )
        router_summaries.append(router_summary)
    return finalize_evaluation_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        router_summaries=router_summaries,
    )


def write_router_evaluation_artifacts(
    *,
    artifact_root: Path,
    payload: dict[str, Any],
    report_enabled: bool = True,
) -> RouterEvaluationSummary:
    routers_root = _ensure_artifact_root(artifact_root)
    return _write_router_artifacts(
        routers_root=routers_root,
        payload=payload,
        report_enabled=report_enabled,
    )


def write_evaluation_summary(
    *,
    artifact_root: Path,
    router_summaries: list[RouterEvaluationSummary],
) -> Path:
    _ensure_artifact_root(artifact_root)
    summary_path = artifact_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "routers": [
                    router_summary.to_dict() for router_summary in router_summaries
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_path


def finalize_evaluation_artifacts(
    *,
    artifact_root: Path,
    manifest: EvaluationManifest,
    router_summaries: list[RouterEvaluationSummary],
) -> EvaluationRunResult:
    _ensure_artifact_root(artifact_root)
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path = write_evaluation_summary(
        artifact_root=artifact_root,
        router_summaries=router_summaries,
    )

    result = EvaluationRunResult(
        artifact_root=artifact_root,
        manifest_path=manifest_path,
        summary_path=summary_path,
        manifest=manifest,
        router_summaries=list(router_summaries),
    )
    validate_evaluation_artifact_contract(result)
    return result


def _ensure_artifact_root(artifact_root: Path) -> Path:
    artifact_root.mkdir(parents=True, exist_ok=True)
    routers_root = artifact_root / "routers"
    routers_root.mkdir(parents=True, exist_ok=True)
    return routers_root


def validate_evaluation_artifact_contract(result: EvaluationRunResult) -> None:
    required_paths = [result.artifact_root, result.manifest_path, result.summary_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise ValueError(
            f"Evaluation artifact root is missing required files: {missing}"
        )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    required_sections = {
        "schema_version",
        "inputs",
        "outputs",
        "dataset",
        "baselines",
        "routers",
    }
    missing_sections = sorted(required_sections - set(manifest))
    if missing_sections:
        raise ValueError(
            f"evaluation manifest is missing required sections: {missing_sections}"
        )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary.get("routers"), list):
        raise ValueError("evaluation summary must contain a routers list.")

    for router_summary in result.router_summaries:
        router_root = router_summary.artifact_root
        required_router_paths = [
            router_root / "summary.json",
            router_root / "prompts.jsonl",
            router_root / "pairs.jsonl",
            router_root / "failures.jsonl",
        ]
        report_root = router_root / REPORT_DIRNAME
        report_index_path = report_root / REPORT_INDEX_FILENAME
        if report_root.exists() or report_index_path.exists():
            required_router_paths.extend([report_root, report_index_path])
        missing_router_paths = [
            str(path) for path in required_router_paths if not path.exists()
        ]
        if missing_router_paths:
            raise ValueError(
                "Router evaluation artifact root is missing required files: "
                f"{missing_router_paths}"
            )
        neighbors_path = router_root / "neighbors.jsonl"
        if neighbors_path.exists():
            _validate_optional_neighbors_artifact(
                neighbors_path,
                expected_router_key=router_summary.router_key,
            )


def _write_router_artifacts(
    *,
    routers_root: Path,
    payload: dict[str, Any],
    report_enabled: bool,
) -> RouterEvaluationSummary:
    router_summary: RouterEvaluationSummary = payload["summary"]
    router_root = routers_root / router_summary.router_key
    router_root.mkdir(parents=True, exist_ok=True)

    final_summary = RouterEvaluationSummary(
        router_key=router_summary.router_key,
        router_family=router_summary.router_family,
        router_variant=router_summary.router_variant,
        manifest_path=router_summary.manifest_path,
        target_suffix=router_summary.target_suffix,
        artifact_root=router_root,
        counts=router_summary.counts,
        metrics=router_summary.metrics,
        evaluation_status=router_summary.evaluation_status,
        runtime_config_effective=dict(router_summary.runtime_config_effective),
        cache=(
            dict(router_summary.cache) if router_summary.cache is not None else None
        ),
        warnings=(
            dict(router_summary.warnings)
            if router_summary.warnings is not None
            else None
        ),
    )

    _write_jsonl(router_root / "prompts.jsonl", payload["prompt_records"])
    _write_jsonl(router_root / "pairs.jsonl", payload["pair_records"])
    _write_jsonl(router_root / "failures.jsonl", payload["failure_records"])
    _write_optional_neighbors_artifact(
        router_root / "neighbors.jsonl", payload.get("neighbor_records")
    )
    if report_enabled:
        _write_report_artifacts(
            router_root=router_root,
            router_summary=final_summary,
            prompt_records=payload["prompt_records"],
            pair_records=payload["pair_records"],
            failure_records=payload["failure_records"],
        )
    else:
        report_root = router_root / REPORT_DIRNAME
        if report_root.exists():
            shutil.rmtree(report_root)
    (router_root / "summary.json").write_text(
        json.dumps(final_summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return final_summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _write_optional_neighbors_artifact(path: Path, rows: Any) -> None:
    if isinstance(rows, list) and rows:
        _write_jsonl(path, rows)
        return
    if path.exists():
        path.unlink()


def _validate_optional_neighbors_artifact(
    path: Path, *, expected_router_key: str
) -> None:
    seen_row_keys: set[tuple[str, str, str, int]] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"neighbors artifact contains invalid JSON: {path} line {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"neighbors artifact rows must be JSON objects: {path} line {line_number}"
            )

        for field_name in (
            "prompt_id",
            "test_item_id",
            "retrieved_item_id",
            "router_key",
            "source_split",
        ):
            value = row.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"neighbors artifact field '{field_name}' must be a non-empty string: {path} line {line_number}"
                )

        if row.get("router_key") != expected_router_key:
            raise ValueError(
                f"neighbors artifact router_key must match router directory '{expected_router_key}': {path} line {line_number}"
            )

        rank = row.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError(
                f"neighbors artifact field 'rank' must be a positive integer: {path} line {line_number}"
            )

        if "retrieval_score" not in row:
            raise ValueError(
                f"neighbors artifact field 'retrieval_score' must be present: {path} line {line_number}"
            )
        retrieval_score = row.get("retrieval_score")
        if retrieval_score is not None and (
            isinstance(retrieval_score, bool)
            or not isinstance(retrieval_score, int | float)
        ):
            raise ValueError(
                f"neighbors artifact field 'retrieval_score' must be numeric or null: {path} line {line_number}"
            )

        retrieval_metadata = row.get("retrieval_metadata")
        if retrieval_metadata is not None:
            if not isinstance(retrieval_metadata, dict):
                raise ValueError(
                    f"neighbors artifact field 'retrieval_metadata' must be an object when present: {path} line {line_number}"
                )
            secret_like_keys = sorted(
                str(key) for key in retrieval_metadata if _is_secret_like_key(str(key))
            )
            if secret_like_keys:
                raise ValueError(
                    "neighbors artifact retrieval_metadata must not contain secret-like keys: "
                    f"{secret_like_keys} at {path} line {line_number}"
                )

        row_key = (
            str(row["prompt_id"]),
            str(row["test_item_id"]),
            str(row["retrieved_item_id"]),
            rank,
        )
        if row_key in seen_row_keys:
            raise ValueError(
                "neighbors artifact contains duplicate "
                f"(prompt_id, test_item_id, retrieved_item_id, rank) rows: {path} line {line_number}"
            )
        seen_row_keys.add(row_key)


def _write_report_artifacts(
    *,
    router_root: Path,
    router_summary: RouterEvaluationSummary,
    prompt_records: list[dict[str, Any]],
    pair_records: list[dict[str, Any]],
    failure_records: list[dict[str, Any]],
) -> None:
    report_root = router_root / REPORT_DIRNAME
    report_root.mkdir(parents=True, exist_ok=True)
    for existing_path in report_root.rglob("*.md"):
        existing_path.unlink()

    cases = _build_report_cases(
        router_key=router_summary.router_key,
        prompt_records=prompt_records,
        pair_records=pair_records,
        failure_records=failure_records,
    )
    for case in tqdm(
        cases,
        total=len(cases),
        desc=f"Building report for {router_summary.router_key}",
        unit="case",
        disable=None,
    ):
        case_root = report_root / str(case["case_subdir"])
        case_root.mkdir(parents=True, exist_ok=True)
        (case_root / str(case["file_name"])).write_text(
            _render_report_case_markdown(case),
            encoding="utf-8",
        )

    (report_root / REPORT_INDEX_FILENAME).write_text(
        _render_report_index(router_summary=router_summary, cases=cases),
        encoding="utf-8",
    )


def _build_report_cases(
    *,
    router_key: str,
    prompt_records: list[dict[str, Any]],
    pair_records: list[dict[str, Any]],
    failure_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    prompt_records_by_id = {
        str(prompt_record.get("prompt_id")): prompt_record
        for prompt_record in prompt_records
        if isinstance(prompt_record.get("prompt_id"), str)
    }

    for pair_record in sorted(
        pair_records,
        key=lambda item: (
            str(item.get("prompt_id", "")),
            str(item.get("left_model_name", "")),
            str(item.get("right_model_name", "")),
        ),
    ):
        prompt_id = pair_record.get("prompt_id")
        prompt_record = (
            prompt_records_by_id.get(prompt_id) if isinstance(prompt_id, str) else None
        )
        is_correct = pair_record.get("correct") is True
        case_kind = "pairwise_correct" if is_correct else "pairwise_incorrect"
        cases.append(
            {
                "router_key": router_key,
                "case_kind": case_kind,
                "case_bucket": "correct" if is_correct else "incorrect",
                "correctness_status": "correct" if is_correct else "incorrect",
                "prompt_id": prompt_id,
                "query": (
                    prompt_record.get("query")
                    if isinstance(prompt_record, dict)
                    else None
                ),
                "selected_model_name": (
                    prompt_record.get("selected_model_name")
                    if isinstance(prompt_record, dict)
                    else None
                ),
                "truth_winner_model_name": pair_record.get("truth_winner_model_name"),
                "prompt_record": prompt_record,
                "pair_record": pair_record,
                "failure_record": None,
            }
        )

    for failure_record in sorted(
        failure_records,
        key=lambda item: (
            str(item.get("failure_stage", "")),
            str(item.get("prompt_id", "")),
            str(item.get("error_type", "")),
        ),
    ):
        cases.append(
            {
                "router_key": router_key,
                "case_kind": "error",
                "case_bucket": "error",
                "correctness_status": "error",
                "prompt_id": failure_record.get("prompt_id"),
                "query": failure_record.get("query"),
                "selected_model_name": None,
                "truth_winner_model_name": None,
                "prompt_record": None,
                "pair_record": None,
                "failure_record": failure_record,
            }
        )

    cases.sort(key=_case_sort_key)
    for index, case in enumerate(cases, start=1):
        case["file_name"] = _build_report_case_file_name(index=index, case=case)
        case["case_subdir"] = REPORT_SUBDIR_BY_CASE_KIND.get(
            str(case.get("case_kind", "")), str(case.get("case_kind", "case"))
        )
    return cases


def _case_sort_key(case: dict[str, Any]) -> tuple[int, str, str, str]:
    case_kind_order = {
        "pairwise_incorrect": 0,
        "error": 1,
        "pairwise_correct": 2,
    }
    pair_record = case.get("pair_record")
    return (
        case_kind_order.get(str(case.get("case_kind", "")), 99),
        str(case.get("prompt_id", "")),
        str(pair_record.get("left_model_name", ""))
        if isinstance(pair_record, dict)
        else "",
        str(pair_record.get("right_model_name", ""))
        if isinstance(pair_record, dict)
        else "",
    )


def _build_report_case_file_name(*, index: int, case: dict[str, Any]) -> str:
    base_parts = [str(case.get("case_kind", "case"))]
    prompt_id = case.get("prompt_id")
    failure_record = case.get("failure_record")
    if isinstance(prompt_id, str) and prompt_id.strip():
        base_parts.append(prompt_id.strip())
    elif isinstance(failure_record, dict):
        failure_stage = failure_record.get("failure_stage")
        if isinstance(failure_stage, str) and failure_stage.strip():
            base_parts.append(failure_stage.strip())
    slug = "--".join(_slugify(part) for part in base_parts if part)
    return f"case-{index:03d}--{slug or 'case'}.md"


def _render_report_index(
    *, router_summary: RouterEvaluationSummary, cases: list[dict[str, Any]]
) -> str:
    pairwise_correct_count = sum(
        1 for case in cases if case["case_kind"] == "pairwise_correct"
    )
    pairwise_incorrect_count = sum(
        1 for case in cases if case["case_kind"] == "pairwise_incorrect"
    )
    error_count = sum(1 for case in cases if case["case_kind"] == "error")
    lines = [
        f"# Report for `{router_summary.router_key}`",
        "",
        f"- Total cases: {len(cases)}",
        f"- Pairwise correct: {pairwise_correct_count}",
        f"- Pairwise incorrect: {pairwise_incorrect_count}",
        f"- Errors: {error_count}",
        "",
    ]
    if not cases:
        lines.extend(
            [
                "No report cases were recorded for this router evaluation.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "| Case | Type | Status | Prompt ID | Selected | Ground Truth Winner | Failure Stage | File |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for case in cases:
        failure_record = case.get("failure_record")
        failure_stage = ""
        if isinstance(failure_record, dict):
            raw_failure_stage = failure_record.get("failure_stage")
            if isinstance(raw_failure_stage, str):
                failure_stage = raw_failure_stage
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case['file_name'].removesuffix('.md')}`",
                    str(case.get("case_kind", "")),
                    str(case.get("correctness_status", "")),
                    _table_cell(case.get("prompt_id")),
                    _table_cell(case.get("selected_model_name")),
                    _table_cell(case.get("truth_winner_model_name")),
                    _table_cell(failure_stage),
                    f"[{case['file_name']}]({case['case_subdir']}/{case['file_name']})",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_report_case_markdown(case: dict[str, Any]) -> str:
    prompt_record = case.get("prompt_record")
    pair_record = case.get("pair_record")
    failure_record = case.get("failure_record")
    query = case.get("query")
    lines = [
        f"# Report Case `{case['file_name'].removesuffix('.md')}`",
        "",
        f"- Router: `{case['router_key']}`",
        f"- Case type: `{case['case_kind']}`",
        f"- Status: `{case.get('correctness_status', '')}`",
    ]
    if isinstance(prompt_record, dict):
        lines.extend(
            [
                f"- Prompt ID: `{prompt_record.get('prompt_id')}`",
                f"- Selected model: `{prompt_record.get('selected_model_name')}`",
                f"- Ground-truth winner: `{case.get('truth_winner_model_name')}`",
            ]
        )
        if isinstance(pair_record, dict):
            lines.extend(
                [
                    f"- Left model: `{pair_record.get('left_model_name')}`",
                    f"- Right model: `{pair_record.get('right_model_name')}`",
                    f"- Predicted winner: `{pair_record.get('predicted_winner_model_name')}`",
                ]
            )
    elif isinstance(failure_record, dict):
        lines.append(
            f"- Failure stage: `{failure_record.get('failure_stage', 'unknown')}`"
        )
        if case.get("prompt_id"):
            lines.append(f"- Prompt ID: `{case['prompt_id']}`")
    lines.append("")

    if isinstance(query, str) and query.strip():
        lines.extend(["## Query", "", "```text", query.strip(), "```", ""])

    if isinstance(prompt_record, dict):
        lines.extend(
            _render_prompt_case_sections(prompt_record=prompt_record, case=case)
        )

    if isinstance(failure_record, dict):
        lines.extend(_render_failure_case_sections(failure_record=failure_record))

    diagnostic_labels = _build_diagnostic_labels(case)
    if diagnostic_labels:
        lines.extend(["## Diagnostic Labels", ""])
        for label in diagnostic_labels:
            lines.append(f"- `{label}`")
        lines.append("")

    return "\n".join(lines)


def _render_prompt_case_sections(
    *, prompt_record: dict[str, Any], case: dict[str, Any]
) -> list[str]:
    sections: list[str] = []
    ground_truth_scores = prompt_record.get("ground_truth_scores")
    ranked_models = prompt_record.get("ranked_models")
    router_metadata = prompt_record.get("router_metadata")
    emphasized_model_names = _resolve_emphasized_model_names(case)
    pair_record = case.get("pair_record")
    if isinstance(pair_record, dict):
        sections.extend(
            _render_pairwise_ground_truth_sections(
                prompt_record=prompt_record,
                pair_record=pair_record,
            )
        )

    if isinstance(ground_truth_scores, dict):
        sections.extend(["## Ground Truth", ""])
        sections.extend(
            [
                "| Model | Score | Label |",
                "| --- | --- | --- |",
            ]
        )
        for model_name, score in _sorted_score_items(ground_truth_scores):
            sections.append(
                f"| `{model_name}` | {_format_number(score)} | {_label_ground_truth_score(score, ground_truth_scores)} |"
            )
        sections.append("")

    if isinstance(ranked_models, list):
        sections.extend(["## Router Ranking", ""])
        sections.extend(
            [
                "| Rank | Model | Score |",
                "| --- | --- | --- |",
            ]
        )
        for rank, ranked_model in enumerate(ranked_models, start=1):
            if not isinstance(ranked_model, dict):
                continue
            rendered_model_name = f"`{ranked_model.get('model_name', '')}`"
            raw_model_name = ranked_model.get("model_name")
            if (
                isinstance(raw_model_name, str)
                and raw_model_name in emphasized_model_names
            ):
                rendered_model_name = f"**{rendered_model_name}**"
            sections.append(
                "| "
                + " | ".join(
                    [
                        str(rank),
                        rendered_model_name,
                        _format_number(ranked_model.get("score")),
                    ]
                )
                + " |"
            )
        sections.append("")

    if isinstance(pair_record, dict):
        sections.extend(["## Pairwise Comparison", ""])
        sections.append(f"- Left model: `{pair_record.get('left_model_name')}`")
        sections.append(f"- Right model: `{pair_record.get('right_model_name')}`")
        sections.append(
            f"- Ground-truth winner: `{pair_record.get('truth_winner_model_name')}`"
        )
        sections.append(
            f"- Predicted winner: `{pair_record.get('predicted_winner_model_name')}`"
        )
        sections.append("")

    ranking_gaps = _build_ranking_gap_items(prompt_record=prompt_record, case=case)
    if ranking_gaps:
        sections.extend(["## Ranking Gaps", ""])
        for label, value in ranking_gaps:
            sections.append(f"- {label}: {_format_number(value)}")
        sections.append("")

    if isinstance(router_metadata, dict):
        difficulty_lines = _render_difficulty_analysis(router_metadata)
        if difficulty_lines:
            sections.extend(difficulty_lines)
        evidence_lines = _render_retrieved_evidence(
            router_metadata,
            model_names=emphasized_model_names or None,
        )
        if evidence_lines:
            sections.extend(evidence_lines)

    return sections


def _render_pairwise_ground_truth_sections(
    *, prompt_record: dict[str, Any], pair_record: dict[str, Any]
) -> list[str]:
    model_outputs = prompt_record.get("ground_truth_model_outputs")
    ground_truth_scores = prompt_record.get("ground_truth_scores")
    ranked_positions = _resolve_ranked_model_positions(
        prompt_record.get("ranked_models")
    )
    model_names = [
        model_name
        for model_name in (
            pair_record.get("left_model_name"),
            pair_record.get("right_model_name"),
        )
        if isinstance(model_name, str) and model_name.strip()
    ]
    if not model_names:
        return []

    sections = ["## Ground-Truth Model Focus", ""]
    for model_name in model_names:
        outputs = []
        if isinstance(model_outputs, dict):
            raw_outputs = model_outputs.get(model_name)
            if isinstance(raw_outputs, list):
                outputs = [
                    output.strip()
                    for output in raw_outputs
                    if isinstance(output, str) and output.strip()
                ]
        rank_item = ranked_positions.get(model_name)
        ground_truth_score = (
            _coerce_number(ground_truth_scores.get(model_name))
            if isinstance(ground_truth_scores, dict)
            else None
        )

        sections.extend([f"### `{model_name}`", ""])
        if rank_item is None:
            sections.append("- Router rank: not ranked")
        else:
            sections.append(f"- Router rank: `{rank_item[0]}`")
            sections.append(f"- Router score: `{_format_number(rank_item[1])}`")
        if ground_truth_score is not None:
            sections.append(
                f"- Ground-truth score: `{_format_number(ground_truth_score)}`"
            )
        sections.append("")

        if outputs:
            for output_index, output in enumerate(outputs, start=1):
                heading = (
                    "#### Answer"
                    if len(outputs) == 1
                    else f"#### Answer {output_index}"
                )
                sections.extend([heading, "", "```text", output, "```", ""])
        else:
            sections.extend(["#### Answer", "", "_Unavailable_", ""])

    return sections


def _render_failure_case_sections(failure_record: dict[str, Any]) -> list[str]:
    sections = ["## Failure Details", ""]
    failure_stage = failure_record.get("failure_stage")
    error_type = failure_record.get("error_type")
    error_message = failure_record.get("error_message")
    if failure_stage is not None:
        sections.append(f"- Failure stage: `{failure_stage}`")
    if error_type is not None:
        sections.append(f"- Error type: `{error_type}`")
    if error_message is not None:
        sections.append(f"- Error message: `{error_message}`")
    affected_prompt_count = failure_record.get("affected_prompt_count")
    if affected_prompt_count is not None:
        sections.append(f"- Affected prompts: `{affected_prompt_count}`")
    sections.append("")
    return sections


def _render_difficulty_analysis(router_metadata: dict[str, Any]) -> list[str]:
    analysis_text = router_metadata.get("analysis_text")
    query_summary = router_metadata.get("query_summary")
    strategy = router_metadata.get("strategy")
    analysis_model = router_metadata.get("analysis_model")
    embedding_model = router_metadata.get("embedding_model")
    input_token_count = router_metadata.get("input_token_count")
    if not any(
        value is not None
        for value in (
            analysis_text,
            query_summary,
            strategy,
            analysis_model,
            embedding_model,
            input_token_count,
        )
    ):
        return []

    lines = ["## Difficulty Analysis", ""]
    if strategy is not None:
        lines.append(f"- Strategy: `{strategy}`")
    if analysis_model is not None:
        lines.append(f"- Analysis model: `{analysis_model}`")
    if embedding_model is not None:
        lines.append(f"- Embedding model: `{embedding_model}`")
    if input_token_count is not None:
        lines.append(f"- Estimated input tokens: `{input_token_count}`")
    if isinstance(analysis_text, str) and analysis_text.strip():
        lines.extend(
            ["", "### Analysis Text", "", "```text", analysis_text.strip(), "```"]
        )
    if isinstance(query_summary, str) and query_summary.strip():
        lines.extend(
            ["", "### Query Summary", "", "```text", query_summary.strip(), "```"]
        )
    lines.append("")
    return lines


def _render_retrieved_evidence(
    router_metadata: dict[str, Any],
    *,
    model_names: set[str] | None = None,
) -> list[str]:
    retrieved_evidence = router_metadata.get("retrieved_evidence")
    if not isinstance(retrieved_evidence, dict) or not retrieved_evidence:
        return []

    lines = ["## Retrieved Evidence", ""]
    for model_name in sorted(retrieved_evidence):
        if model_names is not None and model_name not in model_names:
            continue
        evidence_rows = retrieved_evidence.get(model_name)
        if not isinstance(evidence_rows, list) or not evidence_rows:
            continue
        lines.extend(
            [
                f"### `{model_name}`",
                "",
                "| Rank | Correctness | Distance | Similarity | Prompt ID | Row ID | Score |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for evidence_row in evidence_rows:
            if not isinstance(evidence_row, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        _table_cell(evidence_row.get("rank")),
                        _table_cell(evidence_row.get("correctness")),
                        _format_number(evidence_row.get("distance")),
                        _format_number(evidence_row.get("similarity")),
                        _table_cell(evidence_row.get("prompt_id")),
                        _table_cell(evidence_row.get("row_id")),
                        _format_number(evidence_row.get("score")),
                    ]
                )
                + " |"
            )
            question = evidence_row.get("question")
            if isinstance(question, str) and question.strip():
                lines.extend(
                    [
                        "",
                        f"Prompt text for rank {_table_cell(evidence_row.get('rank'))}:",
                        "```text",
                        question.strip(),
                        "```",
                    ]
                )
            retrieved_document = evidence_row.get("retrieved_document")
            if isinstance(retrieved_document, str) and retrieved_document.strip():
                lines.extend(
                    [
                        "",
                        f"Retrieved summary for rank {_table_cell(evidence_row.get('rank'))}:",
                        "```text",
                        retrieved_document.strip(),
                        "```",
                    ]
                )
            occupational_tags = _extract_occupational_tags(evidence_row)
            if occupational_tags:
                lines.extend(
                    [
                        "",
                        f"Occupational tags for rank {_table_cell(evidence_row.get('rank'))}: {', '.join(f'`{tag}`' for tag in occupational_tags)}",
                    ]
                )
        lines.append("")
    return lines if len(lines) > 2 else []


def _extract_occupational_tags(evidence_row: dict[str, Any]) -> list[str]:
    row_metadata = evidence_row.get("row_metadata")
    if not isinstance(row_metadata, dict):
        return []
    raw_tags = row_metadata.get("occupational_tags")
    if not isinstance(raw_tags, list):
        return []

    seen: set[str] = set()
    tags: list[str] = []
    for raw_tag in raw_tags:
        if not isinstance(raw_tag, str):
            continue
        tag = raw_tag.strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def _resolve_ranked_model_positions(
    ranked_models: object,
) -> dict[str, tuple[int, float | None]]:
    if not isinstance(ranked_models, list):
        return {}

    positions: dict[str, tuple[int, float | None]] = {}
    for rank, ranked_model in enumerate(ranked_models, start=1):
        if not isinstance(ranked_model, dict):
            continue
        model_name = ranked_model.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            continue
        positions[model_name] = (rank, _coerce_number(ranked_model.get("score")))
    return positions


def _build_ranking_gap_items(
    *, prompt_record: dict[str, Any], case: dict[str, Any]
) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    ranked_models = prompt_record.get("ranked_models")
    ground_truth_scores = prompt_record.get("ground_truth_scores")
    selected_model_name = prompt_record.get("selected_model_name")
    truth_winner_model_name = case.get("truth_winner_model_name")
    ranked_scores = _coerce_ranked_scores(ranked_models)
    if len(ranked_scores) >= 2:
        top_two = list(ranked_scores.values())[:2]
        items.append(("Router top-1 minus top-2 score", top_two[0] - top_two[1]))
    if (
        isinstance(ground_truth_scores, dict)
        and isinstance(selected_model_name, str)
        and isinstance(truth_winner_model_name, str)
    ):
        selected_truth_score = _coerce_number(
            ground_truth_scores.get(selected_model_name)
        )
        truth_winner_score = _coerce_number(
            ground_truth_scores.get(truth_winner_model_name)
        )
        if selected_truth_score is not None and truth_winner_score is not None:
            items.append(
                (
                    "Ground-truth winner minus selected score",
                    truth_winner_score - selected_truth_score,
                )
            )
    if (
        isinstance(selected_model_name, str)
        and isinstance(truth_winner_model_name, str)
        and selected_model_name in ranked_scores
        and truth_winner_model_name in ranked_scores
    ):
        items.append(
            (
                "Router selected minus ground-truth winner score",
                ranked_scores[selected_model_name]
                - ranked_scores[truth_winner_model_name],
            )
        )
    return items


def _build_diagnostic_labels(case: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    case_kind = case.get("case_kind")
    if case_kind in {"pairwise_incorrect", "pairwise_correct"}:
        labels.extend(
            [
                "pairwise",
                "pairwise-correct"
                if case_kind == "pairwise_correct"
                else "pairwise-incorrect",
            ]
        )
        if case_kind == "pairwise_correct":
            labels.append("pairwise-agrees-with-ground-truth")
        else:
            labels.append("pairwise-disagrees-with-ground-truth")
        pair_record = case.get("pair_record")
        if isinstance(pair_record, dict):
            truth_winner = pair_record.get("truth_winner_model_name")
            predicted_winner = pair_record.get("predicted_winner_model_name")
            if truth_winner is not None:
                labels.append(f"truth-winner-{_slugify(str(truth_winner))}")
            if predicted_winner is not None:
                labels.append(f"predicted-winner-{_slugify(str(predicted_winner))}")
        prompt_record = case.get("prompt_record")
        if isinstance(prompt_record, dict):
            router_metadata = prompt_record.get("router_metadata")
            if isinstance(router_metadata, dict):
                if router_metadata.get("query_summary"):
                    labels.append("difficulty-analysis-available")
                retrieved_evidence = router_metadata.get("retrieved_evidence")
                if isinstance(retrieved_evidence, dict) and retrieved_evidence:
                    labels.append("retrieved-evidence-available")
    else:
        labels.append("error")
        failure_record = case.get("failure_record")
        if isinstance(failure_record, dict):
            failure_stage = failure_record.get("failure_stage")
            if isinstance(failure_stage, str) and failure_stage.strip():
                labels.append(_slugify(failure_stage))
                if failure_stage in {"router_setup", "manifest_load"}:
                    labels.append("router-setup-failure")
                if failure_stage == "prompt_route":
                    labels.append("prompt-route-failure")
    return labels


def _resolve_emphasized_model_names(case: dict[str, Any]) -> set[str]:
    pair_record = case.get("pair_record")
    if not isinstance(pair_record, dict):
        return set()
    emphasized_model_names: set[str] = set()
    for key in ("left_model_name", "right_model_name"):
        model_name = pair_record.get(key)
        if isinstance(model_name, str) and model_name.strip():
            emphasized_model_names.add(model_name)
    return emphasized_model_names


def _sorted_score_items(scores: dict[str, Any]) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for model_name, raw_score in scores.items():
        score = _coerce_number(raw_score)
        if score is None:
            continue
        items.append((str(model_name), score))
    return sorted(items, key=lambda item: (-item[1], item[0]))


def _label_ground_truth_score(score: float, ground_truth_scores: dict[str, Any]) -> str:
    sorted_scores = _sorted_score_items(ground_truth_scores)
    if not sorted_scores:
        return "unknown"
    winning_score = sorted_scores[0][1]
    if math.isclose(score, winning_score, rel_tol=0.0, abs_tol=1e-12):
        winners = [
            model_name
            for model_name, winner_score in sorted_scores
            if math.isclose(winner_score, winning_score, rel_tol=0.0, abs_tol=1e-12)
        ]
        return "winner" if len(winners) == 1 else "tied-winner"
    return "non-winner"


def _coerce_ranked_scores(ranked_models: object) -> dict[str, float]:
    if not isinstance(ranked_models, list):
        return {}
    scores: dict[str, float] = {}
    for ranked_model in ranked_models:
        if not isinstance(ranked_model, dict):
            continue
        model_name = ranked_model.get("model_name")
        score = _coerce_number(ranked_model.get("score"))
        if isinstance(model_name, str) and score is not None:
            scores[model_name] = score
    return scores


def _coerce_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _format_number(value: object) -> str:
    numeric_value = _coerce_number(value)
    if numeric_value is None:
        return ""
    return f"{numeric_value:.6f}"


def _table_cell(value: object) -> str:
    if value is None:
        return ""
    rendered = str(value).replace("|", "\\|").strip()
    return rendered


def _slugify(value: str) -> str:
    slug = "".join(
        character.lower() if character.isalnum() else "-" for character in value.strip()
    )
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")

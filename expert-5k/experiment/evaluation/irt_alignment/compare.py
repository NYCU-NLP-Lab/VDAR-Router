from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

DEFAULT_NEAR_RATE_FACTORS = (0.1, 0.25, 0.5)


def build_alignment_rows(
    *,
    router_key: str,
    neighbor_rows: list[dict[str, Any]],
    canonical_row_by_id: dict[str, Any],
    train_item_ids: set[str],
    test_item_ids: set[str],
    difficulty_lookup: dict[str, dict[str, Any]],
    difficulty_source: str,
    validate_neighbor_row: Any,
    build_missing_reason: Any,
) -> list[dict[str, Any]]:
    alignment_rows: list[dict[str, Any]] = []
    for line_number, row in enumerate(neighbor_rows, start=1):
        validate_neighbor_row(router_key=router_key, row=row, line_number=line_number)
        test_item_id = str(row["test_item_id"])
        retrieved_item_id = str(row["retrieved_item_id"])
        if test_item_id not in canonical_row_by_id:
            raise ValueError(
                f"neighbors.jsonl references unknown test_item_id '{test_item_id}' for router '{router_key}'."
            )
        if retrieved_item_id not in canonical_row_by_id:
            raise ValueError(
                f"neighbors.jsonl references unknown retrieved_item_id '{retrieved_item_id}' for router '{router_key}'."
            )
        if test_item_id not in test_item_ids:
            raise ValueError(
                f"neighbors.jsonl test_item_id '{test_item_id}' must resolve to the Stage 1 test split for router '{router_key}'."
            )
        if retrieved_item_id not in train_item_ids:
            raise ValueError(
                f"neighbors.jsonl retrieved_item_id '{retrieved_item_id}' must resolve to the Stage 1 train split for router '{router_key}'."
            )
        if row.get("source_split") != "train":
            raise ValueError(
                f"neighbors.jsonl source_split must equal 'train' for router '{router_key}'."
            )
        test_difficulty = difficulty_lookup.get(test_item_id)
        retrieved_difficulty = difficulty_lookup.get(retrieved_item_id)
        b_test = test_difficulty["difficulty"] if test_difficulty is not None else None
        b_retrieved = (
            retrieved_difficulty["difficulty"]
            if retrieved_difficulty is not None
            else None
        )
        delta_b = (
            float(b_retrieved) - float(b_test)
            if b_test is not None and b_retrieved is not None
            else None
        )
        abs_delta_b = abs(delta_b) if delta_b is not None else None
        alignment_row: dict[str, Any] = {
            "router_key": router_key,
            "prompt_id": row["prompt_id"],
            "test_item_id": test_item_id,
            "retrieved_item_id": retrieved_item_id,
            "rank": row["rank"],
            "b_test": b_test,
            "b_retrieved": b_retrieved,
            "delta_b": delta_b,
            "abs_delta_b": abs_delta_b,
            "difficulty_source": difficulty_source,
            "comparison_arm": "reward_topk",
        }
        for optional_key in (
            "retrieval_score",
            "selected_model_name",
            "candidate_model_name",
            "distance",
            "similarity",
            "query_text",
            "target_suffix",
        ):
            if optional_key in row:
                alignment_row[optional_key] = row[optional_key]
        missing_reason = build_missing_reason(b_test=b_test, b_retrieved=b_retrieved)
        if missing_reason is not None:
            alignment_row["missing_reason"] = missing_reason
        alignment_rows.append(alignment_row)
    return alignment_rows


def build_reward_neighbor_rows(
    alignment_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in alignment_rows:
        reward_row = dict(row)
        reward_row["comparison_arm"] = "reward_topk"
        rows.append(reward_row)
    return rows


def build_query_embedding_knn_rows(
    *,
    router_key: str,
    test_item_id: str,
    reward_rows: list[dict[str, Any]],
    train_item_ids: list[str],
    canonical_row_by_id: dict[str, Any],
    difficulty_lookup: dict[str, dict[str, Any]],
    embedding_lookup: dict[str, list[float]],
    top_k: int,
) -> list[dict[str, Any]]:
    import numpy as np

    if test_item_id not in embedding_lookup or top_k <= 0:
        return []
    test_vector = np.asarray(embedding_lookup[test_item_id], dtype=float)
    test_norm = float(np.linalg.norm(test_vector))
    if test_norm <= 0 or not math.isfinite(test_norm):
        return []
    reward_row = reward_rows[0]
    prompt_id = str(reward_row["prompt_id"])
    selected_reward_model = reward_row.get("selected_model_name")
    b_test = reward_row.get("b_test")
    candidates: list[tuple[str, float]] = []
    for candidate_item_id in train_item_ids:
        if candidate_item_id not in embedding_lookup:
            continue
        retrieved_difficulty = difficulty_lookup.get(candidate_item_id)
        if retrieved_difficulty is None or retrieved_difficulty["difficulty"] is None:
            continue
        candidate_vector = np.asarray(embedding_lookup[candidate_item_id], dtype=float)
        candidate_norm = float(np.linalg.norm(candidate_vector))
        if candidate_norm <= 0 or not math.isfinite(candidate_norm):
            continue
        similarity = float(
            np.dot(test_vector, candidate_vector) / (test_norm * candidate_norm)
        )
        if not math.isfinite(similarity):
            continue
        candidates.append((candidate_item_id, similarity))
    candidates.sort(key=lambda item: item[1], reverse=True)
    rows: list[dict[str, Any]] = []
    for rank, (retrieved_item_id, similarity) in enumerate(candidates[:top_k], start=1):
        retrieved_difficulty = difficulty_lookup[retrieved_item_id]
        b_retrieved = retrieved_difficulty["difficulty"]
        delta_b = (
            float(b_retrieved) - float(b_test)
            if b_test is not None and b_retrieved is not None
            else None
        )
        rows.append(
            {
                "router_key": router_key,
                "prompt_id": prompt_id,
                "test_item_id": test_item_id,
                "retrieved_item_id": retrieved_item_id,
                "rank": rank,
                "b_test": b_test,
                "b_retrieved": b_retrieved,
                "delta_b": delta_b,
                "abs_delta_b": abs(delta_b) if delta_b is not None else None,
                "comparison_arm": "query_embedding_knn",
                "selected_reward_model": selected_reward_model,
                "similarity": similarity,
                "query_text": canonical_row_by_id[test_item_id].input,
            }
        )
    return rows


def build_random_baseline_rows(
    *,
    router_key: str,
    test_item_id: str,
    reward_rows: list[dict[str, Any]],
    train_item_ids: list[str],
    canonical_row_by_id: dict[str, Any],
    difficulty_lookup: dict[str, dict[str, Any]],
    top_k: int,
    random_repeats: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    available_train_item_ids = [
        item_id
        for item_id in train_item_ids
        if item_id in difficulty_lookup
        and difficulty_lookup[item_id]["difficulty"] is not None
    ]
    if not available_train_item_ids or top_k <= 0:
        return []
    reward_row = reward_rows[0]
    prompt_id = str(reward_row["prompt_id"])
    selected_reward_model = reward_row.get("selected_model_name")
    b_test = reward_row.get("b_test")
    query_text = canonical_row_by_id[test_item_id].input
    sample_size = min(top_k, len(available_train_item_ids))
    rows: list[dict[str, Any]] = []
    for repeat_index in range(random_repeats):
        sampled_item_ids = rng.sample(available_train_item_ids, sample_size)
        for rank, retrieved_item_id in enumerate(sampled_item_ids, start=1):
            b_retrieved = difficulty_lookup[retrieved_item_id]["difficulty"]
            delta_b = (
                float(b_retrieved) - float(b_test)
                if b_test is not None and b_retrieved is not None
                else None
            )
            rows.append(
                {
                    "router_key": router_key,
                    "prompt_id": prompt_id,
                    "test_item_id": test_item_id,
                    "retrieved_item_id": retrieved_item_id,
                    "rank": rank,
                    "repeat_index": repeat_index,
                    "b_test": b_test,
                    "b_retrieved": b_retrieved,
                    "delta_b": delta_b,
                    "abs_delta_b": abs(delta_b) if delta_b is not None else None,
                    "comparison_arm": "random",
                    "selected_reward_model": selected_reward_model,
                    "query_text": query_text,
                }
            )
    return rows


def build_per_query_summary(
    *,
    reward_rows: list[dict[str, Any]],
    query_embedding_rows: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    reward_row = reward_rows[0]
    reward_values = extract_matched_values(reward_rows)
    query_values = extract_matched_values(query_embedding_rows)
    random_values = extract_matched_values(random_rows)
    payload = {
        "router_key": reward_row["router_key"],
        "prompt_id": reward_row["prompt_id"],
        "test_item_id": reward_row["test_item_id"],
        "query_text": reward_row.get("query_text"),
        "selected_reward_model": reward_row.get("selected_model_name"),
        "reward_neighbor_count": len(reward_rows),
        "query_embedding_knn_neighbor_count": len(query_embedding_rows),
        "random_neighbor_count": len(random_rows),
        "reward_topk_abs_delta_b": summarize_values(reward_values),
        "query_embedding_knn_abs_delta_b": summarize_values(query_values),
        "random_abs_delta_b": summarize_values(random_values),
        "near_rates": {},
    }
    for factor_key, threshold in thresholds.items():
        payload["near_rates"][factor_key] = {
            "threshold": threshold,
            "reward_topk": aggregate_near_rate(reward_values, threshold),
            "query_embedding_knn": aggregate_near_rate(query_values, threshold),
            "random": aggregate_near_rate(random_values, threshold),
        }
    payload["pairwise_arm_comparisons"] = {
        "reward_lt_random_mean": mean_less_than(reward_values, random_values),
        "reward_lt_query_embedding_mean": mean_less_than(reward_values, query_values),
        "query_embedding_lt_random_mean": mean_less_than(query_values, random_values),
    }
    return payload


def build_comparative_summary(
    *,
    reward_rows: list[dict[str, Any]],
    query_embedding_rows: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    per_query_rows: list[dict[str, Any]],
    train_prompt_values: list[float],
    test_prompt_values: list[float],
    measurement_family: str,
    difficulty_source: str,
    random_repeats: int,
    query_embedding_artifact_path: Path | None,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    reward_values = extract_matched_values(reward_rows)
    query_values = extract_matched_values(query_embedding_rows)
    random_values = extract_matched_values(random_rows)
    by_selected_reward_model: dict[str, list[float]] = defaultdict(list)
    for row in reward_rows:
        model_name = row.get("selected_model_name")
        if (
            isinstance(model_name, str)
            and model_name
            and row.get("abs_delta_b") is not None
        ):
            by_selected_reward_model[model_name].append(float(row["abs_delta_b"]))
    near_rates: dict[str, Any] = {}
    for factor_key, threshold in thresholds.items():
        near_rates[factor_key] = {
            "threshold": threshold,
            "reward_topk": aggregate_near_rate(reward_values, threshold),
            "query_embedding_knn": aggregate_near_rate(query_values, threshold),
            "random": aggregate_near_rate(random_values, threshold),
        }
    return {
        "difficulty_source": difficulty_source,
        "measurement_family": measurement_family,
        "query_embedding_artifact_path": str(query_embedding_artifact_path)
        if query_embedding_artifact_path is not None
        else None,
        "random_repeats": random_repeats,
        "evaluated_queries": len(per_query_rows),
        "reward_neighbor_rows": len(reward_rows),
        "query_embedding_knn_neighbor_rows": len(query_embedding_rows),
        "random_neighbor_rows": len(random_rows),
        "train_b": summarize_values(train_prompt_values),
        "test_b": summarize_values(test_prompt_values),
        "reward_topk_abs_delta_b": summarize_values(reward_values),
        "query_embedding_knn_abs_delta_b": summarize_values(query_values),
        "random_abs_delta_b": summarize_values(random_values),
        "near_rates": near_rates,
        "by_selected_reward_model": {
            model_name: summarize_values(values)
            for model_name, values in sorted(by_selected_reward_model.items())
        },
        "pairwise_query_wins": {
            "reward_lt_random_mean": paired_summary(
                per_query_rows,
                left_key="reward_topk_abs_delta_b",
                right_key="random_abs_delta_b",
            ),
            "reward_lt_query_embedding_mean": paired_summary(
                per_query_rows,
                left_key="reward_topk_abs_delta_b",
                right_key="query_embedding_knn_abs_delta_b",
            ),
            "query_embedding_lt_random_mean": paired_summary(
                per_query_rows,
                left_key="query_embedding_knn_abs_delta_b",
                right_key="random_abs_delta_b",
            ),
        },
    }


def build_split_prompt_difficulties(
    *, rows: Sequence[Any], difficulty_lookup: dict[str, dict[str, Any]]
) -> list[float]:
    prompt_to_value: dict[str, float] = {}
    for row in rows:
        difficulty_row = difficulty_lookup.get(row.id)
        if difficulty_row is None or difficulty_row["difficulty"] is None:
            continue
        prompt_to_value.setdefault(
            str(row.prompt_id), float(difficulty_row["difficulty"])
        )
    return list(prompt_to_value.values())


def build_near_rate_thresholds(train_prompt_values: list[float]) -> dict[str, float]:
    if len(train_prompt_values) <= 1:
        return {f"std_factor_{factor:g}": 0.0 for factor in DEFAULT_NEAR_RATE_FACTORS}
    train_mean = sum(train_prompt_values) / len(train_prompt_values)
    variance = sum((value - train_mean) ** 2 for value in train_prompt_values) / len(
        train_prompt_values
    )
    train_std = math.sqrt(variance)
    return {
        f"std_factor_{factor:g}": factor * train_std
        for factor in DEFAULT_NEAR_RATE_FACTORS
    }


def load_query_embedding_artifact(
    path: Path, *, canonical_row_by_id: dict[str, Any], read_jsonl_objects: Any
) -> dict[str, list[float]]:
    if not path.exists():
        raise ValueError(f"Query embedding artifact does not exist: {path}")
    rows = read_jsonl_objects(path)
    lookup: dict[str, list[float]] = {}
    expected_dimension: int | None = None
    for line_number, row in enumerate(rows, start=1):
        item_id = row.get("item_id")
        embedding = row.get("embedding")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(
                f"Query embedding artifact item_id must be a non-empty string: {path} line {line_number}"
            )
        if item_id not in canonical_row_by_id:
            raise ValueError(
                f"Query embedding artifact item_id '{item_id}' is not present in the Stage 1 canonical dataset: {path} line {line_number}"
            )
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(
                f"Query embedding artifact embedding must be a non-empty array: {path} line {line_number}"
            )
        values: list[float] = []
        for value in embedding:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(
                    f"Query embedding artifact embedding values must be numeric: {path} line {line_number}"
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"Query embedding artifact embedding values must be finite: {path} line {line_number}"
                )
            values.append(float(value))
        if item_id in lookup:
            raise ValueError(
                f"Query embedding artifact contains duplicate item_id '{item_id}': {path} line {line_number}"
            )
        if expected_dimension is None:
            expected_dimension = len(values)
        elif len(values) != expected_dimension:
            raise ValueError(
                f"Query embedding artifact embedding dimensions must be consistent: {path} line {line_number}"
            )
        lookup[item_id] = values
    return lookup


def build_missing_reason(
    *, b_test: float | None, b_retrieved: float | None
) -> str | None:
    if b_test is None and b_retrieved is None:
        return "missing_test_and_retrieved_difficulty"
    if b_test is None:
        return "missing_test_difficulty"
    if b_retrieved is None:
        return "missing_retrieved_difficulty"
    return None


def build_router_metrics(
    *, router_key: str, alignment_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    matched_values = extract_matched_values(alignment_rows)
    row_count = len(alignment_rows)
    matched_row_count = len(matched_values)
    missing_row_count = row_count - matched_row_count
    return {
        "router_key": router_key,
        "row_count": row_count,
        "matched_row_count": matched_row_count,
        "missing_row_count": missing_row_count,
        "unique_test_item_count": len(
            {str(row["test_item_id"]) for row in alignment_rows}
        ),
        "unique_retrieved_item_count": len(
            {str(row["retrieved_item_id"]) for row in alignment_rows}
        ),
        "abs_delta_b_mean": mean(matched_values) if matched_values else None,
        "abs_delta_b_median": median(matched_values) if matched_values else None,
        "abs_delta_b_p90": percentile(matched_values, 0.9),
        "abs_delta_b_p95": percentile(matched_values, 0.95),
        "abs_delta_b_max": max(matched_values) if matched_values else None,
    }


def extract_matched_values(rows: Sequence[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        abs_delta_b = row.get("abs_delta_b")
        if abs_delta_b is None:
            continue
        value = float(abs_delta_b)
        if not math.isfinite(value):
            raise ValueError("Matched abs_delta_b values must be finite.")
        values.append(value)
    return values


def summarize_values(values: Sequence[float]) -> dict[str, Any]:
    realized = [float(value) for value in values if math.isfinite(float(value))]
    if not realized:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(realized),
        "mean": mean(realized),
        "median": median(realized),
        "p90": percentile(realized, 0.9),
        "p95": percentile(realized, 0.95),
        "min": min(realized),
        "max": max(realized),
    }


def aggregate_near_rate(values: Sequence[float], threshold: float) -> float | None:
    realized = [float(value) for value in values if math.isfinite(float(value))]
    if not realized:
        return None
    return float(sum(value <= threshold for value in realized) / len(realized))


def mean_less_than(
    left_values: Sequence[float], right_values: Sequence[float]
) -> bool | None:
    if not left_values or not right_values:
        return None
    return float(mean(left_values)) < float(mean(right_values))


def paired_summary(
    rows: Sequence[dict[str, Any]], *, left_key: str, right_key: str
) -> dict[str, Any]:
    comparable = 0
    left_wins = 0
    right_wins = 0
    ties = 0
    for row in rows:
        left_summary = row.get(left_key)
        right_summary = row.get(right_key)
        if not isinstance(left_summary, dict) or not isinstance(right_summary, dict):
            continue
        left_mean = left_summary.get("mean")
        right_mean = right_summary.get("mean")
        if left_mean is None or right_mean is None:
            continue
        comparable += 1
        if float(left_mean) < float(right_mean):
            left_wins += 1
        elif float(right_mean) < float(left_mean):
            right_wins += 1
        else:
            ties += 1
    return {
        "comparable_query_count": comparable,
        "left_wins": left_wins,
        "right_wins": right_wins,
        "ties": ties,
    }


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    if lower_index == upper_index:
        return lower_value
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def write_router_ecdf_plot(
    path: Path, *, router_key: str, alignment_rows: list[dict[str, Any]]
) -> None:
    matched_values = sorted(extract_matched_values(alignment_rows))
    if not matched_values:
        return
    x_values = [0.0, *matched_values]
    y_values = [
        0.0,
        *[(index + 1) / len(matched_values) for index in range(len(matched_values))],
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(x_values, y_values, linewidth=2)
    axis.set_xlabel("|Rasch beta gap|")
    axis.set_ylabel("ECDF")
    axis.set_title(f"{router_key} Difficulty Gap")
    axis.grid(True, alpha=0.3)
    axis.set_ylim(0.0, 1.02)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def write_comparative_plots(
    *,
    analysis_root: Path,
    reward_rows: list[dict[str, Any]],
    query_embedding_rows: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    measurement_family: str,
) -> bool:
    reward_values = sorted(extract_matched_values(reward_rows))
    query_values = sorted(extract_matched_values(query_embedding_rows))
    random_values = sorted(extract_matched_values(random_rows))
    if not reward_values and not query_values and not random_values:
        return False
    label = "Rasch beta" if measurement_family != "mirt" else "MIRT b"
    figure, axis = plt.subplots(figsize=(8, 5))
    if reward_values:
        axis.plot(
            [0.0, *reward_values],
            [
                0.0,
                *[
                    (index + 1) / len(reward_values)
                    for index in range(len(reward_values))
                ],
            ],
            label="Reward Router top-k",
        )
    if query_values:
        axis.plot(
            [0.0, *query_values],
            [
                0.0,
                *[
                    (index + 1) / len(query_values)
                    for index in range(len(query_values))
                ],
            ],
            label="Query embedding kNN",
        )
    if random_values:
        axis.plot(
            [0.0, *random_values],
            [
                0.0,
                *[
                    (index + 1) / len(random_values)
                    for index in range(len(random_values))
                ],
            ],
            label="Random top-k",
            linestyle="--",
        )
    axis.set_xlabel(f"|{label} gap|")
    axis.set_ylabel("ECDF")
    axis.set_title("Difficulty Gap of Retrieved Neighbors")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(analysis_root / "abs_delta_b_ecdf.png")
    plt.close(figure)
    all_values = [*reward_values, *query_values, *random_values]
    if not all_values:
        return False
    bins = min(60, max(10, int(math.sqrt(len(all_values)))))
    figure, axis = plt.subplots(figsize=(8, 5))
    if reward_values:
        axis.hist(reward_values, bins=bins, alpha=0.72, label="Reward Router top-k")
    if query_values:
        axis.hist(query_values, bins=bins, alpha=0.55, label="Query embedding kNN")
    if random_values:
        axis.hist(random_values, bins=bins, alpha=0.45, label="Random top-k")
    axis.set_xlabel(f"|{label} gap|")
    axis.set_ylabel("Count")
    axis.set_title("Distribution of Difficulty Gaps")
    axis.legend()
    figure.tight_layout()
    figure.savefig(analysis_root / "abs_delta_b_hist.png")
    plt.close(figure)
    return True


def write_router_comparison_ecdf_plot(
    path: Path,
    *,
    reward_rows: list[dict[str, Any]],
    measurement_family: str,
) -> bool:
    rows_by_router: dict[str, list[float]] = defaultdict(list)
    for row in reward_rows:
        abs_delta_b = row.get("abs_delta_b")
        router_key = row.get("router_key")
        if (
            abs_delta_b is None
            or not isinstance(router_key, str)
            or not router_key.strip()
        ):
            continue
        value = float(abs_delta_b)
        if not math.isfinite(value):
            continue
        rows_by_router[router_key].append(value)
    if not rows_by_router:
        return False
    label = "Rasch beta" if measurement_family != "mirt" else "MIRT b"
    figure, axis = plt.subplots(figsize=(8, 5))
    for router_key in sorted(rows_by_router):
        values = sorted(rows_by_router[router_key])
        axis.plot(
            [0.0, *values],
            [0.0, *[(index + 1) / len(values) for index in range(len(values))]],
            label=router_key,
        )
    axis.set_xlabel(f"|{label} gap|")
    axis.set_ylabel("ECDF")
    axis.set_title("Difficulty Gap by Router")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return True

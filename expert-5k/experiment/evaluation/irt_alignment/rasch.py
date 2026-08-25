from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

DEFAULT_SCORE_EPSILON = 1e-6
DEFAULT_RASCH_TRAIN_STEPS = 250
DEFAULT_RASCH_TEST_STEPS = 150
DEFAULT_RASCH_LEARNING_RATE = 0.05
DEFAULT_RASCH_INIT_SD = 0.2
DEFAULT_RASCH_MIN_SD = 1e-4
DEFAULT_RASCH_SIGMA_THETA = 1.0
DEFAULT_RASCH_SIGMA_BETA = 1.0
DEFAULT_RASCH_GRAD_CLIP = 1.0
DEFAULT_RASCH_MC_SAMPLES = 1


@dataclass(slots=True)
class DifficultySurface:
    rows: list[dict[str, Any]]
    lookup: dict[str, dict[str, Any]]
    source: str
    manifest_path: str
    measurement_family: str


@dataclass(slots=True)
class _RaschResponseData:
    prompts: list[str]
    models: list[str]
    prompt_ids: Any
    model_ids: Any
    labels: Any

    @property
    def rows(self) -> int:
        return int(self.labels.numel())


def resolve_difficulty_surface(
    *,
    difficulty_artifact_path: Path | None,
    measurement_family: str,
    canonical_rows: list[Any],
    train_rows: list[Any],
    test_rows: list[Any],
    canonical_row_by_id: dict[str, Any],
    seed: int,
) -> DifficultySurface:
    if difficulty_artifact_path is not None:
        resolved_path = difficulty_artifact_path.resolve()
        difficulty_rows = load_difficulty_artifact(
            resolved_path, canonical_row_by_id=canonical_row_by_id
        )
        difficulty_lookup = {row["item_id"]: row for row in difficulty_rows}
        return DifficultySurface(
            rows=difficulty_rows,
            lookup=difficulty_lookup,
            source="loaded",
            manifest_path=str(resolved_path),
            measurement_family=measurement_family,
        )
    if measurement_family == "bayesian_rasch":
        return fit_bayesian_rasch_surface(
            canonical_rows=canonical_rows,
            train_rows=train_rows,
            test_rows=test_rows,
            seed=seed,
        )
    if measurement_family == "rasch_score_fallback":
        difficulty_rows = fit_difficulties_from_stage1_rows(canonical_rows)
        difficulty_lookup = {row["item_id"]: row for row in difficulty_rows}
        return DifficultySurface(
            rows=difficulty_rows,
            lookup=difficulty_lookup,
            source="fit",
            manifest_path="analysis/irt_alignment/rasch_difficulties.jsonl",
            measurement_family="rasch_score_fallback",
        )
    raise ValueError(
        "measurement_family='mirt' requires a prefit difficulty artifact; fitting MIRT is not supported by this Stage 3-1 runner."
    )


def load_difficulty_artifact(
    path: Path, *, canonical_row_by_id: dict[str, Any]
) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"Difficulty artifact does not exist: {path}")
    rows = _read_jsonl_objects(path)
    difficulty_rows: list[dict[str, Any]] = []
    seen_item_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        for required_key in ("item_id", "difficulty", "difficulty_se"):
            if required_key not in row:
                raise ValueError(
                    f"Difficulty artifact is missing required key '{required_key}': {path} line {line_number}"
                )
        item_id = row["item_id"]
        difficulty = row["difficulty"]
        difficulty_se = row["difficulty_se"]
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(
                f"Difficulty artifact item_id must be a non-empty string: {path} line {line_number}"
            )
        if item_id not in canonical_row_by_id:
            raise ValueError(
                f"Difficulty artifact item_id '{item_id}' is not present in the Stage 1 canonical dataset: {path} line {line_number}"
            )
        if item_id in seen_item_ids:
            raise ValueError(f"Duplicate difficulty item_id '{item_id}' in {path}")
        if difficulty is not None and (
            isinstance(difficulty, bool) or not isinstance(difficulty, int | float)
        ):
            raise ValueError(
                f"Difficulty artifact difficulty must be numeric or null: {path} line {line_number}"
            )
        if difficulty_se is not None and (
            isinstance(difficulty_se, bool)
            or not isinstance(difficulty_se, int | float)
        ):
            raise ValueError(
                f"Difficulty artifact difficulty_se must be numeric or null: {path} line {line_number}"
            )
        if isinstance(difficulty, int | float) and not math.isfinite(float(difficulty)):
            raise ValueError(
                f"Difficulty artifact difficulty must be finite: {path} line {line_number}"
            )
        if difficulty_se is not None and not math.isfinite(float(difficulty_se)):
            raise ValueError(
                f"Difficulty artifact difficulty_se must be finite or null: {path} line {line_number}"
            )
        seen_item_ids.add(item_id)
        difficulty_rows.append(
            {
                "item_id": item_id,
                "difficulty": float(difficulty) if difficulty is not None else None,
                "difficulty_se": (
                    float(difficulty_se) if difficulty_se is not None else None
                ),
            }
        )
    return sorted(difficulty_rows, key=lambda row: row["item_id"])


def fit_difficulties_from_stage1_rows(
    canonical_rows: list[Any],
) -> list[dict[str, Any]]:
    fitted_rows: list[dict[str, Any]] = []
    for row in sorted(canonical_rows, key=lambda item: item.id):
        clipped_score = min(
            max(float(row.score), DEFAULT_SCORE_EPSILON), 1.0 - DEFAULT_SCORE_EPSILON
        )
        difficulty = math.log((1.0 - clipped_score) / clipped_score)
        fitted_rows.append(
            {
                "item_id": row.id,
                "difficulty": float(difficulty),
                "difficulty_se": None,
            }
        )
    return fitted_rows


def fit_bayesian_rasch_surface(
    *, canonical_rows: list[Any], train_rows: list[Any], test_rows: list[Any], seed: int
) -> DifficultySurface:
    train_data = _build_rasch_response_data(train_rows)
    if train_data.rows == 0:
        raise ValueError("No usable train Rasch response rows for bayesian_rasch fit.")
    test_data = _build_rasch_response_data(test_rows, model_order=train_data.models)
    if test_data.rows == 0:
        raise ValueError("No usable test Rasch response rows for bayesian_rasch fit.")

    train_fit = _fit_train_rasch(train_data=train_data, seed=seed)
    test_fit = _infer_test_rasch(
        test_data=test_data,
        theta_mu_cpu=train_fit["theta_mu"],
        seed=seed,
    )

    prompt_difficulty_by_prompt_id: dict[str, tuple[float, float]] = {}
    for prompt_id, beta_mu, beta_sd in zip(
        train_data.prompts,
        train_fit["beta_mu"].tolist(),
        train_fit["beta_sd"].tolist(),
        strict=True,
    ):
        prompt_difficulty_by_prompt_id[prompt_id] = (float(beta_mu), float(beta_sd))
    for prompt_id, beta_mu, beta_sd in zip(
        test_data.prompts,
        test_fit["beta_mu"].tolist(),
        test_fit["beta_sd"].tolist(),
        strict=True,
    ):
        prompt_difficulty_by_prompt_id[prompt_id] = (float(beta_mu), float(beta_sd))

    fitted_rows: list[dict[str, Any]] = []
    for row in sorted(canonical_rows, key=lambda item: item.id):
        beta_stats = prompt_difficulty_by_prompt_id.get(row.prompt_id)
        if beta_stats is None:
            fitted_rows.append(
                {"item_id": row.id, "difficulty": None, "difficulty_se": None}
            )
            continue
        difficulty, difficulty_se = beta_stats
        fitted_rows.append(
            {
                "item_id": row.id,
                "difficulty": float(difficulty),
                "difficulty_se": float(difficulty_se),
            }
        )
    difficulty_lookup = {row["item_id"]: row for row in fitted_rows}
    return DifficultySurface(
        rows=fitted_rows,
        lookup=difficulty_lookup,
        source="fit",
        manifest_path="analysis/irt_alignment/rasch_difficulties.jsonl",
        measurement_family="bayesian_rasch",
    )


def _build_rasch_response_data(
    rows: Sequence[Any], model_order: Sequence[str] | None = None
) -> _RaschResponseData:
    import torch

    if model_order is None:
        models = sorted({_extract_raw_model_name(row) for row in rows})
    else:
        models = [str(model_name) for model_name in model_order]
    model_index = {model_name: index for index, model_name in enumerate(models)}
    prompt_ids_raw: list[str] = []
    model_ids_raw: list[int] = []
    labels_raw: list[float] = []
    prompts_seen: set[str] = set()
    prompts: list[str] = []
    prompt_index: dict[str, int] = {}

    for row in rows:
        model_name = _extract_raw_model_name(row)
        if model_name not in model_index:
            continue
        prompt_id = str(row.prompt_id)
        if prompt_id not in prompts_seen:
            prompts_seen.add(prompt_id)
            prompt_index[prompt_id] = len(prompts)
            prompts.append(prompt_id)
        prompt_ids_raw.append(prompt_id)
        model_ids_raw.append(model_index[model_name])
        labels_raw.append(float(row.score))

    return _RaschResponseData(
        prompts=prompts,
        models=models,
        prompt_ids=torch.tensor(
            [prompt_index[prompt_id] for prompt_id in prompt_ids_raw], dtype=torch.long
        ),
        model_ids=torch.tensor(model_ids_raw, dtype=torch.long),
        labels=torch.tensor(labels_raw, dtype=torch.float32),
    )


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _normal_kl(mu: Any, rho: Any, prior_std: float, min_std: float) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as F

    std = F.softplus(rho) + min_std
    prior_var = prior_std * prior_std
    kl = torch.log(torch.tensor(prior_std, device=mu.device, dtype=mu.dtype) / std)
    kl = kl + (std.pow(2) + mu.pow(2)) / (2.0 * prior_var) - 0.5
    return kl.sum(), std


def _init_rasch_means(data: _RaschResponseData) -> tuple[list[float], list[float]]:
    import torch

    eps = 1e-4
    prompt_sums = torch.zeros(len(data.prompts), dtype=torch.float32)
    prompt_counts = torch.zeros(len(data.prompts), dtype=torch.float32)
    model_sums = torch.zeros(len(data.models), dtype=torch.float32)
    model_counts = torch.zeros(len(data.models), dtype=torch.float32)
    prompt_sums.scatter_add_(0, data.prompt_ids, data.labels)
    prompt_counts.scatter_add_(0, data.prompt_ids, torch.ones_like(data.labels))
    model_sums.scatter_add_(0, data.model_ids, data.labels)
    model_counts.scatter_add_(0, data.model_ids, torch.ones_like(data.labels))
    prompt_rate = torch.clamp(
        prompt_sums / torch.clamp(prompt_counts, min=1.0), eps, 1.0 - eps
    )
    model_rate = torch.clamp(
        model_sums / torch.clamp(model_counts, min=1.0), eps, 1.0 - eps
    )
    theta_mu = torch.logit(model_rate)
    beta_mu = -torch.logit(prompt_rate)
    theta_mu = theta_mu - theta_mu.mean()
    beta_mu = beta_mu - beta_mu.mean()
    return theta_mu.tolist(), beta_mu.tolist()


def _fit_train_rasch(*, train_data: _RaschResponseData, seed: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    device = torch.device("cpu")
    prompt_ids = train_data.prompt_ids.to(device)
    model_ids = train_data.model_ids.to(device)
    labels = train_data.labels.to(device)
    n_rows = labels.numel()
    batch_size = min(4096, int(n_rows))
    theta_init, beta_init = _init_rasch_means(train_data)
    theta_mu = torch.nn.Parameter(
        torch.tensor(theta_init, device=device, dtype=torch.float32)
    )
    beta_mu = torch.nn.Parameter(
        torch.tensor(beta_init, device=device, dtype=torch.float32)
    )
    theta_rho = torch.nn.Parameter(
        torch.full(
            (len(train_data.models),),
            _inverse_softplus(DEFAULT_RASCH_INIT_SD),
            device=device,
        )
    )
    beta_rho = torch.nn.Parameter(
        torch.full(
            (len(train_data.prompts),),
            _inverse_softplus(DEFAULT_RASCH_INIT_SD),
            device=device,
        )
    )
    optimizer = torch.optim.Adam(
        [theta_mu, theta_rho, beta_mu, beta_rho], lr=DEFAULT_RASCH_LEARNING_RATE
    )
    likelihood_scale = n_rows / batch_size
    for _ in range(DEFAULT_RASCH_TRAIN_STEPS):
        batch_idx = torch.randint(0, n_rows, (batch_size,), device=device)
        b_prompt = prompt_ids[batch_idx]
        b_model = model_ids[batch_idx]
        b_labels = labels[batch_idx]
        kl_theta, theta_sd = _normal_kl(
            theta_mu, theta_rho, DEFAULT_RASCH_SIGMA_THETA, DEFAULT_RASCH_MIN_SD
        )
        kl_beta, beta_sd = _normal_kl(
            beta_mu, beta_rho, DEFAULT_RASCH_SIGMA_BETA, DEFAULT_RASCH_MIN_SD
        )
        nll = 0.0
        for _sample_index in range(DEFAULT_RASCH_MC_SAMPLES):
            theta = theta_mu[b_model] + theta_sd[b_model] * torch.randn(
                batch_size, device=device
            )
            beta = beta_mu[b_prompt] + beta_sd[b_prompt] * torch.randn(
                batch_size, device=device
            )
            logits = theta - beta
            nll = nll + F.binary_cross_entropy_with_logits(
                logits, b_labels, reduction="sum"
            )
        nll = nll / DEFAULT_RASCH_MC_SAMPLES
        loss = likelihood_scale * nll + kl_theta + kl_beta
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [theta_mu, theta_rho, beta_mu, beta_rho], max_norm=DEFAULT_RASCH_GRAD_CLIP
        )
        optimizer.step()
    with torch.no_grad():
        _, theta_sd = _normal_kl(
            theta_mu, theta_rho, DEFAULT_RASCH_SIGMA_THETA, DEFAULT_RASCH_MIN_SD
        )
        _, beta_sd = _normal_kl(
            beta_mu, beta_rho, DEFAULT_RASCH_SIGMA_BETA, DEFAULT_RASCH_MIN_SD
        )
    return {
        "theta_mu": theta_mu.detach().cpu(),
        "theta_sd": theta_sd.detach().cpu(),
        "beta_mu": beta_mu.detach().cpu(),
        "beta_sd": beta_sd.detach().cpu(),
    }


def _infer_test_rasch(
    *, test_data: _RaschResponseData, theta_mu_cpu: Any, seed: int
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed + 1)
    device = torch.device("cpu")
    prompt_ids = test_data.prompt_ids.to(device)
    model_ids = test_data.model_ids.to(device)
    labels = test_data.labels.to(device)
    theta_mu = theta_mu_cpu.to(device).float()
    n_rows = labels.numel()
    batch_size = min(4096, int(n_rows))
    beta_mu = torch.nn.Parameter(
        torch.zeros(len(test_data.prompts), device=device, dtype=torch.float32)
    )
    beta_rho = torch.nn.Parameter(
        torch.full(
            (len(test_data.prompts),),
            _inverse_softplus(DEFAULT_RASCH_INIT_SD),
            device=device,
        )
    )
    optimizer = torch.optim.Adam([beta_mu, beta_rho], lr=DEFAULT_RASCH_LEARNING_RATE)
    likelihood_scale = n_rows / batch_size
    for _ in range(DEFAULT_RASCH_TEST_STEPS):
        batch_idx = torch.randint(0, n_rows, (batch_size,), device=device)
        b_prompt = prompt_ids[batch_idx]
        b_model = model_ids[batch_idx]
        b_labels = labels[batch_idx]
        kl_beta, beta_sd = _normal_kl(
            beta_mu, beta_rho, DEFAULT_RASCH_SIGMA_BETA, DEFAULT_RASCH_MIN_SD
        )
        nll = 0.0
        for _sample_index in range(DEFAULT_RASCH_MC_SAMPLES):
            beta = beta_mu[b_prompt] + beta_sd[b_prompt] * torch.randn(
                batch_size, device=device
            )
            logits = theta_mu[b_model] - beta
            nll = nll + F.binary_cross_entropy_with_logits(
                logits, b_labels, reduction="sum"
            )
        nll = nll / DEFAULT_RASCH_MC_SAMPLES
        loss = likelihood_scale * nll + kl_beta
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [beta_mu, beta_rho], max_norm=DEFAULT_RASCH_GRAD_CLIP
        )
        optimizer.step()
    with torch.no_grad():
        _, beta_sd = _normal_kl(
            beta_mu, beta_rho, DEFAULT_RASCH_SIGMA_BETA, DEFAULT_RASCH_MIN_SD
        )
    return {
        "beta_mu": beta_mu.detach().cpu(),
        "beta_sd": beta_sd.detach().cpu(),
    }


def _extract_raw_model_name(row: Any) -> str:
    model_name = row.metadata.get("raw_model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(
            "Canonical rows must contain a non-empty metadata.raw_model_name for Stage 3-1 Rasch fitting."
        )
    return model_name


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL rows must be objects in {path} line {line_number}")
        rows.append(payload)
    return rows

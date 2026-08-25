# Difficulty-Aware Router Contract

This document defines the current router-family-specific contract for `difficulty_aware_router`.

It extends the shared Stage 2 contract defined in the repository root `contract.md` and defines family-local Stage 3 runtime-config semantics for this router family.

The shared Stage 2 router family string for this package is:

```text
difficulty-aware-router
```

## Scope

The difficulty-aware family uses the shared Stage 2 manifest for discovery and reload.

The family currently has two Stage 2 trainer entrypoints under the same router family namespace:

- `difficulty_aware_router.training:DifficultyAwareTrainer`
- `difficulty_aware_router.training.example_trainer:ExampleTrainer`

The primary difficulty-aware runtime contract begins at:

```text
<artifact_path>/router.json
```

`DifficultyAwareTrainer.load_router(...)` reconstructs the runtime from that file and additional router-local artifacts under the same artifact root.

The example trainer uses a separate router-local artifact:

```text
<artifact_path>/example_router.json
```

and reconstructs `ExampleRouter` from that file.

## Stage 1 input requirements

Difficulty-aware training consumes the canonical Stage 1 `train.jsonl` only.

Rows used by the family must satisfy the canonical shared Stage 1 row contract.

The primary `DifficultyAwareTrainer` currently requires:

- `metadata.raw_model_name` as a non-empty string
- `score` to be finite and within `[0.0, 1.0]`
- `id`, `prompt_id`, `input`, `input_token`, and `output_tokken` to be present and valid

The example trainer currently requires at least:

- `metadata.raw_model_name` as a non-empty string on at least one Stage 1 row

These are family-specific runtime/training assumptions and are not part of the shared Stage 2 contract.

## Family-specific config surface

The primary difficulty-aware trainer currently accepts and validates these config fields:

- `dry_run`
- `limit`
- `analysis_prompt_version`
- `analysis_model`
- `analysis_embedding_model`
- `embedding_model_name`
- `cache_mode`
- `request_options`

Current behavior notes:

- `dry_run` disables artifact materialization entirely
- `limit` must be a non-negative integer
- `analysis_prompt_version`, `analysis_model`, `analysis_embedding_model`, and `embedding_model_name` are normalized into non-empty identifiers when used
- `analysis_prompt_version` must name a supported `DifficultyAnalysisAgent` prompt registry entry
- `cache_mode` must be either a supported mode or an endpoint-specific object with exactly `chat_completions` and `embeddings`
- `request_options` must be an endpoint-specific object with exactly `chat_completions` and `embeddings`

Alias behavior notes:

- `embedding_model_name` is currently accepted as a fallback input for `analysis_embedding_model`
- `cache_mode="deferred"` can raise `CacheDeferredRequest` before router-local artifacts or the shared Stage 2 manifest are materialized

The example trainer currently accepts `seed` as its family-local config surface.

## Evaluation-time runtime config surface

Stage 3 may provide an optional evaluation-time runtime config entry for a difficulty-aware router after the shared Stage 2 manifest has been resolved.

This runtime config is not part of the shared Stage 2 artifact schema.

It is a family-local runtime override surface owned by the difficulty-aware router.

When supported by the loader/runtime implementation, the evaluation-time runtime config may override runtime settings including:

- `analysis_model`
- `analysis_prompt_version`
- `embedding_model`
- `cache_dir`
- `cache_mode`
- `request_options`
- `top_k`
- `gamma`
- `cost_normalization_scale`

Family-local runtime config rules:

- `cache_mode` may be a single supported cache mode or an endpoint-specific object with exactly `chat_completions` and `embeddings`
- when `cache_dir` is omitted in the direct loader path, runtime falls through to `DifficultyAwareRouter` defaults and uses `<artifact_root>/cache`
- Stage 3 evaluation may synthesize a separate persistent runtime cache path before calling the loader, but that behavior belongs to the evaluation pipeline rather than this family-local loader contract
- explicit `runtime_config.cache_dir` overrides the default loader/runtime cache location
- when `analysis_model` or `embedding_model` is omitted from runtime config, runtime falls through to the `DifficultyAwareRouter` runtime defaults and environment settings rather than training artifact metadata
- `request_options` is endpoint-specific and must be an object with exactly `chat_completions` and `embeddings`
- `request_options.chat_completions` applies to the runtime difficulty analysis chat call over `query_input["query"]`
- `request_options.embeddings` applies to the runtime query-summary embedding call
- when `analysis_prompt_version` is omitted from runtime config, runtime falls through to the training artifact config default
- `gamma` may be overridden at runtime and defaults to the router's built-in scoring default when omitted
- `cost_normalization_scale` is an evaluation-time runtime input used to normalize estimated cost before scoring; Stage 3 may synthesize it from the train split before calling the loader
- this runtime config surface must not redefine the training-time analysis input contract over canonical Stage 1 `input` / runtime `query`
- this runtime config surface must not change artifact-discovered candidate models, Chroma collection topology, or other required router-local files

## Shared Stage 2 integration

The difficulty-aware family must produce a valid shared Stage 2 manifest at the artifact root.

The family-specific loader depends only on `training_manifest.artifact_path` from that shared manifest.

## Required router-local artifacts

The difficulty-aware artifact root must contain at minimum:

- `router.json`
- `candidate_models.json`
- `model_cost_stats.json`
- `difficulty_aware_artifact_manifest.json`
- `chroma/`

The `chroma/` directory must contain the collections referenced by `router.json`.

The example trainer artifact root must contain:

- `example_router.json`

## `candidate_models.json` contract

`candidate_models.json` must be a JSON object with this shape:

```json
{
  "candidate_models": ["model-a", "model-b"]
}
```

Validation rules:

- `candidate_models` must exist
- `candidate_models` must be a non-empty list of non-empty strings

Current behavior note:

- `candidate_models.json` is an emitted family-local artifact
- the current runtime reload path does not read it directly
- the active loader boundary is `router.json` plus the artifacts referenced from `router.json`

## `router.json` contract

`router.json` must be a JSON object containing at least:

```json
{
  "candidate_models": ["model-a", "model-b"],
  "artifact_manifest": "difficulty_aware_artifact_manifest.json",
  "candidate_models_path": "candidate_models.json",
  "chroma_dir": "chroma",
  "collections": {
    "shared": "difficulty_aware_shared",
    "by_model": {
      "model-a": "difficulty_aware_model__...",
      "model-b": "difficulty_aware_model__..."
    }
  },
  "model_cost_stats_path": "model_cost_stats.json"
}
```

Validation rules:

- `candidate_models` must be a non-empty list of non-empty strings
- `artifact_manifest` must be a non-empty string naming an existing file in the artifact root
- `candidate_models_path` is currently emitted as router-local metadata pointing to `candidate_models.json`
- `chroma_dir` must be a non-empty string naming an existing directory in the artifact root
- `collections` must be an object
- `collections.shared` must be a non-empty string
- `collections.by_model` must be a non-empty object mapping model names to collection names
- `collections.by_model` must provide a collection mapping for every model listed in `candidate_models`
- `model_cost_stats_path` must be a non-empty string naming an existing file in the artifact root

Current loader boundary note:

- the active runtime loader reads `candidate_models`, `chroma_dir`, `artifact_manifest`, `collections`, and `model_cost_stats_path` from `router.json`
- it does not currently read `candidate_models_path` directly during reload

## `difficulty_aware_artifact_manifest.json` contract

`difficulty_aware_artifact_manifest.json` is the family-local training metadata manifest.

It currently includes:

- `schema_version`
- `dataset`
- `outputs`
- `counts`
- `config`
- `analysis`

The `analysis` section is training metadata and is not part of the runtime model-selection fallback chain.

## `example_router.json` contract

`example_router.json` must be a JSON object containing at least:

```json
{
  "candidate_models": ["model-a", "model-b"],
  "seed": 7,
  "ranking_algorithm": "random_shuffle"
}
```

Validation rules:

- `candidate_models` must be a non-empty list of non-empty strings
- `seed` must be an integer when present, or `null`
- `ranking_algorithm` is currently emitted as `random_shuffle`

## `model_cost_stats.json` contract

`model_cost_stats.json` must be a JSON object keyed by model name.

Each model entry must be a JSON object whose values are numeric.

The runtime expects this file to be loadable into a mapping of:

- model name
- statistic name
- numeric value

## Runtime contract

The runtime reconstruction path must be able to derive:

- `candidate_models`
- `chroma_path`
- `model_collection_names`
- `shared_collection_name`
- `analysis_model`
- `embedding_model`
- `model_cost_stats`

from the artifact root.

Evaluation-time runtime config may override a family-defined subset of runtime behavior after artifact reload, but it does not change which router-local artifacts are required or how the artifact root is discovered.

The loaded runtime is `DifficultyAwareRouter`.

`DifficultyAwareRouter` requires `query_input["query"]` to be a non-empty string.

When invoked successfully, the runtime returns ranked models plus metadata including at least:

- `strategy`
- `candidate_count`
- `top_k`
- `query_summary`
- `analysis_model`
- `embedding_model`
- `model_rewards`
- `model_neighbor_counts`

The example trainer loads `ExampleRouter` from `example_router.json`.

`ExampleRouter` currently shuffles the candidate models and returns ranked output with metadata including:

- `strategy`
- `candidate_count`
- `implementation`

## Family-specific validation ownership

The difficulty-aware family is responsible for validating:

- the family-specific training config
- Stage 1 row requirements beyond the shared row contract
- required router-local files and directories
- `router.json` structure
- `example_router.json` structure when the example trainer is used
- `difficulty_aware_artifact_manifest.json` structure
- `model_cost_stats.json` structure
- the existence of referenced Chroma collections
- the ability to reconstruct the runtime from the artifact root

These checks are family-local and must not be promoted into the shared Stage 2 contract.

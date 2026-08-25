from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import run_dataset_pipeline
from .settings import get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.generate_dataset",
        description="Generate canonical Stage 1 dataset artifacts.",
    )
    parser.add_argument("--source-adapter", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--version")
    parser.add_argument(
        "--llm-config-path",
        default="config/llm.json",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-groups", type=int, default=None)
    parser.add_argument("--max-prompt-groups", type=int, default=None)
    parser.add_argument("--max-evaluation-order", type=int, default=1)
    parser.add_argument("--source-split", default="train")
    parser.add_argument(
        "--score-mode",
        default="pairwise_binary",
    )
    parser.add_argument(
        "--source-config-summary-json",
        default="{}",
        help=(
            "JSON object string passed into Stage 1 source_config_summary and "
            "recorded in manifest.inputs.source_config_summary. This surface can "
            "also control occupational_tags OOD splitting, for example: "
            '{"occupational_tags": ["legal"]}'
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    version = (
        args.version or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    settings = get_settings()
    source_config_summary = json.loads(args.source_config_summary_json)
    if not isinstance(source_config_summary, dict):
        raise ValueError("--source-config-summary-json must decode to a JSON object.")

    result = run_dataset_pipeline(
        source_adapter=args.source_adapter,
        dataset_id=args.dataset_id,
        version=version,
        llm_config_path=Path(args.llm_config_path),
        test_fraction=args.test_fraction,
        seed=args.seed,
        max_model_groups=args.max_model_groups,
        max_prompt_groups=args.max_prompt_groups,
        max_evaluation_order=args.max_evaluation_order,
        source_split=args.source_split,
        source_config_summary=source_config_summary,
        score_mode=args.score_mode,
        settings=settings,
    )

    print(
        json.dumps(
            {
                "artifact_root": str(result.artifact_root),
                "full": str(result.full_path),
                "train": str(result.train_path),
                "test": str(result.test_path),
                "manifest": str(result.manifest_path),
                "counts": {
                    "total_rows": result.counts.total_rows,
                    "train_rows": result.counts.train_rows,
                    "test_rows": result.counts.test_rows,
                },
                "metrics": {
                    "average_models_per_prompt": result.metrics.average_models_per_prompt,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import random
from typing import Any

from experiment.training.base import RankedModel, RouterResult, UnifiedRouterBase


class ExampleRouter(UnifiedRouterBase):
    def __init__(
        self,
        candidate_models: list[str],
        seed: int | None = None,
    ) -> None:
        super().__init__(model=None, yaml_path=None, resources=candidate_models)
        if not candidate_models:
            raise ValueError("candidate_models must not be empty.")
        self.candidate_models = list(candidate_models)
        self._random = random.Random(seed)

    def route_single_ranked(self, query_input: dict[str, Any]) -> RouterResult:
        ranked_names = list(self.candidate_models)
        self._random.shuffle(ranked_names)
        total = len(ranked_names)
        ranked_models = [
            RankedModel(model_name=model_name, score=float(total - index))
            for index, model_name in enumerate(ranked_names)
        ]
        return RouterResult(
            ranked_models=ranked_models,
            metadata={
                "strategy": "random_shuffle",
                "candidate_count": total,
                "implementation": "example_router",
            },
        )

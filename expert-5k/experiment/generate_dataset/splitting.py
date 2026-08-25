from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from .contracts import CanonicalRow, SplitConfig


def split_rows(
    rows: list[CanonicalRow], split_config: SplitConfig
) -> tuple[list[CanonicalRow], list[CanonicalRow]]:
    if not rows:
        return [], []

    if split_config.strategy == "occupational_tag_ood":
        return _split_rows_by_occupational_tag_ood(rows, split_config)

    groups: dict[str, list[CanonicalRow]] = defaultdict(list)
    for row in rows:
        groups[row.prompt_id].append(row)

    prompt_ids = sorted(groups)
    rng = random.Random(split_config.seed)
    rng.shuffle(prompt_ids)

    if len(prompt_ids) == 1:
        return [row for row in rows], []

    target_test_groups = max(1, round(len(prompt_ids) * split_config.test_fraction))
    target_test_groups = min(target_test_groups, len(prompt_ids) - 1)
    test_ids = set(prompt_ids[:target_test_groups])
    train_ids = set(prompt_ids[target_test_groups:])

    train_rows = [row for row in rows if row.prompt_id in train_ids]
    test_rows = [row for row in rows if row.prompt_id in test_ids]
    return train_rows, test_rows


def _split_rows_by_occupational_tag_ood(
    rows: list[CanonicalRow], split_config: SplitConfig
) -> tuple[list[CanonicalRow], list[CanonicalRow]]:
    selected_tags = set(split_config.occupational_tags or ())
    if not selected_tags:
        raise ValueError(
            "occupational_tag_ood split requires split_config.occupational_tags to be set."
        )

    groups: dict[str, list[CanonicalRow]] = defaultdict(list)
    for row in rows:
        groups[row.prompt_id].append(row)

    prompt_tags_by_id = {
        prompt_id: _extract_prompt_occupational_tags(prompt_rows)
        for prompt_id, prompt_rows in groups.items()
    }

    test_ids = {
        prompt_id
        for prompt_id, prompt_tags in prompt_tags_by_id.items()
        if prompt_tags & selected_tags
    }
    test_tags = set().union(*(prompt_tags_by_id[prompt_id] for prompt_id in test_ids))

    expanded = True
    while expanded:
        expanded = False
        for prompt_id, prompt_tags in prompt_tags_by_id.items():
            if prompt_id in test_ids or not (prompt_tags & test_tags):
                continue
            test_ids.add(prompt_id)
            test_tags.update(prompt_tags)
            expanded = True

    train_ids: set[str] = set()
    for prompt_id, prompt_rows in groups.items():
        prompt_tags = prompt_tags_by_id[prompt_id]
        if prompt_id not in test_ids and not (prompt_tags & test_tags):
            train_ids.add(prompt_id)

    if not test_ids:
        raise ValueError(
            "occupational_tags did not match any prompt groups for the occupational_tag_ood split."
        )
    if not train_ids:
        raise ValueError(
            "occupational_tags matched every prompt group, which would leave the train split empty."
        )

    train_rows = [row for row in rows if row.prompt_id in train_ids]
    test_rows = [row for row in rows if row.prompt_id in test_ids]
    _validate_occupational_tag_disjointness(train_rows, test_rows)
    return train_rows, test_rows


def _extract_prompt_occupational_tags(rows: list[CanonicalRow]) -> set[str]:
    tags: set[str] = set()
    for row in rows:
        tags.update(_extract_row_occupational_tags(row.metadata))
    return tags


def _extract_row_occupational_tags(metadata: dict[str, Any]) -> set[str]:
    raw_tags = metadata.get("occupational_tags")
    if not isinstance(raw_tags, list):
        return set()

    normalized_tags: set[str] = set()
    for raw_tag in raw_tags:
        if isinstance(raw_tag, str):
            tag = raw_tag.strip()
            if tag:
                normalized_tags.add(tag)
    return normalized_tags


def _validate_occupational_tag_disjointness(
    train_rows: list[CanonicalRow],
    test_rows: list[CanonicalRow],
) -> None:
    train_tags = _extract_prompt_occupational_tags(train_rows)
    test_tags = _extract_prompt_occupational_tags(test_rows)
    overlap = train_tags & test_tags
    if overlap:
        raise ValueError(
            "occupational_tag_ood split leaked occupational_tags across train/test: "
            f"{sorted(overlap)}"
        )


def validate_split_disjointness(
    train_rows: list[CanonicalRow],
    test_rows: list[CanonicalRow],
) -> None:
    overlap = {row.prompt_id for row in train_rows} & {
        row.prompt_id for row in test_rows
    }
    if overlap:
        raise ValueError(
            f"prompt_id values must not appear in multiple splits: {sorted(overlap)}"
        )

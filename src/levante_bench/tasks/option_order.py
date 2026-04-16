"""Shared helpers for deterministic option ordering in task loaders."""

from __future__ import annotations

import random
from collections.abc import Sequence


def deterministic_option_order(
    *,
    answer: str,
    alternatives: Sequence[str],
    seed_value: object,
    option_labels: Sequence[str],
) -> tuple[list[str], str]:
    """Return shuffled options and the corresponding correct label.

    The shuffle is pseudo-random but repeatable for a given ``seed_value``.
    """
    all_options = [answer] + list(alternatives)
    rng = random.Random(seed_value)
    rng.shuffle(all_options)

    correct_idx = all_options.index(answer)
    correct_label = str(option_labels[correct_idx])
    return all_options, correct_label


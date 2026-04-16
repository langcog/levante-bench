"""Shared helpers for deterministic option ordering in task loaders."""

from __future__ import annotations

import random
from collections.abc import Sequence


def deterministic_option_order(
    *,
    answer: str,
    alternatives: Sequence[str],
    seed_value: object | None = None,
    option_labels: Sequence[str],
    true_random: bool = False,
) -> tuple[list[str], str]:
    """Return shuffled options and the corresponding correct label.

    By default, the shuffle is pseudo-random but repeatable for a given
    ``seed_value``.

    Set ``true_random=True`` to use system entropy for a non-repeatable
    permutation each call.
    """
    all_options = [answer] + list(alternatives)
    if true_random:
        rng: random.Random = random.SystemRandom()
    else:
        if seed_value is None:
            raise ValueError("seed_value is required when true_random is False.")
        rng = random.Random(seed_value)
    rng.shuffle(all_options)

    correct_idx = all_options.index(answer)
    correct_label = str(option_labels[correct_idx])
    return all_options, correct_label


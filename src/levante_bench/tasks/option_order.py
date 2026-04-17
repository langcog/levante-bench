"""Shared helpers for deterministic option ordering in task loaders."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence


def derive_true_random_item_seed(*, run_seed: int, item_key: object) -> int:
    """Derive a stable per-item seed for one true-random run."""
    key = f"{run_seed}|{item_key}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def deterministic_option_order(
    *,
    answer: str,
    alternatives: Sequence[str],
    seed_value: object | None = None,
    option_labels: Sequence[str],
    true_random: bool = False,
    true_random_seed: int | None = None,
) -> tuple[list[str], str, str]:
    """Return shuffled options and the corresponding correct label.

    By default, the shuffle is pseudo-random but repeatable for a given
    ``seed_value``.

    Set ``true_random=True`` to randomize using ``true_random_seed``.
    """
    all_options = [answer] + list(alternatives)
    if true_random:
        if true_random_seed is None:
            true_random_seed = random.SystemRandom().getrandbits(63)
        rng = random.Random(true_random_seed)
        seed_used = str(true_random_seed)
    else:
        if seed_value is None:
            raise ValueError("seed_value is required when true_random is False.")
        rng = random.Random(seed_value)
        seed_used = str(seed_value)
    rng.shuffle(all_options)

    correct_idx = all_options.index(answer)
    correct_label = str(option_labels[correct_idx])
    return all_options, correct_label, seed_used


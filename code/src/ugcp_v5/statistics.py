"""Paired statistics used by the fresh formal experiments."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any

import numpy as np


def stable_seed(*parts: str | int) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def exact_mcnemar(full: np.ndarray, other: np.ndarray) -> tuple[int, int, float]:
    better = int(np.sum((full == 1) & (other == 0)))
    worse = int(np.sum((full == 0) & (other == 1)))
    discordant = better + worse
    if discordant == 0:
        return better, worse, 1.0
    tail = min(better, worse)
    probability = 2 * sum(
        math.exp(
            math.lgamma(discordant + 1)
            - math.lgamma(index + 1)
            - math.lgamma(discordant - index + 1)
            - discordant * math.log(2)
        )
        for index in range(tail + 1)
    )
    return better, worse, min(1.0, probability)


def paired_bootstrap_interval(
    full: np.ndarray,
    other: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    difference = full.astype(int) - other.astype(int)
    counts = Counter(int(value) for value in difference)
    probabilities = np.asarray(
        [counts[-1], counts[0], counts[1]], dtype=float
    ) / len(difference)
    samples = np.random.default_rng(seed).multinomial(
        len(difference), probabilities, size=repetitions
    )
    effects = 100 * (samples[:, 2] - samples[:, 0]) / len(difference)
    low, high = np.quantile(effects, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(tests: list[dict[str, Any]]) -> None:
    order = sorted(range(len(tests)), key=lambda index: tests[index]["p_raw"])
    running = 0.0
    total = len(tests)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * float(tests[index]["p_raw"]))
        running = max(running, adjusted)
        tests[index]["p_holm"] = running
        tests[index]["holm_significant_0_05"] = running < 0.05

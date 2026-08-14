"""Shared statistical helpers for experiment summaries."""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Sequence


def summarize(values: Sequence[float]) -> dict[str, float | int | list[float]]:
    """Summarize samples with a normal-approximation 95% confidence interval."""
    if not values:
        raise ValueError("at least one value is required")

    sample_mean = mean(values)
    sample_deviation = stdev(values) if len(values) > 1 else 0.0
    margin = 1.96 * sample_deviation / math.sqrt(len(values))
    return {
        "count": len(values),
        "mean": sample_mean,
        "standard_deviation": sample_deviation,
        "confidence_interval_95": [sample_mean - margin, sample_mean + margin],
    }
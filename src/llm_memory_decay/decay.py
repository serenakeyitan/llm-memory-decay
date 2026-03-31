import math
from datetime import datetime
from enum import Enum


class DecayStrategy(str, Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    STEP = "step"
    NONE = "none"


def exponential_decay(age_days: float, half_life_days: float = 30.0) -> float:
    """Returns weight in [0,1]. At age=half_life_days, weight=0.5."""
    return math.exp(-math.log(2) * age_days / half_life_days)


def linear_decay(age_days: float, max_age_days: float = 90.0) -> float:
    """Returns weight in [0,1]. At age=max_age_days, weight=0.0."""
    return max(0.0, 1.0 - age_days / max_age_days)


def step_decay(age_days: float, thresholds: list[tuple[float, float]] | None = None) -> float:
    """Stepped decay: recent=1.0, medium=0.5, old=0.1"""
    if thresholds is None:
        thresholds = [(7, 1.0), (30, 0.5), (90, 0.1), (float("inf"), 0.0)]
    for max_age, weight in thresholds:
        if age_days <= max_age:
            return weight
    return 0.0


def compute_decay_weight(
    created_at: datetime,
    strategy: DecayStrategy = DecayStrategy.EXPONENTIAL,
    half_life_days: float = 30.0,
    max_age_days: float = 90.0,
    now: datetime | None = None,
) -> float:
    """Compute the decay weight for a memory entry based on its age and strategy."""
    if strategy == DecayStrategy.NONE:
        return 1.0
    if now is None:
        now = datetime.utcnow()
    age_days = (now - created_at).total_seconds() / 86400.0
    if strategy == DecayStrategy.EXPONENTIAL:
        return exponential_decay(age_days, half_life_days)
    elif strategy == DecayStrategy.LINEAR:
        return linear_decay(age_days, max_age_days)
    elif strategy == DecayStrategy.STEP:
        return step_decay(age_days)
    return 1.0

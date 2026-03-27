from datetime import datetime, timedelta

import pytest

from llm_memory_decay.decay import (
    exponential_decay,
    linear_decay,
    step_decay,
    compute_decay_weight,
    DecayStrategy,
)


def test_exponential_decay_at_half_life():
    weight = exponential_decay(age_days=30, half_life_days=30)
    assert abs(weight - 0.5) < 0.001


def test_exponential_decay_at_zero():
    weight = exponential_decay(age_days=0, half_life_days=30)
    assert abs(weight - 1.0) < 0.001


def test_exponential_decay_decreases_monotonically():
    weights = [exponential_decay(age_days=d, half_life_days=30) for d in [0, 10, 30, 60, 90]]
    for i in range(len(weights) - 1):
        assert weights[i] > weights[i + 1]


def test_linear_decay_at_max():
    weight = linear_decay(age_days=90, max_age_days=90)
    assert weight == 0.0


def test_linear_decay_at_zero():
    weight = linear_decay(age_days=0, max_age_days=90)
    assert weight == 1.0


def test_linear_decay_beyond_max_clamps_to_zero():
    weight = linear_decay(age_days=200, max_age_days=90)
    assert weight == 0.0


def test_step_decay_recent():
    weight = step_decay(age_days=3)
    assert weight == 1.0


def test_step_decay_medium():
    weight = step_decay(age_days=15)
    assert weight == 0.5


def test_step_decay_old():
    weight = step_decay(age_days=60)
    assert weight == 0.1


def test_step_decay_very_old():
    weight = step_decay(age_days=200)
    assert weight == 0.0


def test_compute_decay_weight_none_strategy():
    now = datetime.utcnow()
    created = now - timedelta(days=999)
    weight = compute_decay_weight(created, strategy=DecayStrategy.NONE, now=now)
    assert weight == 1.0


def test_compute_decay_weight_exponential():
    now = datetime.utcnow()
    created = now - timedelta(days=30)
    weight = compute_decay_weight(
        created, strategy=DecayStrategy.EXPONENTIAL, half_life_days=30, now=now
    )
    assert abs(weight - 0.5) < 0.01


def test_compute_decay_weight_linear():
    now = datetime.utcnow()
    created = now - timedelta(days=45)
    weight = compute_decay_weight(
        created, strategy=DecayStrategy.LINEAR, max_age_days=90, now=now
    )
    assert abs(weight - 0.5) < 0.01


def test_compute_decay_weight_step():
    now = datetime.utcnow()
    created = now - timedelta(days=15)
    weight = compute_decay_weight(created, strategy=DecayStrategy.STEP, now=now)
    assert weight == 0.5

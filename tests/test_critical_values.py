from __future__ import annotations

import numpy as np

from snrcps import critical_value, load_critical_value_records, simulate_critical_values


def test_stored_critical_values_are_accessible() -> None:
    records = load_critical_value_records()
    assert len(records) >= 5
    assert critical_value(delta=0.10) == critical_value(name="main_0.90")


def test_small_simulation_is_reproducible() -> None:
    first = simulate_critical_values([0.90], paths=1_000, grid_size=50, seed=7)
    second = simulate_critical_values([0.90], paths=1_000, grid_size=50, seed=7)
    np.testing.assert_allclose(first, second)

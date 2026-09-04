from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _compare_csv(actual: Path, expected: Path, atol: float = 1e-5) -> None:
    left = pd.read_csv(actual)
    right = pd.read_csv(expected)
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        raise AssertionError(f"Schema mismatch: {actual.name}")
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]):
            if not np.allclose(left[column], right[column], rtol=1e-7, atol=atol, equal_nan=True):
                raise AssertionError(f"Numerical mismatch: {actual.name}:{column}")
        elif not left[column].fillna("").equals(right[column].fillna("")):
            raise AssertionError(f"Text mismatch: {actual.name}:{column}")


def verify_outputs(root: Path) -> None:
    generated = root / "results" / "demo"
    expected = root / "results" / "expected"
    expected_metrics = json.loads((expected / "forecast_metrics.json").read_text(encoding="utf-8"))
    actual_metrics = json.loads((generated / "forecast_metrics.json").read_text(encoding="utf-8"))
    for key, expected_value in expected_metrics.items():
        actual_value = actual_metrics[key]
        if isinstance(expected_value, (int, float)) and not np.isclose(actual_value, expected_value, rtol=1e-8, atol=1e-8):
            raise AssertionError(f"Metric mismatch: {key}")
    _compare_csv(generated / "local_expert_forecast.csv", expected / "local_expert_forecast.csv")
    _compare_csv(generated / "decision_results.csv", expected / "decision_results.csv", atol=1e-3)
    print("Verification passed: generated numerical outputs match the frozen references.")

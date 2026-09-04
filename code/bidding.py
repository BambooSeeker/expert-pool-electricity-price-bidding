from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


def production_cost(power: np.ndarray, p: dict[str, float]) -> np.ndarray:
    power = np.asarray(power, dtype=float)
    load = np.clip(power / p["Prated"], 0.0, 1.0)
    standard_coal = p["a"] * load**2 + p["b"] * load + p["c"]
    factor = (p["p_coal"] + p["p_trans"]) * (1.0 + p["loss_trans"] / 100.0) * 7.0
    factor /= (1.0 - p["pp_rate"] / 100.0) * (p["coal_hv"] - p["loss_heat"])
    return power * standard_coal * factor * 0.5


def dispatch(prices: np.ndarray, q: np.ndarray, offer: np.ndarray, p: dict[str, float]) -> np.ndarray:
    prices = np.atleast_2d(np.asarray(prices, dtype=float))
    index = np.sum(prices[:, :, None] >= offer[None, None, :], axis=2) - 1
    raw = q[np.clip(index, 0, len(q) - 1)]
    raw[index < 0] = 0.0
    out = np.maximum(raw, p["Pmin"])
    for t in range(1, out.shape[1]):
        out[:, t] = np.minimum(out[:, t], out[:, t - 1] + p["Pup"] / 2.0)
        out[:, t] = np.maximum(out[:, t], out[:, t - 1] + p["Pdown"] / 2.0)
        out[:, t] = np.clip(out[:, t], p["Pmin"], p["Prated"])
    return out


def scenario_margin(prices: np.ndarray, q: np.ndarray, offer: np.ndarray, p: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prices = np.atleast_2d(np.asarray(prices, dtype=float))
    x = dispatch(prices, q, offer, p)
    period = x * prices * 0.5 - production_cost(x, p)
    return period.sum(axis=1), x, period


def lower_tail(values: np.ndarray, probabilities: np.ndarray, mass: float) -> float:
    order = np.argsort(values)
    remaining = float(mass)
    total = 0.0
    for i in order:
        take = min(remaining, float(probabilities[i]))
        total += take * float(values[i])
        remaining -= take
        if remaining <= 1e-12:
            break
    return total / mass


def decode(z: np.ndarray, p: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    logits = z[:9] - np.max(z[:9])
    shares = np.exp(logits)
    shares /= shares.sum()
    widths = 0.05 * p["Prated"] + (p["Prated"] - p["Pmin"] - 9 * 0.05 * p["Prated"]) * shares
    q = np.r_[p["Pmin"], p["Pmin"] + np.cumsum(widths)]
    q[-1] = p["Prated"]
    offer = np.r_[p["price_min"], np.sort(z[9:17]), p["price_max"]]
    return q, offer


def optimize_offer(scenarios: np.ndarray, probabilities: np.ndarray, p: dict, seed: int, risk_weight: float, maxiter: int) -> tuple[np.ndarray, np.ndarray]:
    def objective(z: np.ndarray) -> float:
        q, offer = decode(z, p)
        values, _, _ = scenario_margin(scenarios, q, offer, p)
        expected = float(np.dot(probabilities, values))
        tail = lower_tail(values, probabilities, 0.10)
        return -((1.0 - risk_weight) * expected + risk_weight * tail)

    result = differential_evolution(
        objective,
        bounds=[(-4.0, 4.0)] * 9 + [(p["price_min"], p["price_max"])] * 8,
        seed=seed,
        maxiter=maxiter,
        popsize=7,
        polish=False,
        tol=1e-5,
        workers=1,
    )
    return decode(result.x, p)


def run_decision_demo(forecast: pd.DataFrame, residuals: pd.DataFrame, case_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case = json.loads(case_path.read_text(encoding="utf-8"))
    p = {k: float(v) for k, v in case["parameters"].items()}
    plant_q = np.asarray(case["plant_original_offer"]["output_MW"], dtype=float)
    plant_offer = np.asarray(case["plant_original_offer"]["price_CNY_per_MWh"], dtype=float)
    rcols = [f"residual_t{i:02d}" for i in range(1, 49)]
    residual_matrix = residuals[rcols].to_numpy(dtype=float)
    probabilities = residuals["probability"].to_numpy(dtype=float)
    daily, curves, periods = [], [], []
    for day_index, (day, group) in enumerate(forecast.groupby("target_date", sort=True)):
        group = group.sort_values("period")
        median = group["price_day_ahead_0.5"].to_numpy(dtype=float)
        realized = group["true_price"].to_numpy(dtype=float)
        scenarios = np.clip(median[None, :] + residual_matrix, p["price_min"], p["price_max"])
        q, offer = optimize_offer(scenarios, probabilities, p, 20260804 + day_index, risk_weight=0.20, maxiter=35)
        oracle_q, oracle_offer = optimize_offer(realized[None, :], np.array([1.0]), p, 20260904 + day_index, risk_weight=0.0, maxiter=70)
        policies = {
            "Proposed framework": (q, offer),
            "Plant original strategy": (plant_q, plant_offer),
            "Perfect-information benchmark": (oracle_q, oracle_offer),
        }
        row = {"target_date": day}
        for label, (policy_q, policy_offer) in policies.items():
            total, x, period = scenario_margin(realized[None, :], policy_q, policy_offer, p)
            key = label.lower().replace("-", "_").replace(" ", "_")
            row[f"{key}_margin_CNY"] = float(total[0])
            for segment, (power, price) in enumerate(zip(policy_q, policy_offer), 1):
                curves.append({"target_date": day, "policy": label, "segment": segment, "power_MW": power, "offer_price_CNY_per_MWh": price})
            for t in range(48):
                periods.append({"target_date": day, "period": t + 1, "policy": label, "market_price_CNY_per_MWh": realized[t], "dispatch_MW": x[0, t], "gross_margin_CNY": period[0, t]})
        row["incremental_decision_value_CNY"] = row["proposed_framework_margin_CNY"] - row["plant_original_strategy_margin_CNY"]
        daily.append(row)
    return pd.DataFrame(daily), pd.DataFrame(curves), pd.DataFrame(periods)

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from bidding import run_decision_demo  # noqa: E402
from model import infer_local_expert  # noqa: E402
from verify import verify_outputs  # noqa: E402


def main() -> None:
    demo = ROOT / "data" / "demo"
    out = ROOT / "results" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    frozen = pd.read_csv(demo / "forecast_inputs.csv")
    local = infer_local_expert(
        demo / "model_inputs.csv",
        ROOT / "models" / "Proposed_VMD_h16.pt",
        ROOT / "models" / "Proposed_VMD_h16_preprocessors.joblib",
    )
    local.to_csv(out / "local_expert_forecast.csv", index=False)

    residuals = pd.read_csv(demo / "residual_medoids.csv")
    daily, curves, periods = run_decision_demo(frozen, residuals, ROOT / "configs" / "unit_case.json")
    daily.to_csv(out / "decision_results.csv", index=False)
    curves.to_csv(out / "offer_curves.csv", index=False)
    periods.to_csv(out / "period_results.csv", index=False)

    pred = frozen.pivot(index="target_date", columns="period", values="price_day_ahead_0.5").to_numpy()
    true = frozen.pivot(index="target_date", columns="period", values="true_price").to_numpy()
    nonzero = true != 0
    metrics = {
        "days": int(len(pred)),
        "periods_per_day": 48,
        "MAPE_percent": float(np.mean(np.abs((true[nonzero] - pred[nonzero]) / true[nonzero])) * 100.0),
        "RMSE_CNY_per_MWh": float(np.sqrt(np.mean((true - pred) ** 2))),
        "MAE_CNY_per_MWh": float(np.mean(np.abs(true - pred))),
    }
    (out / "forecast_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=160)
    ax.plot(true.reshape(-1), label="Realized price", color="#222222", linewidth=1.5)
    ax.plot(pred.reshape(-1), label="Released proposed forecast", color="#1f77b4", linewidth=1.3)
    ax.set_xlabel("Half-hour observations across the seven-day demo")
    ax.set_ylabel("CNY/MWh")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "forecast_demo.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=160)
    for policy, group in curves.groupby("policy"):
        mean_curve = group.groupby("segment", as_index=False)[["power_MW", "offer_price_CNY_per_MWh"]].mean()
        ax.plot(mean_curve["power_MW"], mean_curve["offer_price_CNY_per_MWh"], marker="o", linewidth=1.5, label=policy)
    ax.set_xlabel("Output (MW)")
    ax.set_ylabel("Offer price (CNY/MWh)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "offer_demo.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=160)
    for policy, group in periods.groupby("policy", sort=False):
        ax.plot(group["gross_margin_CNY"].to_numpy(), linewidth=1.25, label=policy)
    ax.set_xlabel("Half-hour observations across the seven-day demo")
    ax.set_ylabel("Gross margin (CNY)")
    ax.legend(frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(out / "gross_margin_demo.png", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(daily.to_string(index=False))
    verify_outputs(ROOT)


if __name__ == "__main__":
    main()

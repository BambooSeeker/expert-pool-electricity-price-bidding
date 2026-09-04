# Expert-Pool Coordination of Pretrained Priors and Market Fundamentals for Electricity Price Forecasting and Risk-Aware Spot-Market Bidding

This repository accompanies the manuscript submitted to IEEE Transactions on Smart Grid. It provides a reproducible seven-day demonstration of the reported forecast-to-bid workflow for a Zhejiang day-ahead electricity-market case.

Version 1.0.1 is permanently archived at [Zenodo](https://doi.org/10.5281/zenodo.22291573). The GitHub repository hosts the maintained source, while the Zenodo record fixes the exact files cited by the manuscript.

The package is an inference and decision demonstration. It includes a frozen local VMD sequence-expert checkpoint, the seven-day public demo inputs, the released coordinated forecast curves, whole-day residual medoids, the generator case, and code that evaluates the forecast and solves a constrained ten-segment offer problem. The complete 399-day market table, full training matrices, and unrestricted third-party TSFM checkpoints are not included.

## Quick start

```bash
python -m pip install -r environment/requirements-lock.txt
python run_demo.py
```

The command writes `results/demo/forecast_metrics.json`, `results/demo/local_expert_forecast.csv`, `results/demo/decision_results.csv`, `results/demo/offer_curves.csv`, and three PNG figures. The optimizer uses the released 20 whole-day residual medoids, a 0.20 CVaR weight, a 0.10 lower-tail mass, the ten-segment price bounds of -200 to 800 CNY/MWh, and the generator parameters in `configs/unit_case.json`.

## Scope and interpretation

The released coordinated forecasts are the frozen outputs used for the seven-day demonstration. The local checkpoint is provided for a genuine inference check on the same period. The seven-day output is a compact public demonstration and does not reproduce the full-sample training, validation selection, 58-day test aggregation, or all manuscript figures.

The bidding result is an ex-ante single-unit offer evaluation under the stated market and unit constraints. It does not reconstruct Zhejiang market clearing, SCUC, SCED, network constraints, competing offers, or compensation mechanisms.

## Contents

- `run_demo.py`: one-command demonstration entry point.
- `code/model.py`: frozen local-expert inference.
- `code/bidding.py`: residual-scenario construction and CVaR offer optimization.
- `data/demo/`: seven-day inputs, frozen coordinated forecasts, and residual medoids.
- `models/`: local sequence-expert checkpoint and its preprocessing objects.
- `configs/`: feature groups and the public generator case.
- `docs/`: data, model, and reproducibility cards.
- `results/expected/`: frozen numerical outputs used by the verification step.

## Citation

Please cite the accompanying paper and the archived software release: `https://doi.org/10.5281/zenodo.22291573`. Machine-readable citation metadata are provided in `CITATION.cff`.

## Data and code availability

The complete market data are restricted to the authors and are not redistributed. The public files are limited to the seven-day demonstration and derived artifacts needed to run it. Questions about access to the restricted source data should be directed to the corresponding author listed in the paper.

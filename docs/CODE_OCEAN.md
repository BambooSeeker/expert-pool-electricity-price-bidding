# Code Ocean Capsule

Create the capsule by importing the tagged GitHub release. Use the repository root as the working directory and set the run command to:

```bash
python run_demo.py
```

The capsule requires Python 3.11 and the packages pinned in `environment/requirements-lock.txt`. A standard CPU machine is sufficient. The verified run completes in approximately 30 seconds on the development workstation and writes all public outputs to `results/demo/`.

Before publication, check that the capsule result contains:

- `forecast_metrics.json`;
- `local_expert_forecast.csv`;
- `decision_results.csv`;
- `offer_curves.csv` and `period_results.csv`;
- `forecast_demo.png`, `offer_demo.png`, and `gross_margin_demo.png`;
- the final line `Verification passed` in the run log.

Use `Yet To Be Published` for the article field until the manuscript receives its permanent citation. After publication, update the article metadata and archive a new capsule version without changing the original DOI record.

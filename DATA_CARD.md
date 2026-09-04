# Data Card

## Data included

`data/demo/forecast_inputs.csv` contains 336 half-hour records for seven published demonstration days. It includes the released coordinated forecast interval and the realized day-ahead price used for the public evaluation.

`data/demo/model_inputs.csv` contains the pre-clearing feature rows needed to run the released local sequence expert for the same seven target days. Target prices are kept in the separate evaluation file and are not used by the inference code.

`data/demo/residual_medoids.csv` contains 20 weighted whole-day residual vectors derived from the validation residual distribution. They preserve the 48-period shape of a daily forecast error and are used to form price scenarios for the public bidding demonstration.

## Provenance and restrictions

The files are derived from the Zhejiang provincial spot-market study described in the accompanying manuscript. The full source table and unrestricted training matrices are not redistributed because their access and use are governed by the authors' data arrangements. The demo files are released solely to document the public computational example.

## Known limitations

The demo covers seven days and is not a substitute for the full-sample evaluation. The residual medoids are derived artifacts, and the public package does not permit reconstruction of the original 399 daily samples.

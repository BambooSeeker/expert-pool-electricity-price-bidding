# Reproducibility Scope

The public release supports three checks:

1. Loading the released local-expert parameters and producing a forecast for seven target days.
2. Evaluating the released coordinated forecast against the seven observed price curves.
3. Passing the released coordinated forecasts and weighted whole-day residual medoids into the constrained ten-segment bidding optimizer.

The release does not reproduce the full 399-day data preparation, chronological training and validation, standalone TSFM calls, 58-day aggregate tables, or proprietary market-clearing processes. These boundaries are deliberate and are documented so that the public demo is not mistaken for a full-data replication package.

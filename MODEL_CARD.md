# Model Card

## Model

`models/Proposed_VMD_h16.pt` is the frozen state dictionary of the VMD sequence expert used in the forecasting workflow. It is a residual model with market, historical-price, and calendar branches, recurrent encoders, multi-head attention, and a nonlinear prediction head.

The associated preprocessing object stores the training-fitted imputers, scalers, target transformation, and feature-column map. The public inference code loads these objects without refitting them.

## Intended use

The checkpoint is intended for research inspection and the seven-day demonstration in this repository. It should not be treated as a deployable market-trading system or as a replacement for validation under a new market design.

## Out-of-scope use

Do not infer performance on unrepresented markets, periods, units, or operating regimes from this demo. Third-party Chronos-Bolt, TimesFM, and Moirai checkpoints are not redistributed here; their published model repositories remain the source of those weights.

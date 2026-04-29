# Benchmark Design

The recommended core benchmark is a factorized switched state-space generator. It is controlled enough for rigorous evaluation and rich enough to avoid pure sine-wave shortcuts.

## Scientific Goal

Synthetic data is used here as a measurement instrument. Real industrial datasets motivate the problem, but synthetic trajectories let us vary one factor at a time and know the hidden truth.

A useful benchmark should test:

- linear accessibility of relevant factors,
- geometric faithfulness of neighborhoods,
- factor structure rather than memorized full labels,
- transition awareness,
- robustness to nuisance variation and domain shift.

## FRS: Factorized Regime Sequence

The formal RQ1 generator creates trajectories with regimes composed from latent factors:

- spectral family, such as periodic, damped, multi-periodic, or quasi-aperiodic dynamics,
- coupling level, such as low, medium, or high cross-channel interaction,
- mode as the combination of spectral family and coupling level,
- load as a continuous variable affecting dynamics and noise,
- device and observation effects such as gain, bias, and dropout,
- transition windows caused by regime boundaries.

The key design principle is separation of factors. Each factor should influence the signal through interpretable dynamics, not by directly leaking simple statistics that a baseline can trivially exploit.

## Splitting Policy

Splits must happen at the trajectory, generator-instance, or device level. Randomly splitting overlapping windows causes leakage because nearby windows from the same trajectory can appear in both train and test.

Default split ratio:

```text
train / validation / test = 70 / 15 / 15
```

Every split should include a leakage report.

## Clean, Noisy, And OOD Settings

Clean datasets should expose the intended latent structure with minimal nuisance. Noisy datasets add process noise, observation noise, device variation, channel dropout, and offset shifts.

OOD tests should distinguish:

- nuisance shift, such as new device gain or noise regime,
- held-out parameter ranges,
- held-out factor combinations,
- truly unseen operating modes.

These cases should not be collapsed into one generic "test" condition.

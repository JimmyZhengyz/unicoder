# Model and Training Code Release Plan

This repository currently releases the evaluation pipeline for UniCoder. It does not include UniCoder model weights, training datasets, or the full UniCoder training code used for the paper.

## Current Release

- Unified evaluation wrapper for ChartMimic and UniSVG.
- Model API wrappers for reproducing evaluation with supported local or API-based models.
- Instructions for downloading benchmark data from the original benchmark sources.

## Planned Future Release

We plan to release the following materials after the paper release process is finalized:

- UniCoder model weights.
- UniCoder training scripts and configuration files.
- Additional reproducibility notes for data processing, reward computation, and rollout settings.

## Notes

Benchmark data should be obtained from the original benchmark providers. Users are responsible for following the licenses and terms of use for ChartMimic, UniSVG, and any external model APIs or checkpoints used during evaluation.

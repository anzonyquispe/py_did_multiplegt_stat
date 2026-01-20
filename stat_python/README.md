# py_did_multiplegt_stat

Python implementation of the `did_multiplegt_stat` Stata/R command developed by Chaisemartin & D'Haultfœuille.

## Description

This package estimates causal effects in difference-in-differences designs with **continuous treatment** and **multiple time periods**.

## Implemented Features

### Estimators
- **AOSS** (Average of Switchers' Slopes)
- **WAOSS/WAS** (Weighted Average of Switchers' Slopes)  
- **IV-WAOSS** (Instrumental Variable WAS)

### Estimation Methods
- **DR** (Doubly Robust)
- **RA** (Regression Adjustment)
- **PS** (Propensity Score)

### Options
- `placebo` - Parallel trends tests
- `aoss_vs_waoss` - Test of difference between estimators
- `noextrapolation` - Common support restriction
- `order` - Polynomial order
- `cluster` - Clustered standard errors
- `weight` - Sampling weights

## Project Structure

| File | Description |
|------|-------------|
| `did_multiplegt_stat.ipynb` | Main function (wrapper) |
| `did_multiplegt_stat_main.ipynb` | Aggregation logic |
| `did_multiplegt_stat_pairwise.ipynb` | Pairwise (g,t) estimation |
| `did_multiplegt_stat_quantiles.ipynb` | Quantile binning |
| `utils.ipynb` | Utility functions |
| `stata_logit.ipynb` | GLM logit for propensity scores |
| `lpredict.ipynb` | Linear prediction |
| `print.ipynb` | Print functions |
| `did_multiplegt_stat_examples_1.ipynb` | Examples with plots |

## Basic Usage

```python
result = did_multiplegt_stat(
    df=df,
    Y="outcome",
    ID="unit_id", 
    Time="year",
    D="treatment",
    estimator=["aoss", "waoss"],
    estimation_method="dr",
    order=2,
    noextrapolation=True,
    placebo=True
)

summary_did_multiplegt_stat(result)


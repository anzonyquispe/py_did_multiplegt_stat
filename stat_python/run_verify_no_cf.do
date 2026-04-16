/**********************************************************************
  Verify applied examples WITHOUT cross_fitting

  Purpose: Generate Stata reference for examples that had no
  previous reference without cross_fitting:
    - V.01: Reduced form (lngca) — as_vs_was, no placebo
    - V.02: First stage (lngpinc) — as_vs_was (AS has no ref)
    - V.04: on_placebo_sample reduced form
    - V.05: on_placebo_sample first stage
**********************************************************************/

cap log close _all
log using "verify_no_cf.log", text replace

cap prog drop did_multiplegt_stat
cap prog drop did_multiplegt_stat2
cap prog drop did_multiplegt_stat_pairwise
qui do "..\stataADO\did_multiplegt_stat.ado"

use "gazoline_did_multiplegt_stat.dta", clear

// V.01 — Reduced form: as_vs_was, no placebo, no cross_fitting
di "================================================================"
di "--- V.01: Reduced-form (no cross_fitting) ---"
di "================================================================"
did_multiplegt_stat lngca id year tau, or(1) as_vs_was controls(lngpinc)

// V.02 — First stage: as_vs_was, no cross_fitting
di "================================================================"
di "--- V.02: First-stage (no cross_fitting) ---"
di "================================================================"
did_multiplegt_stat lngpinc id year tau, or(1) as_vs_was controls(lngpinc)

// V.04 — on_placebo_sample reduced form: no cross_fitting
di "================================================================"
di "--- V.04: on_placebo_sample, reduced-form (no cross_fitting) ---"
di "================================================================"
did_multiplegt_stat lngca id year tau, or(1) controls(lngpinc) estimator(was) on_placebo_sample

// V.05 — on_placebo_sample first stage: no cross_fitting
di "================================================================"
di "--- V.05: on_placebo_sample, first-stage (no cross_fitting) ---"
di "================================================================"
did_multiplegt_stat lngpinc id year tau, or(1) controls(lngpinc) estimator(was) on_placebo_sample

log close

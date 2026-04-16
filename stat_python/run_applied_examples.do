/**********************************************************************
  APPLIED PAPER EXAMPLES — did_multiplegt_stat
  Reference: Companion paper, Sections 3.1 and 3.2

  These examples run WITHOUT cross_fitting so that Python and Stata
  produce identical results (cross_fitting uses different RNG draws).
**********************************************************************/

cap log close _all
log using "applied_examples_stata.log", text replace

cap prog drop did_multiplegt_stat
qui do "did_multiplegt_stat.ado"

/***********************************************************************
  APPLIED PAPER 1 — Li et al. (2014): Gasoline Taxes and Consumer Behavior
  Reference: Companion paper, Section 3.1
***********************************************************************/
di ""
di "================================================================"
di "  APPLIED PAPER 1 — Li et al. (2014): Gasoline Taxes"
di "  Reference: Companion paper, Section 3.1"
di "================================================================"

use "gazoline_did_multiplegt_stat.dta", clear

// V.01 — Reduced-form
di "--- V.01: Reduced-form ---"
eststo clear
eststo ReducedForm: did_multiplegt_stat lngca id year tau, or(1) as_vs_was controls(lngpinc)

// V.02 — First-stage
di "--- V.02: First-stage ---"
eststo FirstStage: did_multiplegt_stat lngpinc id year tau, or(1) as_vs_was controls(lngpinc)

// V.03 — IV-WAS
di "--- V.03: IV-WAS, bootstrap(5), seed(1) ---"
eststo IV: did_multiplegt_stat lngca id year lngpinc tau, or(1) controls(lngpinc) bootstrap(5) seed(1)

// V.04 — on_placebo_sample reduced-form
di "--- V.04: on_placebo_sample, reduced-form ---"
did_multiplegt_stat lngca id year tau, or(1) controls(lngpinc) estimator(was) on_placebo_sample

// V.05 — on_placebo_sample first-stage
di "--- V.05: on_placebo_sample, first-stage ---"
did_multiplegt_stat lngpinc id year tau, or(1) controls(lngpinc) estimator(was) on_placebo_sample


/***********************************************************************
  APPLIED PAPER 2 — Gentzkow et al. (2011): Newspapers and Electoral Politics
  Reference: Companion paper, Section 3.2
***********************************************************************/
di ""
di "================================================================"
di "  APPLIED PAPER 2 — Gentzkow et al. (2011): Newspapers and Turnout"
di "  Reference: Companion paper, Section 3.2"
di "================================================================"

use "gentzkowetal_didtextbook.dta", clear

// VI.01 — exact_match with placebo
di "--- VI.01: exact_match, placebo(1) ---"
did_multiplegt_stat prestout cnty90 year numdailies, placebo(1) exact_match

log close

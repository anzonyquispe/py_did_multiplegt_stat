cd "C:\Users\Usuario\Documents\GitHub\py_did_multiplegt_stat\stat_python"

log using "logfile_2026-02-27.log", append

cap prog drop did_multiplegt_stat
qui do "did_multiplegt_stat.ado"

/***********************************************************************
  VIGNETTE — ILLUSTRATING ALL OPTIONS
  Reference: Companion paper, Section 2
***********************************************************************/
di ""
di "================================================================"
di "  VIGNETTE — ILLUSTRATING ALL OPTIONS (Gazoline Data)"
di "  Reference: Companion paper, Section 2 — Li et al. (2014)"
di "================================================================"

use "gazoline_did_multiplegt_stat.dta", clear

// I.01 — order(1,2,3,4)
di "--- I.01: order(1,2,3,4) ---"
did_multiplegt_stat lngca id year tau, or(1 2 3 4)

// I.02 — IV with order(1,2,3,4,5,6,7,8)
di "--- I.02: IV with order(1,2,3,4,5,6,7,8) ---"
did_multiplegt_stat lngca id year lngpinc tau, or(1 2 3 4 5 6 7 8)

// I.03 — weights(pop), estimator(as), placebo(2), disaggregate
di "--- I.03: weights(pop), estimator(as), placebo(2), disaggregate ---"
did_multiplegt_stat lngca id year tau, or(1) weights(pop) estimator(as) placebo(2) disag

// I.04 — order=1, controls(lngpinc), estimator(as was), placebo(3), as_vs_was
di "--- I.04: order=1, controls, estimator(as was), placebo(3), as_vs_was ---"
did_multiplegt_stat lngca id year tau, or(1) controls(lngpinc) estimator(as was) placebo(3) as_vs_was

// I.05 — order=2, controls(lngpinc), estimator(as was), placebo(3), as_vs_was
di "--- I.05: order=2, controls, estimator(as was), placebo(3), as_vs_was ---"
did_multiplegt_stat lngca id year tau, or(2) controls(lngpinc) estimator(as was) placebo(3) as_vs_was

// I.06 — First-stage (Y=lngpinc), order=1
di "--- I.06: First-stage (Y=lngpinc), order=1 ---"
did_multiplegt_stat lngpinc id year tau, or(1) controls(lngpinc) estimator(as was) placebo(3) as_vs_was

// I.07 — First-stage (Y=lngpinc), order=2
di "--- I.07: First-stage (Y=lngpinc), order=2 ---"
did_multiplegt_stat lngpinc id year tau, or(2) controls(lngpinc) estimator(as was) placebo(3) as_vs_was

// I.08 — IV-WAS, order=1, controls
di "--- I.08: IV-WAS, order=1, controls ---"
did_multiplegt_stat lngca id year lngpinc tau, controls(lngpinc) or(1) estimator(iv-was)

// I.09 — IV-WAS, order=2, controls
di "--- I.09: IV-WAS, order=2, controls ---"
did_multiplegt_stat lngca id year lngpinc tau, controls(lngpinc) or(2) estimator(iv-was)

// I.10 — IV-WAS, bootstrap(2)
di "--- I.10: IV-WAS, bootstrap(2) ---"
did_multiplegt_stat lngca id year lngpinc tau, or(1) estimator(iv-was) bootstrap(2)

// I.11 — IV-WAS, bootstrap(2), seed(1)
di "--- I.11: IV-WAS, bootstrap(2), seed(1) ---"
did_multiplegt_stat lngca id year lngpinc tau, or(1) estimator(iv-was) bootstrap(2) seed(1)

// I.12 — IV-WAS, bootstrap(2), seed(1) reproducibility
di "--- I.12: IV-WAS, bootstrap(2), seed(1) reproducibility ---"
did_multiplegt_stat lngca id year lngpinc tau, or(1) estimator(iv-was) bootstrap(2) seed(1)

// I.13 — weights(lngpinc), estimator(as was), placebo(2)
di "--- I.13: weights(lngpinc), estimator(as was), placebo(2) ---"
did_multiplegt_stat lngca id year tau, or(1) weights(lngpinc) estimator(as was) placebo(2)

// I.14 — weights(pop), exact_match, estimator(as was), placebo(2)
di "--- I.14: weights(pop), exact_match, estimator(as was), placebo(2) ---"
did_multiplegt_stat lngca id year tau, or(1) weights(pop) estimator(as was) placebo(2) exact_match

// I.15 — estimator(was), placebo(2)
di "--- I.15: estimator(was), placebo(2) ---"
did_multiplegt (old) lngca id year tau
did_multiplegt_stat lngca id year tau, or(1) estimator(was) placebo(2)

// I.16 — controls(lngpinc lncars), order=1, placebo(3), as_vs_was
di "--- I.16: controls(lngpinc lncars), order=1, placebo(3), as_vs_was ---"
did_multiplegt_stat lngca id year tau, or(1) controls(lngpinc lncars) estimator(as was) placebo(3) as_vs_was

// I.17 — controls(lngpinc lncars), order=2, placebo(3), as_vs_was
di "--- I.17: controls(lngpinc lncars), order=2, placebo(3), as_vs_was ---"
did_multiplegt_stat lngca id year tau, or(2) controls(lngpinc lncars) estimator(as was) placebo(3) as_vs_was

// I.18 — switchers(up)
di "--- I.18: switchers(up) ---"
did_multiplegt_stat lngca id year tau, or(1) estimator(was) switchers(up)

// I.19 — switchers(down)
di "--- I.19: switchers(down) ---"
did_multiplegt_stat lngca id year tau, or(1) estimator(was) switchers(down)

// I.20 — bysort Democrat_Gov1966
di "--- I.20: bysort Democrat_Gov1966 ---"
cap drop Democrat_Gov1966 Democrat_Gov1966_temp
gen Democrat_Gov1966_temp = Democrat_Gov if year == 1966
drop Democrat_Gov1966
bys state: egen Democrat_Gov1966 = sum(Democrat_Gov1966_temp)
bysort Democrat_Gov1966: did_multiplegt_stat lngca id year tau, or(1) estimator(was) graph_off
did_multiplegt_stat lngca id year tau, or(1) estimator(was)
bysort Democrat_Gov1966: did_multiplegt_stat lngca id year tau, or(1) estimator(was)

// I.21 — noextrapolation, switchers(up)
di "--- I.21: noextrapolation, switchers(up) ---"
did_multiplegt_stat lngca id year tau, or(1) estimator(was) switchers(up) noextra

// I.22 — switchers(up), without noextrapolation
di "--- I.22: switchers(up), without noextrapolation ---"
did_multiplegt_stat lngca id year tau, or(1) estimator(was) switchers(up)

// I.23 — cluster(cluster_id_letter), controls, placebo(1)
di "--- I.23: cluster(cluster_id_letter), controls, placebo(1) ---"
cap drop cluster_id_letter
gen cluster_id_letter = substr(state, 1, 1)
did_multiplegt_stat lngca id year tau, or(1) controls(lngpinc) estimator(was) placebo(1) cluster(cluster_id_letter)

// I.24 — controls, placebo(1), without cluster
di "--- I.24: controls, placebo(1), without cluster ---"
did_multiplegt_stat lngca id year tau, or(1) controls(lngpinc) estimator(was) placebo(1)

// I.25 — twfe(percentile same_sample), bootstrap(5), placebo(1), as_vs_was
di "--- I.25: twfe(percentile same_sample), bootstrap(5), placebo(1) ---"
did_multiplegt_stat lngca id year tau, or(1) estimator(was) bootstrap(5) placebo(1) twfe(percentile same_sample) as_vs_was

// I.26 — twfe(same_sample), bootstrap(5), placebo(1)
di "--- I.26: twfe(same_sample), bootstrap(5), placebo(1) ---"
did_multiplegt_stat lngca id year tau, or(1) estimator(was) bootstrap(5) placebo(1) twfe(same_sample) as_vs_was

// I.27 — IV-WAS, twfe(percentile same_sample), bootstrap(5), placebo(1)
di "--- I.27: IV-WAS, twfe(percentile same_sample), bootstrap(5) ---"
did_multiplegt_stat lngca id year lngpinc tau, or(1) estimator(iv-was) bootstrap(5) placebo(1) twfe(percentile same_sample)

// I.28 — cross_validation
di "--- I.28: cross_validation(loocv, tol=0.01, max_k=3, seed=1, kfolds=2) ---"
did_multiplegt_stat lngca id year tau, estimator(as was) placebo(1) cross_validation(algo(loocv) tole(0.01) max_k(3) seed(1) kfolds(2))

// I.29 — trimming(10) + cross_fitting(2)
di "--- I.29: trimming(10) + cross_fitting(2) ---"
did_multiplegt_stat lngca id year tau, or(1) controls(lngpinc) estimator(as was) placebo(3) as_vs_was trimming(10) cross_fitting(2)

// I.30 — IV-WAS, trimming(10) + cross_fitting(2)
di "--- I.30: IV-WAS, trimming(10) + cross_fitting(2) ---"
did_multiplegt_stat lngca id year lngpinc tau, controls(lngpinc) or(1) estimator(iv-was) trimming(10) cross_fitting(2)


di ""
di "================================================================"
di "  VIGNETTE — ESTOUT: EXPORT TO LATEX"
di "  Reference: Companion paper, Section 4 — Li et al. (2014)"
di "================================================================"

// II.01 — Linear reduced-form
di "--- II.01: Linear reduced-form, order=1 ---"
eststo clear
eststo linearReducedForm: did_multiplegt_stat lngca id year tau, or(1) as_vs_was controls(lngpinc) cross_fitting(10)

// II.02 — Quadratic reduced-form
di "--- II.02: Quadratic reduced-form, order=2 ---"
eststo quadReducedForm: did_multiplegt_stat lngca id year tau, or(2) as_vs_was controls(lngpinc) cross_fitting(10)


di ""
di "================================================================"
di "  VIGNETTE — BY QUANTILE AND FD"
di "  Reference: Companion paper, Section 2 — Li et al. (2014)"
di "================================================================"

// III.01 — by_fd(2)
di "--- III.01: by_fd(2), estimator(as was), placebo(1), as_vs_was ---"
did_multiplegt_stat lngca id year tau, or(1) estimator(as was) placebo(1) as_vs_was by_fd(2)

// III.02 — IV-WAS, by_fd(2)
di "--- III.02: IV-WAS, by_fd(2), controls, placebo(1) ---"
did_multiplegt_stat lngca id year lngpinc tau, or(1) estimator(iv-was) controls(lngpinc) placebo(1) by_fd(2)

// III.03 — by_baseline(5)
di "--- III.03: by_baseline(5), estimator(was), placebo(1) ---"
did_multiplegt_stat lngca id year tau, or(1) estimator(was) placebo(1) by_baseline(5)

// III.04 — by_baseline(3)
di "--- III.04: by_baseline(3), estimator(as was), placebo(1) ---"
eststo clear
glob nb_quantiles = 3
eststo model_quant: did_multiplegt_stat lngca id year tau, or(1) estimator(as was) placebo(1) by_baseline($nb_quantiles)


di ""
di "================================================================"
di "  VIGNETTE — WAGEPAN DATA: EXACT MATCH"
di "  Reference: Companion paper, Section 2 — Wagepan (Vella & Verbeek, 1998)"
di "================================================================"

bcuse wagepan, clear

// IV.01 — binary treatment
di "--- IV.01: exact_match, estimator(was), placebo(1) ---"
did_multiplegt_stat lwage nr year union, estimator(was) exact_match placebo(1)

// IV.02 — discrete treatment
di "--- IV.02: discrete treatment, exact_match, estimator(was as), placebo(1) ---"
set seed 12345
gen dis_u = runiformint(1, 5) + union
gen lwage2 = lwage + 3*dis_u
did_multiplegt_stat lwage2 nr year dis_u, estimator(was as) exact_match placebo(1)

// IV.03 — IV-WAS exact_match
di "--- IV.03: IV-WAS, exact_match, placebo(1) ---"
did_multiplegt_stat lwage2 nr year dis_u union, estimator(iv-was) exact_match placebo(1)

// IV.04 — other_treatments
di "--- IV.04: other_treatments, exact_match, placebo(1) ---"
gen othertreat  = (lwage > 1.5)
gen othertreat2 = (lwage < 1) | (lwage > 2)
did_multiplegt_stat lwage nr year union, estimator(was) placebo(1) other_treatments(othertreat othertreat2) exact_match


/***********************************************************************
  APPLIED PAPER 1 — Li et al. (2014): Gasoline Taxes and Consumer Behavior
  Reference: Companion paper, Section 3.1
***********************************************************************/
di ""
di "================================================================"
di "  APPLIED PAPER 1 — Li et al. (2014): Gasoline Taxes"
di "================================================================"

use "gazoline_did_multiplegt_stat.dta", clear

// V.01 — Reduced-form
di "--- V.01: Reduced-form, cross_fitting(10) ---"
eststo clear
eststo ReducedForm: did_multiplegt_stat lngca id year tau, or(1) as_vs_was controls(lngpinc) cross_fitting(10)

// V.02 — First-stage
di "--- V.02: First-stage, cross_fitting(10) ---"
eststo FirstStage: did_multiplegt_stat lngpinc id year tau, or(1) as_vs_was controls(lngpinc) cross_fitting(10)

// V.03 — IV-WAS
di "--- V.03: IV-WAS, cross_fitting(10), bootstrap(10), seed(1) ---"
eststo IV: did_multiplegt_stat lngca id year lngpinc tau, or(1) controls(lngpinc) cross_fitting(10) bootstrap(10) seed(1)

// V.04 — on_placebo_sample reduced-form
di "--- V.04: on_placebo_sample, reduced-form ---"
did_multiplegt_stat lngca id year tau, or(1) controls(lngpinc) estimator(was) cross_fitting(10) on_placebo_sample

// V.05 — on_placebo_sample first-stage
di "--- V.05: on_placebo_sample, first-stage ---"
did_multiplegt_stat lngpinc id year tau, or(1) controls(lngpinc) estimator(was) cross_fitting(10) on_placebo_sample


/***********************************************************************
  APPLIED PAPER 2 — Gentzkow et al. (2011): Newspaper Entry and Electoral Politics
  Reference: Companion paper, Section 3.2
***********************************************************************/
di ""
di "================================================================"
di "  APPLIED PAPER 2 — Gentzkow et al. (2011): Newspapers and Turnout"
di "================================================================"

use "gentzkowetal_didtextbook.dta", clear

// VI.01 — exact_match + placebo(1)
di "--- VI.01: exact_match + placebo(1) ---"
did_multiplegt_stat prestout cnty90 year numdailies, placebo(1) exact_match

// VI.02 — exact_match + placebo(1) + by_baseline(2)
di "--- VI.02: exact_match + placebo(1) + by_baseline(2) ---"
did_multiplegt_stat prestout cnty90 year numdailies, placebo(1) exact_match by_baseline(2)


log close

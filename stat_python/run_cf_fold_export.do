/**********************************************************************
  Cross-fitting fold export test

  Purpose: Run V.01 with cross_fitting(10) AND export_cf_folds()
  so that Python can use the same fold assignments.
  Then Python should produce identical results.
**********************************************************************/

cap log close _all
log using "cf_fold_export.log", text replace

cap prog drop did_multiplegt_stat
cap prog drop did_multiplegt_stat2
cap prog drop did_multiplegt_stat_pairwise
qui do "..\stataADO\did_multiplegt_stat.ado"

use "gazoline_did_multiplegt_stat.dta", clear

// V.01 with cross_fitting(10) + fold export
di "================================================================"
di "--- V.01: cross_fitting(10) + export_cf_folds ---"
di "================================================================"
did_multiplegt_stat lngca id year tau, or(1) as_vs_was controls(lngpinc) cross_fitting(10) export_cf_folds(cf_folds.csv)

di ""
di "Fold IDs exported to cf_folds.csv"
di "Now run Python with cf_folds_file='cf_folds.csv' to compare."

log close

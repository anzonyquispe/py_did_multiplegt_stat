/***********************************************************************
	  I.  GAZOLINE DATA TRYING DIFFERENT OPTIONS
***********************************************************************/
cd "/Users/anzony.quisperojas/Documents/GitHub/py_did_multiplegt_stat/stat_python"

local today : display %tdCY-N-D date(c(current_date), "DMY")
log using "logfile_2026-02-27.log", append

cap prog drop did_multiplegt_stat
qui do "did_multiplegt_stat.ado"

use "gazoline_did_multiplegt_stat.dta", clear

//order option with multiple inputs
did_multiplegt_stat lngca id year tau, or(1 2 3 4) 

did_multiplegt_stat lngca id year lngpinc tau, or(1 2 3 4 5 6 7 8) 
did_multiplegt_stat lngca id year tau, or(1)  weights(pop) estimator(as )   placebo(2) disag

		//ssc install did_multiplegt_stat
        //net get did_multiplegt_stat
        //use gazoline_did_multiplegt_stat.dta, clear

		
//Reproducing first-stage and reduced-form from paper
did_multiplegt_stat lngca id year tau, or(1)  controls(lngpinc) estimator(as was)  placebo(3)  as_vs_was
did_multiplegt_stat lngca id year tau, or(2)  controls(lngpinc) estimator(as was)  placebo(3)  as_vs_was
did_multiplegt_stat lngpinc id year tau, or(1)  controls(lngpinc) estimator(as was)  placebo(3)  as_vs_was
did_multiplegt_stat lngpinc id year tau, or(2)  controls(lngpinc) estimator(as was)  placebo(3)  as_vs_was

// Reproducing iv
did_multiplegt_stat lngca id year lngpinc tau, controls(lngpinc) or(1) estimator(iv-was)   
did_multiplegt_stat lngca id year lngpinc tau, controls(lngpinc) or(2) estimator(iv-was)  

//Bootstrap + iv-was
did_multiplegt_stat lngca id year lngpinc tau, or(1)  estimator(iv-was)   bootstrap(2)
did_multiplegt_stat lngca id year lngpinc tau, or(1)  estimator(iv-was)   bootstrap(2) seed(1)
did_multiplegt_stat lngca id year lngpinc tau, or(1)  estimator(iv-was)   bootstrap(2) seed(1)

// weightss
did_multiplegt_stat lngca id year tau, or(1)  weights(lngpinc) estimator(as was)   placebo(2)
did_multiplegt_stat lngca id year tau, or(1)  weights(pop) estimator(as was)   placebo(2) exact_match
did_multiplegt (old) lngca id year tau
did_multiplegt_stat lngca id year tau, or(1) estimator(was)   placebo(2)

// more than one control variable
did_multiplegt_stat lngca id year tau, or(1)  controls(lngpinc lncars) estimator(as was)  placebo(3)  as_vs_was
did_multiplegt_stat lngca id year tau, or(2)  controls(lngpinc lncars) estimator(as was)  placebo(3)  as_vs_was

// up/down switchers
did_multiplegt_stat lngca id year tau, or(1) estimator(was) switchers(up)
did_multiplegt_stat lngca id year tau, or(1) estimator(was) switchers(down)

// bysort:
cap drop Democrat_Gov1966 Democrat_Gov1966_temp
gen Democrat_Gov1966_temp=Democrat_Gov if year==1966
drop Democrat_Gov1966 
bys state: egen Democrat_Gov1966=sum(Democrat_Gov1966_temp)
bysort Democrat_Gov1966: did_multiplegt_stat lngca id year tau, or(1) estimator(was) graph_off
did_multiplegt_stat lngca id year tau, or(1) estimator(was)
bysort Democrat_Gov1966: did_multiplegt_stat lngca id year tau, or(1) estimator(was)

//No extrapolation
did_multiplegt_stat lngca id year tau, or(1) estimator(was) switchers(up) noextra
did_multiplegt_stat lngca id year tau, or(1) estimator(was) switchers(up) 

//cluster 
cap drop cluster_id_letter
gen cluster_id_letter=substr(state,1,1) 
did_multiplegt_stat lngca id year tau, or(1) controls(lngpinc) estimator(was)   placebo(1) cluster(cluster_id_letter)
did_multiplegt_stat lngca id year tau, or(1) controls(lngpinc) estimator(was)   placebo(1)


//twfe vs was + percentile bootstrap + same_sample, and was vs as
did_multiplegt_stat lngca id year tau, or(1)  estimator(was)    bootstrap(5) placebo(1) twfe(percentile same_sample) as_vs_was

//twfe vs was + normal bootstrap + same_sample, and was vs as
did_multiplegt_stat lngca id year tau, or(1)  estimator(was)    bootstrap(5) placebo(1) twfe(same_sample) as_vs_was

//2sls-twfe vs iv-was
did_multiplegt_stat lngca id year lngpinc tau , or(1)  estimator(iv-was)    bootstrap(5) placebo(1) twfe(percentile same_sample)

//Cross_validation
did_multiplegt_stat lngca id year tau, estimator(as was) placebo(1)    cross_validation(algo(loocv) tole(0.01) max_k(3) seed(1) kfolds(2)) 

// //trimming without croiss-fitting
//
// did_multiplegt_stat lngca id year tau, or(1)  controls(lngpinc) estimator(as was)  placebo(3)  as_vs_was trimming_up(80)
// did_multiplegt_stat lngca id year tau, or(1)  controls(lngpinc) estimator(as was)  placebo(3)  as_vs_was trimming_down(30)
// did_multiplegt_stat lngca id year tau, or(1)  controls(lngpinc) estimator(as was)  placebo(3)  as_vs_was trimming_down(30)
//
// did_multiplegt_stat lngpinc id year tau, or(1)  controls(lngpinc) estimator(as was)  placebo(3)  as_vs_was trimming_up(80)
// did_multiplegt_stat lngca id year lngpinc tau, controls(lngpinc) or(1) estimator(iv-was)  trimming_up(80)
//
// did_multiplegt_stat lngpinc id year tau, or(1)  controls(lngpinc) estimator(as was)  placebo(3)  as_vs_was trimming_up(105)
// did_multiplegt_stat lngpinc id year tau, or(1)  controls(lngpinc) estimator(as was)  placebo(3)  as_vs_was trimming_down(30) trimming_up(20)

// //trimming  + cross_fitting 
// did_multiplegt_stat lngca id year tau, or(1)  controls(lngpinc) estimator(as was)  placebo(3)  as_vs_was trimming(10) cross_fitting(2)
// did_multiplegt_stat lngca id year lngpinc tau, controls(lngpinc) or(1) estimator(iv-was) trimming(10) cross_fitting(2)


/******************************************************************************
	   III. By Quantile and FD
******************************************************************************/

//1. The command

did_multiplegt_stat lngca id year tau, or(1)  estimator(as was)  placebo(1) as_vs_was by_fd(2)

did_multiplegt_stat lngca id year lngpinc tau, or(1)  estimator(iv-was) controls(lngpinc)  placebo(1) by_fd(2)

did_multiplegt_stat lngca id year tau, or(1)  estimator(was) placebo(1) by_baseline(5)

//2. Using estout even when submodels are estimated with  By Quantile and FD
eststo clear

glob nb_quantiles = 3
eststo model_quant: did_multiplegt_stat lngca id year tau, or(1)  estimator(as was)   placebo(1) by_baseline($nb_quantiles)

/******************************************************************************
       IV. Trimming + Cross-fitting (Gazoline)
******************************************************************************/

// trimming + cross_fitting
did_multiplegt_stat lngca id year tau, or(1) controls(lngpinc) estimator(as was) placebo(3) as_vs_was trimming(10) cross_fitting(2)
did_multiplegt_stat lngca id year lngpinc tau, controls(lngpinc) or(1) estimator(iv-was) trimming(10) cross_fitting(2)


/******************************************************************************
       V. Wagepan Data — Exact Match
******************************************************************************/

bcuse wagepan, clear

// binary treatment
did_multiplegt_stat lwage nr year union, estimator(was) exact_match placebo(1)

// discrete treatment
set seed 12345
gen dis_u = runiformint(1, 5) + union
gen lwage2 = lwage + 3*dis_u
did_multiplegt_stat lwage2 nr year dis_u, estimator(was as) exact_match placebo(1)

// IV-WAS exact_match
did_multiplegt_stat lwage2 nr year dis_u union, estimator(iv-was) exact_match placebo(1)

// other_treatments
gen othertreat  = (lwage > 1.5)
gen othertreat2 = (lwage < 1) | (lwage > 2)
did_multiplegt_stat lwage nr year union, estimator(was) placebo(1) other_treatments(othertreat othertreat2) exact_match


/******************************************************************************
       VI. Companion Paper — Gazoline with cross_fitting
******************************************************************************/

use "gazoline_did_multiplegt_stat.dta", clear

// Reduced-form
did_multiplegt_stat lngca id year tau, or(1) as_vs_was controls(lngpinc) cross_fitting(10)

// First-stage
did_multiplegt_stat lngpinc id year tau, or(1) as_vs_was controls(lngpinc) cross_fitting(10)

// IV-WAS
did_multiplegt_stat lngca id year lngpinc tau, or(1) controls(lngpinc) cross_fitting(10) bootstrap(10) seed(1)


// Robust to dynamic effects up to one lag
did_multiplegt_stat lngca id year tau, or(1) controls(lngpinc) estimator(was) cross_fitting(10) on_placebo_sample
did_multiplegt_stat lngpinc id year tau, or(1) controls(lngpinc) estimator(was) cross_fitting(10) on_placebo_sample


/******************************************************************************
       VII. Gentzkow et al. (2011)
******************************************************************************/

use "gentzkowetal_didtextbook.dta", clear

// Main
did_multiplegt_stat prestout cnty90 year numdailies, placebo(1) exact_match

// By baseline
did_multiplegt_stat prestout cnty90 year numdailies, placebo(1) exact_match by_baseline(2)


log close

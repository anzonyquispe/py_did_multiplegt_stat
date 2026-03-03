/***********************************************************************
	  I.  GAZOLINE DATA TRYING DIFFERENT OPTIONS
***********************************************************************/
cd "/Users/anzony.quisperojas/Documents/GitHub/py_did_multiplegt_stat/stat_python"

local today : display %tdCY-N-D date(c(current_date), "DMY")
log using "logfile_`today'.log", replace

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


log close


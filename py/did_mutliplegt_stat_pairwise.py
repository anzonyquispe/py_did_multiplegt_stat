"""
did_multiplegt_stat_pairwise.py

A faithful, **functional** Python translation (pandas + statsmodels) of the R internal
function `did_multiplegt_stat_pairwise()` you pasted.

Goal:
- Keep the **same logic/flow** as the R code (pairwise DiD between consecutive periods).
- Use the same variable names with `_XX` suffix to minimize mental diffs with the R version.
- Provide **very detailed English comments** explaining what each block does.

Dependencies:
    pip install pandas numpy statsmodels patsy

IMPORTANT ASSUMPTIONS (same as the R code path you pasted):
- `df` already contains standardized columns created upstream (as in the R "main"):
    ID_XX, T_XX, Y_XX, D_XX, (optional) Z_XX,
    tsfilled_XX (0/1 flag from panel "fill"),
    weight_XX (observation weights),
    and if clustering is used: cluster_XX and weight_c_XX (cluster weights).
- This function is called for a given `pairwise` time index p, and it uses T_XX in {p-1,p}
  (or {p-2,p-1,p} when placebo=True), then internally relabels time to 1..2 (or 1..3).
- The arguments Y, ID, Time, D, Z, estimator are kept only for interface compatibility;
  the function uses the standardized _XX columns.

Return format:
    {
      "scalars": updated_scalars_dict,
      "to_add":  df_with_ID_and_influence_functions_and_aux_columns  (or None)
    }

This is intended to be called from a "main" function that aggregates across p.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf


# ============================================================
# Helper utilities (ported from utils.R semantics)
# ============================================================

def _get_weight_col(df: pd.DataFrame, w: Optional[str]) -> Optional[str]:
    """Return weight column name if it exists, else None."""
    if w is None:
        w = "weight_XX"
    return w if w in df.columns else None


def wSum(df: pd.DataFrame, w: Optional[str] = None) -> float:
    """
    Weighted sum of weights (NOT sum of a variable).
    In the R code, `wSum(df)` is often used as 'effective sample size denominator'
    for sd/sqrt(wSum).
    """
    wcol = _get_weight_col(df, w)
    if wcol is None:
        return float(len(df))
    return float(np.nansum(df[wcol].to_numpy(dtype=float)))


def Mean(var: str, df: pd.DataFrame, w: Optional[str] = None) -> float:
    """
    Weighted mean of df[var] using df[w].
    Mirrors utils.R::Mean usage in your code.
    """
    if var not in df.columns:
        return float("nan")
    wcol = _get_weight_col(df, w)
    x = df[var].to_numpy(dtype=float)

    mask = ~np.isnan(x)
    if wcol is None:
        return float(np.nanmean(x)) if mask.any() else float("nan")

    ww = df[wcol].to_numpy(dtype=float)
    mask = mask & ~np.isnan(ww)
    if not mask.any():
        return float("nan")
    return float(np.average(x[mask], weights=ww[mask]))


def Sum(var: str, df: pd.DataFrame, w: Optional[str] = None) -> float:
    """
    Weighted sum of df[var] using df[w]:
        Sum = Σ_i w_i * x_i
    This matches how the R code uses `Sum()` for numerators/denominators.
    """
    if var not in df.columns:
        return 0.0
    wcol = _get_weight_col(df, w)
    x = df[var].to_numpy(dtype=float)
    if wcol is None:
        return float(np.nansum(x))
    ww = df[wcol].to_numpy(dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(ww)
    return float(np.sum(x[mask] * ww[mask]))


def Sd(var: str, df: pd.DataFrame, w: Optional[str] = None) -> float:
    """
    Weighted standard deviation (population version):
        sqrt( Σ w (x-mean_w)^2 / Σ w )
    The R code sometimes uses Sd(...) and sometimes uses base sd(...).
    We provide this for the Sd(...) cases.
    """
    if var not in df.columns:
        return float("nan")
    wcol = _get_weight_col(df, w)
    x = df[var].to_numpy(dtype=float)

    if wcol is None:
        return float(np.nanstd(x, ddof=1))  # close to R sd() default ddof=1

    ww = df[wcol].to_numpy(dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(ww)
    if mask.sum() <= 1:
        return float("nan")
    x = x[mask]
    ww = ww[mask]
    mu = np.average(x, weights=ww)
    varw = np.average((x - mu) ** 2, weights=ww)
    return float(np.sqrt(varw))


def stata_logit(formula: str, df: pd.DataFrame, wcol: str = "weight_XX",
                maxit: int = 200, tol: float = 1e-10):
    """
    Emulate the R `stata_logit()` wrapper:
    - Fit a (weighted) logit via GLM Binomial with frequency weights.
    """
    freq_w = df[wcol] if wcol in df.columns else None
    # statsmodels GLM uses `freq_weights` for frequency weights
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), freq_weights=freq_w)
    res = model.fit(maxiter=maxit, tol=tol, disp=0)
    return res


def lpredict(df: pd.DataFrame, outcol: str, fitted_model, prob: bool = False) -> pd.DataFrame:
    """
    Emulate the R `lpredict()` behavior for our simplified (fact_reg=False) path:
    - Adds a column `outcol` with model predictions for all rows in df.
    - For GLM Binomial, statsmodels returns predicted probabilities by default.
      For WLS/OLS, returns fitted values.
    """
    pred = fitted_model.predict(df)
    df[outcol] = np.asarray(pred, dtype=float)
    return df


# ============================================================
# Main translation: did_multiplegt_stat_pairwise
# ============================================================

def did_multiplegt_stat_pairwise(
    df: pd.DataFrame,
    Y: str,
    ID: str,
    Time: str,
    D: str,
    Z: Optional[str],
    estimator: str,
    order: int,
    noextrapolation: bool,
    weight: Optional[str],
    switchers: Optional[str],
    pairwise: int,
    IDs: Any,
    aoss: int,
    waoss: int,
    ivwaoss: int,
    estimation_method: str,
    scalars: Dict[str, Any],
    placebo: bool,
    exact_match: bool,
    cluster: Optional[str],
    by_fd_opt: Optional[int],
    other_treatments: Optional[List[str]],
) -> Dict[str, Any]:
    """
    Internal function for estimation of pairwise DiD between consecutive time periods.

    Parameters are kept to mirror the R interface; key behavior relies on df having *_XX columns.

    Returns:
        dict with keys:
            - "scalars": updated scalars dict
            - "to_add":  DataFrame with group-specific variables (IFs etc.) for aggregation
    """

    # ------------------------------------------------------------
    # 0) Preallocation / local copies
    # ------------------------------------------------------------
    df = df.copy()

    # These exist in the R function but are not operationally used later in the pasted code.
    IV_req_XX = None
    PS_0_XX = None

    # Placebo suffix used to name scalars/columns consistently with the R code.
    pl = "_pl" if placebo else ""

    # ------------------------------------------------------------
    # 1) Subset to the relevant time window: {p-1,p} or {p-2,p-1,p}
    # ------------------------------------------------------------
    if placebo:
        df = df[df["T_XX"].isin([pairwise - 2, pairwise - 1, pairwise])]
    else:
        df = df[df["T_XX"].isin([pairwise - 1, pairwise])]

    # If no rows survive, mirror the R early-exit convention downstream.
    # (R exits later after additional steps; here we keep flow but will handle nrow==0 below.)

    # ------------------------------------------------------------
    # 2) Detect "gap" using tsfilled_XX pattern (same computation as R)
    #    - For each time, compute min(tsfilled_XX).
    #    - gap_XX = max over time of these minima.
    #    Interpretation: a crude indicator that the panel chunk has fill/gaps.
    # ------------------------------------------------------------
    df["tsfilled_min_XX"] = df.groupby("T_XX")["tsfilled_XX"].transform(
        lambda s: np.nanmin(s.to_numpy(dtype=float))
    )
    gap_XX = float(np.nanmax(df["tsfilled_min_XX"].to_numpy(dtype=float))) if len(df) else 0.0

    # ------------------------------------------------------------
    # 3) Relabel time to consecutive ids: 1..2 (or 1..3 if placebo)
    #    Equivalent to:
    #       group_by(T_XX) %>% mutate(Tbis_XX = cur_group_id())
    # ------------------------------------------------------------
    # Keep unique times sorted, map to 1..k
    tvals = np.sort(df["T_XX"].dropna().unique())
    tmap = {t: i + 1 for i, t in enumerate(tvals)}
    df["T_XX"] = df["T_XX"].map(tmap).astype(float)

    # ------------------------------------------------------------
    # 4) Sort by (ID_XX, T_XX) then compute within-ID first differences:
    #       delta_Y = Y_t - Y_{t-1}
    #       delta_D = D_t - D_{t-1}
    #       delta_Z if IV
    #    In the R/plm path, `diff()` on pdata.frame does this within panel index.
    # ------------------------------------------------------------
    df["ID_XX"] = pd.to_numeric(df["ID_XX"], errors="coerce")
    df = df.sort_values(["ID_XX", "T_XX"], kind="mergesort").reset_index(drop=True)

    df["delta_Y_XX"] = df.groupby("ID_XX")["Y_XX"].diff()
    df["delta_D_XX"] = df.groupby("ID_XX")["D_XX"].diff()

    if ivwaoss == 1:
        if "Z_XX" not in df.columns:
            raise ValueError("ivwaoss==1 but df has no Z_XX column.")
        df["delta_Z_XX"] = df.groupby("ID_XX")["Z_XX"].diff()

    # Other treatments: compute within-ID diffs, then collapse to a per-ID constant:
    # `fd_ot_XX := sum(diff(ot), na.rm=TRUE)` (over the two/three rows).
    if other_treatments:
        for v in other_treatments:
            temp = df.groupby("ID_XX")[v].diff()
            df[f"fd_{v}_temp_XX"] = temp
        for v in other_treatments:
            df[f"fd_{v}_XX"] = df.groupby("ID_XX")[f"fd_{v}_temp_XX"].transform(
                lambda s: np.nansum(s.to_numpy(dtype=float))
            )
            df.drop(columns=[f"fd_{v}_temp_XX"], inplace=True)

    # If partition_XX exists (by_fd logic), store its lead then drop it.
    if "partition_XX" in df.columns:
        # lead within ID because df is sorted by (ID,T)
        df["partition_lead_XX"] = df.groupby("ID_XX")["partition_XX"].shift(-1)
        df.drop(columns=["partition_XX"], inplace=True)

    # ------------------------------------------------------------
    # 5) Construct delta_Y_XX as a per-ID constant:
    #    - placebo: use ONLY the "pre" delta (when T==2 after relabeling), then replicate.
    #    - non-placebo: average available delta_Y across the (two) rows and replicate.
    # ------------------------------------------------------------
    if placebo:
        df["delta_temp"] = np.where(df["T_XX"] == 2, df["delta_Y_XX"], np.nan)
        df["delta_Y_XX"] = df.groupby("ID_XX")["delta_temp"].transform(np.nanmean)
        df.drop(columns=["delta_temp"], inplace=True)
    else:
        df["delta_Y_XX"] = df.groupby("ID_XX")["delta_Y_XX"].transform(np.nanmean)

    # ------------------------------------------------------------
    # 6) Placebo sample restriction:
    #    - For AOSS/WAOSS placebo: keep units with D_{t-2}=D_{t-1} (delta_D==0 at T==2),
    #      drop T==1, and keep delta_D only for T==3 so later it becomes the (t-1,t) switch.
    #    - For IV placebo: same but using delta_Z==0 at T==2.
    # ------------------------------------------------------------
    if placebo and (aoss == 1 or waoss == 1):
        df["inSamplePlacebo_temp_XX"] = np.where(
            (df["delta_D_XX"] == 0) & (df["T_XX"] == 2),
            1.0,
            0.0,
        )
        df.loc[df["delta_D_XX"].isna(), "inSamplePlacebo_temp_XX"] = np.nan
        df["inSamplePlacebo_XX"] = df.groupby("ID_XX")["inSamplePlacebo_temp_XX"].transform(
            lambda s: np.nanmax(s.to_numpy(dtype=float))
        )
        df = df[df["T_XX"] != 1]  # drop earliest row
        df["delta_D_XX"] = np.where(df["T_XX"] != 3, np.nan, df["delta_D_XX"])

    if placebo and ivwaoss == 1:
        df["inSamplePlacebo_IV_temp_XX"] = np.where(
            (df["delta_Z_XX"] == 0) & (df["T_XX"] == 2),
            1.0,
            0.0,
        )
        df["inSamplePlacebo_XX"] = df.groupby("ID_XX")["inSamplePlacebo_IV_temp_XX"].transform(
            lambda s: np.nanmax(s.to_numpy(dtype=float))
        )
        df = df[df["T_XX"] != 1]
        df["delta_Z_XX"] = np.where(df["T_XX"] != 3, np.nan, df["delta_Z_XX"])

    # ------------------------------------------------------------
    # 7) Empty after placebo/filters => early exit with zeros/NA like R
    # ------------------------------------------------------------
    if len(df) == 0:
        # Set scalar outputs as in R "no obs" block
        if aoss == 1:
            scalars[f"P_{pairwise}{pl}_XX"] = 0.0
        if waoss == 1:
            scalars[f"E_abs_delta_D_{pairwise}{pl}_XX"] = 0.0
        if ivwaoss == 1:
            scalars[f"denom_delta_IV_{pairwise}{pl}_XX"] = 0.0

        scalars[f"non_missing_{pairwise}{pl}_XX"] = 0.0

        for v in ("Switchers", "Stayers"):
            for n in (1, 2, 3):
                scalars[f"N_{v}_{n}_{pairwise}{pl}_XX"] = 0.0

        # deltas/sd/LB/UB for whichever estimators are active
        estims = [aoss, waoss, ivwaoss]
        for i, active in enumerate(estims, start=1):
            if active == 1:
                scalars[f"delta_{i}_{pairwise}{pl}_XX"] = 0.0
                scalars[f"sd_delta_{i}_{pairwise}{pl}_XX"] = np.nan
                scalars[f"LB_{i}_{pairwise}{pl}_XX"] = np.nan
                scalars[f"UB_{i}_{pairwise}{pl}_XX"] = np.nan

        return {"scalars": scalars, "to_add": None}

    # ------------------------------------------------------------
    # 8) Make delta_D and (delta_Z) per-ID constants (same as delta_Y step)
    # ------------------------------------------------------------
    df["delta_D_XX"] = df.groupby("ID_XX")["delta_D_XX"].transform(np.nanmean)

    if ivwaoss == 1:
        df["delta_Z_XX"] = df.groupby("ID_XX")["delta_Z_XX"].transform(np.nanmean)

        # SI_XX is sign(delta_Z): +1 / 0 / -1
        df["SI_XX"] = np.sign(df["delta_Z_XX"]).astype(float)

        # Save baseline Z as Z1_XX and drop Z_XX later
        df["Z1_XX"] = df["Z_XX"]

    # ------------------------------------------------------------
    # 9) Define used_in indicators, switcher direction S_XX, and abs deltas
    # ------------------------------------------------------------
    df[f"used_in_{pairwise}_XX"] = (
        (~df["delta_Y_XX"].isna()) & (~df["delta_D_XX"].isna())
    ).astype(float)

    if ivwaoss == 1:
        df[f"used_in_IV_{pairwise}_XX"] = (
            (df[f"used_in_{pairwise}_XX"] == 1.0) & (~df["delta_Z_XX"].isna())
        ).astype(float)
        df = df[df[f"used_in_IV_{pairwise}_XX"] == 1.0]

    # S_XX is sign(delta_D): +1 up, -1 down, 0 stayer
    df["S_XX"] = np.sign(df["delta_D_XX"]).astype(float)

    # Filter switcher direction if user requested only "up" or only "down"
    if (waoss == 1 or aoss == 1):
        df["abs_delta_D_XX"] = df["S_XX"] * df["delta_D_XX"]  # equals |delta_D|
        if switchers is not None:
            # R logic: drop the opposite direction
            drop_val = (1.0 if switchers == "down" else 0.0) - (1.0 if switchers == "up" else 0.0)
            df = df[~(df["S_XX"] == drop_val)]

    if ivwaoss == 1:
        if switchers is not None:
            drop_val = (1.0 if switchers == "down" else 0.0) - (1.0 if switchers == "up" else 0.0)
            df = df[~(df["SI_XX"] == drop_val)]
        df["abs_delta_Z_XX"] = df["SI_XX"] * df["delta_Z_XX"]  # equals |delta_Z|

    # ------------------------------------------------------------
    # 10) Drop the "second" year line: keep only the first row of the pair
    #     (After relabeling, this is T!=max(T).)
    # ------------------------------------------------------------
    df = df[df["T_XX"] != df["T_XX"].max()]

    # Save baseline D as D1_XX and drop D_XX later
    df["D1_XX"] = df["D_XX"]
    df.drop(columns=["D_XX"], inplace=True)

    # Ht_XX indicates whether the pairwise first differences are available for this unit
    df["Ht_XX"] = ((~df["delta_D_XX"].isna()) & (~df["delta_Y_XX"].isna())).astype(float)

    # Set S_XX missing where Ht_XX==0
    df.loc[df["Ht_XX"] == 0, "S_XX"] = np.nan

    if ivwaoss == 1:
        df["Ht_XX"] = ((df["Ht_XX"] == 1.0) & (~df["delta_Z_XX"].isna())).astype(float)
        df.loc[df["Ht_XX"] == 0, "SI_XX"] = np.nan

    # By_fd bin restriction (keep partition_lead==0 or == selected bin)
    if by_fd_opt is not None and "partition_lead_XX" in df.columns:
        df = df[(df["partition_lead_XX"] == 0) | (df["partition_lead_XX"] == by_fd_opt)]

    # ------------------------------------------------------------
    # 11) Imbalanced-panel adjustments (placebo and other_treatments)
    #     - Set key variables to NA when placebo sample condition fails
    #     - Or when other_treatments changed (fd_ot != 0)
    # ------------------------------------------------------------
    vars_to_set_missing = ["S_XX", "delta_D_XX", "delta_Y_XX", "D1_XX"]
    if aoss == 1 or waoss == 1:
        vars_to_set_missing += ["abs_delta_D_XX"]
    else:
        vars_to_set_missing += ["Z1_XX", "SI_XX"]

    if placebo and "inSamplePlacebo_XX" in df.columns:
        mask_bad = (df["inSamplePlacebo_XX"] == 0)
        for v in vars_to_set_missing:
            if v in df.columns:
                df.loc[mask_bad, v] = np.nan
        df.loc[mask_bad, "Ht_XX"] = 0.0

    if other_treatments:
        for ot in other_treatments:
            colfd = f"fd_{ot}_XX"
            if colfd in df.columns:
                mask_bad = (df[colfd] != 0)
                for v in vars_to_set_missing:
                    if v in df.columns:
                        df.loc[mask_bad, v] = np.nan
                df.loc[mask_bad, "Ht_XX"] = 0.0

    # ------------------------------------------------------------
    # 12) No-extrapolation trimming:
    #     Drop units whose baseline D1 is outside stayers' support (or Z1 for IV).
    # ------------------------------------------------------------
    # Initialize totals if absent (R keeps them in scalars across calls)
    scalars.setdefault("N_drop_total_XX", 0.0)
    scalars.setdefault("N_drop_total_C_XX", 0.0)

    if noextrapolation:
        if aoss == 1 or waoss == 1:
            stayers = df[df["S_XX"] == 0]
            max_D1 = float(np.nanmax(stayers["D1_XX"])) if len(stayers) else float("nan")
            min_D1 = float(np.nanmin(stayers["D1_XX"])) if len(stayers) else float("nan")
            df["outofBounds_XX"] = (df["D1_XX"] < min_D1) | (df["D1_XX"] > max_D1)
            N_drop = float(np.nansum(df["outofBounds_XX"].astype(float)))
            scalars[f"N_drop_{pairwise}{pl}_XX"] = N_drop
            df = df[~df["outofBounds_XX"]]

            if (N_drop > 0) and (not placebo) and (gap_XX == 0) and (N_drop < len(df) - 1):
                scalars["N_drop_total_XX"] += N_drop

        if ivwaoss == 1:
            stayers = df[df["SI_XX"] == 0]
            max_Z1 = float(np.nanmax(stayers["Z1_XX"])) if len(stayers) else float("nan")
            min_Z1 = float(np.nanmin(stayers["Z1_XX"])) if len(stayers) else float("nan")
            df["outofBoundsIV_XX"] = (df["Z1_XX"] < min_Z1) | (df["Z1_XX"] > max_Z1)
            N_IVdrop = float(np.nansum(df["outofBoundsIV_XX"].astype(float)))
            scalars[f"N_IVdrop_{pairwise}{pl}_XX"] = N_IVdrop
            df = df[~df["outofBoundsIV_XX"]]

            if (N_IVdrop > 0) and (not placebo) and (gap_XX == 0) and (N_IVdrop < len(df) - 1):
                scalars["N_drop_total_XX"] += N_IVdrop

    # ------------------------------------------------------------
    # 13) Exact matching feasibility (drop switchers w/o stayers and stayers w/o switchers)
    # ------------------------------------------------------------
    if exact_match:
        if aoss == 1 or waoss == 1:
            group_cols = ["D1_XX"] + (other_treatments or [])
            g = df.groupby(group_cols, dropna=False)
            # For each baseline cell, do we have at least one stayer (abs_delta_D==0)?
            df["has_match_min_XX"] = g["abs_delta_D_XX"].transform(lambda s: np.nanmin(s[~s.isna()].to_numpy(dtype=float)))
            # For each baseline cell, do we have at least one switcher (abs_delta_D>0)?
            df["has_match_max_XX"] = g["abs_delta_D_XX"].transform(lambda s: np.nanmax(s[~s.isna()].to_numpy(dtype=float)))

            # s_has_match_XX only matters for switchers: needs a stayer present in the cell
            df["s_has_match_XX"] = np.where(~df["S_XX"].isna(), (df["has_match_min_XX"] == 0).astype(float), -1.0)
            df.loc[df["S_XX"] == 0, "s_has_match_XX"] = -1.0

            # c_has_match_XX only matters for stayers: needs a switcher present in the cell
            df["c_has_match_XX"] = np.where(~df["S_XX"].isna(), (df["has_match_max_XX"] > 0).astype(float), -1.0)
            df.loc[(df["S_XX"] != 0) & (~df["S_XX"].isna()), "c_has_match_XX"] = -1.0

        else:
            group_cols = ["Z1_XX"] + (other_treatments or [])
            g = df.groupby(group_cols, dropna=False)
            df["has_match_min_XX"] = g["abs_delta_Z_XX"].transform(lambda s: np.nanmin(s[~s.isna()].to_numpy(dtype=float)))
            df["has_match_max_XX"] = g["abs_delta_Z_XX"].transform(lambda s: np.nanmax(s[~s.isna()].to_numpy(dtype=float)))

            df["s_has_match_XX"] = np.where(~df["SI_XX"].isna(), (df["has_match_min_XX"] == 0).astype(float), -1.0)
            df.loc[df["SI_XX"] == 0, "s_has_match_XX"] = -1.0

            df["c_has_match_XX"] = np.where(~df["SI_XX"].isna(), (df["has_match_max_XX"] > 0).astype(float), -1.0)
            df.loc[(df["SI_XX"] != 0) & (~df["SI_XX"].isna()), "c_has_match_XX"] = -1.0

        N_drop_s = float((df["s_has_match_XX"] == 0).sum())
        N_drop_c = float((df["c_has_match_XX"] == 0).sum())
        scalars[f"N_drop_{pairwise}{pl}_XX"] = N_drop_s
        scalars[f"N_drop_{pairwise}{pl}_C_XX"] = N_drop_c

        if (N_drop_s > 0) and (N_drop_s != len(df)) and (gap_XX == 0):
            scalars["N_drop_total_XX"] += N_drop_s
        if (N_drop_c > 0) and (N_drop_c != len(df)) and (gap_XX == 0):
            scalars["N_drop_total_C_XX"] += N_drop_c

        mask_bad = (df["s_has_match_XX"] == 0) | (df["c_has_match_XX"] == 0)
        for v in vars_to_set_missing:
            if v in df.columns:
                df.loc[mask_bad, v] = np.nan
        df.loc[mask_bad, "Ht_XX"] = 0.0

        # Saturate polynomial order to number of unique baseline values (as in R)
        order = int(df["D1_XX"].nunique(dropna=True)) if "D1_XX" in df.columns else order

        # Cleanup
        df.drop(columns=[c for c in ["has_match_min_XX", "has_match_max_XX"] if c in df.columns], inplace=True)

    # ------------------------------------------------------------
    # 14) Basic bookkeeping scalars for this pair
    # ------------------------------------------------------------
    # Wpl_XX = sum of weights; Npl_XX = number of rows (after dropping second-year line)
    df.setdefault("weight_XX", 1.0)
    scalars[f"W{pl}_XX"] = float(np.nansum(df["weight_XX"].to_numpy(dtype=float))) if "weight_XX" in df.columns else float(len(df))
    scalars[f"N{pl}_XX"] = float(len(df))

    # Counts of switchers/stayers (used for feasibility checks and reporting)
    if waoss == 1 or aoss == 1:
        scalars[f"N_Switchers{pl}_XX"] = float(((df["S_XX"] != 0) & (~df["S_XX"].isna())).sum())
        scalars[f"N_Stayers{pl}_XX"] = float((df["S_XX"] == 0).sum())
    if ivwaoss == 1:
        scalars[f"N_Switchers_IV{pl}_XX"] = float(((df["SI_XX"] != 0) & (~df["SI_XX"].isna())).sum())
        scalars[f"N_Stayers_IV{pl}_XX"] = float((df["SI_XX"] == 0).sum())

    # ------------------------------------------------------------
    # 15) Build polynomial regressors and formula strings (reg_pol_XX, IV_reg_pol_XX)
    #     This is used for:
    #       - RA: E(deltaY|baseline, stayer)
    #       - PS: P(stayer|baseline), P(switcher +/-|baseline), etc.
    # ------------------------------------------------------------
    for pol_level in range(1, order + 1):
        df[f"D1_{pol_level}_XX"] = df["D1_XX"] ** pol_level

    reg_pol_terms = " + ".join([f"D1_{k}_XX" for k in range(1, order + 1)])

    if other_treatments:
        # Add saturated interaction block via '*' semantics (main effects + interactions)
        interact = "D1_1_XX"
        for v in other_treatments:
            interact = f"{interact} * {v}"
        reg_pol_terms = f"{reg_pol_terms} + {interact}"

    # IV-side polynomial in Z1
    if ivwaoss == 1:
        for pol_level in range(1, order + 1):
            df[f"Z1_{pol_level}_XX"] = df["Z1_XX"] ** pol_level
        IV_reg_pol_terms = " + ".join([f"Z1_{k}_XX" for k in range(1, order + 1)])
        if other_treatments:
            interact = "Z1_1_XX"
            for v in other_treatments:
                interact = f"{interact} * {v}"
            IV_reg_pol_terms = IV_reg_pol_terms.replace("Z1_1_XX", interact)
    else:
        IV_reg_pol_terms = ""

    # Binary indicator for being a switcher (in D, not direction): S_bis_XX = 1{S != 0}
    df["S_bis_XX"] = np.where(df["S_XX"].isna(), np.nan, (df["S_XX"] != 0).astype(float))

    # ------------------------------------------------------------
    # 16) Feasibility check:
    #     - Need no gaps (gap_XX==0),
    #     - At least one switcher, and at least 2 stayers
    # ------------------------------------------------------------
    if aoss == 1 or waoss == 1:
        feasible_est = (gap_XX == 0) and (scalars[f"N_Switchers{pl}_XX"] > 0) and (scalars[f"N_Stayers{pl}_XX"] > 1)
    else:
        feasible_est = (gap_XX == 0) and (scalars[f"N_Switchers_IV{pl}_XX"] > 0) and (scalars[f"N_Stayers_IV{pl}_XX"] > 1)

    # P_Ht is the share of observations with usable diffs (weighted mean of Ht_XX)
    scalars[f"P_Ht_{pairwise}{pl}_XX"] = Mean("Ht_XX", df)

    # ------------------------------------------------------------
    # 17) Cluster preparation: N_bar_c = avg #IDs per cluster in this pairwise chunk.
    #     In R, they compute:
    #       - first row per ID => id_temp
    #       - sum id_temp within cluster => N_c
    #       - N_bar_c = mean(N_c)
    # ------------------------------------------------------------
    cluster_col = None
    if cluster is not None:
        # Use existing cluster_XX if present; else create it from the provided column name.
        if "cluster_XX" in df.columns:
            cluster_col = "cluster_XX"
        elif cluster in df.columns:
            df["cluster_XX"] = df[cluster]
            cluster_col = "cluster_XX"
        else:
            raise ValueError("cluster specified but neither cluster_XX nor the given cluster column exists in df.")

        # Ensure cluster weights exist
        if "weight_c_XX" not in df.columns:
            df["weight_c_XX"] = 1.0

        df["_first_in_id"] = df.groupby("ID_XX").cumcount().eq(0).astype(float)
        df["_Nc"] = df.groupby(cluster_col)["_first_in_id"].transform(lambda s: np.nansum(s.to_numpy(dtype=float)))
        scalars[f"N_bar_c_{pairwise}{pl}_XX"] = float(np.nanmean(df["_Nc"].to_numpy(dtype=float)))
        df.drop(columns=["_first_in_id", "_Nc"], inplace=True)

    # ============================================================
    # 18) Start estimation if feasible; else fill defaults like R
    # ============================================================
    if feasible_est:

        # ------------------------------------------------------------
        # 18A) Common preliminaries for (A)OSS and WAOSS:
        #      - Fit RA on stayers: E(deltaY|D1, S=0)
        #      - Build residualized outcome: inner_sum_delta_1_2 = deltaY - Ehat(deltaY|.)
        #      - Build S0 = 1 - S_bis
        #      - Estimate P(S=0|D1) via logit unless exact_match
        # ------------------------------------------------------------
        if waoss == 1 or aoss == 1:
            df0 = df[df["S_XX"] == 0].copy()

            # RA model on stayers: delta_Y_XX ~ polynomial(D1) (+ interactions)
            ra_formula = f"delta_Y_XX ~ {reg_pol_terms}"
            ra_model = smf.wls(ra_formula, data=df0, weights=df0["weight_XX"]).fit()
            df = lpredict(df, "mean_pred_XX", ra_model)

            # Residualized deltaY
            df["inner_sum_delta_1_2_XX"] = df["delta_Y_XX"] - df["mean_pred_XX"]

            # S0 = 1 - 1{switcher}
            df["S0_XX"] = 1.0 - df["S_bis_XX"]

            if not exact_match:
                # PS_0_D_1_XX = P(S=0|D1)
                ps0_formula = f"S0_XX ~ {reg_pol_terms}"
                ps0_model = stata_logit(ps0_formula, df)
                df = lpredict(df, "PS_0_D_1_XX", ps0_model, prob=True)
            else:
                # Exact-match: estimate E(S_bis|D1) then 1 - E(S_bis|D1)
                esbis_formula = f"S_bis_XX ~ {reg_pol_terms}"
                esbis_model = smf.wls(esbis_formula, data=df, weights=df["weight_XX"]).fit()
                df = lpredict(df, "ES_bis_XX_D_1", esbis_model)

                # Also estimate E(S|D1) for Phi_2 exact-match case
                es_formula = f"S_XX ~ {reg_pol_terms}"
                es_model = smf.wls(es_formula, data=df, weights=df["weight_XX"]).fit()
                df = lpredict(df, "ES_XX_D_1", es_model)

            scalars[f"PS_0{pl}_XX"] = Mean("S0_XX", df)

        # ------------------------------------------------------------
        # 18B) AOSS estimation (delta_1) and its influence function Phi_1
        # ------------------------------------------------------------
        if aoss == 1:
            # P_p = E(S_bis)*P_Ht (used later in aggregation across p)
            ES = Mean("S_bis_XX", df)
            scalars[f"ES{pl}_XX"] = ES

            scalars[f"P_{pairwise}{pl}_XX"] = ES * scalars[f"P_Ht_{pairwise}{pl}_XX"]
            scalars[f"PS_sum{pl}_XX"] = scalars.get(f"PS_sum{pl}_XX", 0.0) + scalars[f"P_{pairwise}{pl}_XX"]

            # delta_1 = E( (deltaY - mhat)/deltaD ) among switchers (deltaD != 0)
            df["inner_sum_delta_1_XX"] = df["inner_sum_delta_1_2_XX"] / df["delta_D_XX"]
            df.loc[df["delta_D_XX"] == 0, "inner_sum_delta_1_XX"] = np.nan
            scalars[f"delta_1_{pairwise}{pl}_XX"] = Mean("inner_sum_delta_1_XX", df)

            # For IF: need E(S/deltaD | D1)
            df["S_over_delta_D_XX"] = df["S_bis_XX"] / df["delta_D_XX"]
            df.loc[df["S_bis_XX"] == 0, "S_over_delta_D_XX"] = 0.0

            sdd_formula = f"S_over_delta_D_XX ~ {reg_pol_terms}"
            sdd_model = smf.wls(sdd_formula, data=df, weights=df["weight_XX"]).fit()
            df = lpredict(df, "mean_S_over_delta_D_XX", sdd_model)

            # Phi_1 raw
            if not exact_match:
                adj = (1.0 - df["S_bis_XX"]) / df["PS_0_D_1_XX"]
                df[f"Phi_1_{pairwise}{pl}_XX"] = (
                    (df["S_over_delta_D_XX"] - df["mean_S_over_delta_D_XX"] * adj)
                    * df["inner_sum_delta_1_2_XX"]
                )
            else:
                adj = (1.0 - df["S_bis_XX"]) / (1.0 - df["ES_bis_XX_D_1"])
                df[f"Phi_1_{pairwise}{pl}_XX"] = (
                    (df["S_over_delta_D_XX"] - df["mean_S_over_delta_D_XX"] * adj)
                    * df["inner_sum_delta_1_2_XX"]
                )

            # Normalize to make it an IF for delta_1
            df[f"Phi_1_{pairwise}{pl}_XX"] = (
                df[f"Phi_1_{pairwise}{pl}_XX"] - (scalars[f"delta_1_{pairwise}{pl}_XX"] * df["S_bis_XX"])
            ) / (ES * scalars[f"P_Ht_{pairwise}{pl}_XX"])

            # Set IF=0 when Ht=0
            df.loc[df["Ht_XX"] == 0, f"Phi_1_{pairwise}{pl}_XX"] = 0.0

            # Standard error of delta_1 (clustered or not)
            if cluster_col is not None:
                # Sum IF within cluster, keep one row per cluster, normalize by N_bar_c
                phi = f"Phi_1_{pairwise}{pl}_XX"
                df["_phi_c"] = df.groupby(cluster_col)[phi].transform(lambda s: np.nansum(s.to_numpy(dtype=float)))
                df["_first_clus"] = df.groupby(cluster_col).cumcount().eq(0)
                df["_phi_c"] = np.where(df["_first_clus"], df["_phi_c"], np.nan) / scalars[f"N_bar_c_{pairwise}{pl}_XX"]

                nobs_c = wSum(df[~df["_phi_c"].isna()], w="weight_c_XX")
                sd_phi = Sd("_phi_c", df, w="weight_c_XX") / np.sqrt(nobs_c) if nobs_c > 0 else np.nan
                scalars[f"sd_delta_1_{pairwise}{pl}_XX"] = sd_phi

                df.drop(columns=["_phi_c", "_first_clus"], inplace=True)
            else:
                # R uses unweighted sd()/sqrt(wSum(df)) here
                phi_vals = df[f"Phi_1_{pairwise}{pl}_XX"].to_numpy(dtype=float)
                sd_unw = np.nanstd(phi_vals, ddof=1)
                scalars[f"sd_delta_1_{pairwise}{pl}_XX"] = sd_unw / np.sqrt(wSum(df))

            # 95% CI with normal critical value (matches R)
            se = scalars[f"sd_delta_1_{pairwise}{pl}_XX"]
            scalars[f"LB_1_{pairwise}{pl}_XX"] = scalars[f"delta_1_{pairwise}{pl}_XX"] - 1.96 * se
            scalars[f"UB_1_{pairwise}{pl}_XX"] = scalars[f"delta_1_{pairwise}{pl}_XX"] + 1.96 * se

            # Store S_p indicator (used for aggregation)
            df[f"S_{pairwise}{pl}_XX"] = df["S_bis_XX"]
            df.loc[df["Ht_XX"] == 0, f"S_{pairwise}{pl}_XX"] = 0.0

        # ------------------------------------------------------------
        # 18C) WAOSS estimation (delta_2) and its influence function Phi_2
        # ------------------------------------------------------------
        if waoss == 1:
            scalars[f"E_abs_delta_D{pl}_XX"] = Mean("abs_delta_D_XX", df)
            scalars[f"E_abs_delta_D_{pairwise}{pl}_XX"] = scalars[f"E_abs_delta_D{pl}_XX"] * scalars[f"P_Ht_{pairwise}{pl}_XX"]
            scalars[f"E_abs_delta_D_sum{pl}_XX"] = scalars.get(f"E_abs_delta_D_sum{pl}_XX", 0.0) + scalars[f"E_abs_delta_D_{pairwise}{pl}_XX"]

            # Compute delta_2 separately for Plus and Minus switchers, then combine
            for suffix in ("Minus", "Plus"):
                target_S = 1.0 if suffix == "Plus" else -1.0
                df["Ster_XX"] = (df["S_XX"] == target_S)

                # Contribution weights w_suffix based on Σ |deltaD| among that suffix, divided by N
                df["prod_sgn_delta_D_delta_D_XX"] = df["S_XX"] * df["delta_D_XX"]  # = |deltaD|
                sum_prod = Sum("prod_sgn_delta_D_delta_D_XX", df[df["Ster_XX"] == 1])
                scalars[f"w_{suffix}_{pairwise}{pl}_XX"] = sum_prod / scalars[f"N{pl}_XX"]

                denom = Sum("delta_D_XX", df[df["Ster_XX"] == 1])
                scalars[f"denom_delta_2_{suffix}_{pairwise}{pl}_XX"] = denom

                if estimation_method == "ra":
                    if denom == 0:
                        denom = 1.0  # avoid divide-by-zero; numerator is also ~0 in that case
                        scalars[f"denom_delta_2_{suffix}_{pairwise}{pl}_XX"] = denom
                    num = Sum("inner_sum_delta_1_2_XX", df[df["Ster_XX"] == 1])
                    scalars[f"num_delta_2_{suffix}_{pairwise}{pl}_XX"] = num
                    scalars[f"delta_2_{suffix}_{pairwise}{pl}_XX"] = num / denom

                # Switcher shares PS_suffix1 = N_suffix / N
                nb_sw = float(df[df["Ster_XX"] == 1].shape[0])
                scalars[f"nb_Switchers_{suffix}{pl}_XX"] = nb_sw
                scalars[f"PS_{suffix}1{pl}_XX"] = nb_sw / scalars[f"N{pl}_XX"] if scalars[f"N{pl}_XX"] > 0 else 0.0

                if not exact_match:
                    if scalars[f"PS_{suffix}1{pl}_XX"] == 0:
                        scalars[f"delta_2_{suffix}_{pairwise}{pl}_XX"] = 0.0
                        df[f"PS_1_{suffix}_D_1_XX"] = 0.0
                    else:
                        # P(Ster=1 | D1) via logit
                        ps1_formula = f"Ster_XX ~ {reg_pol_terms}"
                        ps1_model = stata_logit(ps1_formula, df)
                        df = lpredict(df, f"PS_1_{suffix}_D_1_XX", ps1_model, prob=True)

                        if estimation_method == "ps":
                            # Reweight stayers to match suffix switchers
                            df[f"delta_Y_P_{suffix}_XX"] = (
                                df["delta_Y_XX"]
                                * (df[f"PS_1_{suffix}_D_1_XX"] / df["PS_0_D_1_XX"])
                                * (scalars[f"PS_0{pl}_XX"] / scalars[f"PS_{suffix}1{pl}_XX"])
                            )
                            mean_delta_Y_P = Mean(f"delta_Y_P_{suffix}_XX", df[df["S_XX"] == 0])
                            mean_delta_Y = Mean("delta_Y_XX", df[df["Ster_XX"] == 1])
                            mean_delta_D = Mean("delta_D_XX", df[df["Ster_XX"] == 1])
                            scalars[f"delta_2_{suffix}_{pairwise}{pl}_XX"] = (mean_delta_Y - mean_delta_Y_P) / mean_delta_D

            # Final weight W_Plus
            if estimation_method in ("ra", "ps"):
                w_plus = scalars.get(f"w_Plus_{pairwise}{pl}_XX", 0.0)
                w_minus = scalars.get(f"w_Minus_{pairwise}{pl}_XX", 0.0)
                denomw = w_plus + w_minus
                scalars[f"W_Plus_{pairwise}{pl}_XX"] = (w_plus / denomw) if denomw != 0 else 0.0

            # DR residual term for WAOSS
            if not exact_match:
                df["dr_delta_Y_XX"] = (
                    (df["S_XX"] - ((df.get("PS_1_Plus_D_1_XX", 0.0) - df.get("PS_1_Minus_D_1_XX", 0.0)) / df["PS_0_D_1_XX"]) * (1.0 - df["S_bis_XX"]))
                    * df["inner_sum_delta_1_2_XX"]
                )
                scalars[f"denom_dr_delta_2{pl}_XX"] = Sum("dr_delta_Y_XX", df)

            # delta_2 overall
            if estimation_method in ("ra", "ps"):
                Wp = scalars[f"W_Plus_{pairwise}{pl}_XX"]
                scalars[f"delta_2_{pairwise}{pl}_XX"] = (
                    Wp * scalars[f"delta_2_Plus_{pairwise}{pl}_XX"] + (1.0 - Wp) * scalars[f"delta_2_Minus_{pairwise}{pl}_XX"]
                )
            elif estimation_method == "dr":
                sum_abs = Sum("abs_delta_D_XX", df)
                scalars[f"delta_2_{pairwise}{pl}_XX"] = scalars[f"denom_dr_delta_2{pl}_XX"] / sum_abs if sum_abs != 0 else 0.0

            # Influence function Phi_2
            if not exact_match:
                df[f"Phi_2_{pairwise}{pl}_XX"] = df["dr_delta_Y_XX"] - scalars[f"delta_2_{pairwise}{pl}_XX"] * df["abs_delta_D_XX"]
            else:
                df[f"Phi_2_{pairwise}{pl}_XX"] = (
                    (df["S_XX"] - df["ES_XX_D_1"] * ((1.0 - df["S_bis_XX"]) / (1.0 - df["ES_bis_XX_D_1"])))
                    * df["inner_sum_delta_1_2_XX"]
                    - scalars[f"delta_2_{pairwise}{pl}_XX"] * df["abs_delta_D_XX"]
                )

            denom_if = scalars[f"P_Ht_{pairwise}{pl}_XX"] * scalars[f"E_abs_delta_D{pl}_XX"]
            df[f"Phi_2_{pairwise}{pl}_XX"] = df[f"Phi_2_{pairwise}{pl}_XX"] / denom_if if denom_if != 0 else np.nan
            df.loc[df["Ht_XX"] == 0, f"Phi_2_{pairwise}{pl}_XX"] = 0.0

            # Standard error of delta_2
            if cluster_col is not None:
                phi = f"Phi_2_{pairwise}{pl}_XX"
                df["_phi_c"] = df.groupby(cluster_col)[phi].transform(lambda s: np.nansum(s.to_numpy(dtype=float)))
                df["_first_clus"] = df.groupby(cluster_col).cumcount().eq(0)
                df["_phi_c"] = np.where(df["_first_clus"], df["_phi_c"], np.nan) / scalars[f"N_bar_c_{pairwise}{pl}_XX"]

                nobs_c = wSum(df[~df["_phi_c"].isna()], w="weight_c_XX")
                sd_phi = Sd("_phi_c", df, w="weight_c_XX") / np.sqrt(nobs_c) if nobs_c > 0 else np.nan
                scalars[f"sd_delta_2_{pairwise}{pl}_XX"] = sd_phi

                df.drop(columns=["_phi_c", "_first_clus"], inplace=True)
            else:
                scalars[f"sd_delta_2_{pairwise}{pl}_XX"] = Sd(f"Phi_2_{pairwise}{pl}_XX", df) / np.sqrt(wSum(df))

            se = scalars[f"sd_delta_2_{pairwise}{pl}_XX"]
            scalars[f"LB_2_{pairwise}{pl}_XX"] = scalars[f"delta_2_{pairwise}{pl}_XX"] - 1.96 * se
            scalars[f"UB_2_{pairwise}{pl}_XX"] = scalars[f"delta_2_{pairwise}{pl}_XX"] + 1.96 * se

            # Store abs_delta_D_p for aggregation
            df[f"abs_delta_D_{pairwise}{pl}_XX"] = np.where(df["Ht_XX"] == 0, 0.0, df["abs_delta_D_XX"])

        # ------------------------------------------------------------
        # 18D) IV-WAOSS block (delta_3 and Phi_3)
        # NOTE: This is long but translated in the same structure as R.
        # ------------------------------------------------------------
        if ivwaoss == 1:
            scalars[f"E_abs_delta_Z{pl}_XX"] = Mean("abs_delta_Z_XX", df)

            df["SI_bis_XX"] = ((df["SI_XX"] != 0) & (~df["SI_XX"].isna())).astype(float)
            df["SI_Plus_XX"] = (df["SI_XX"] == 1).astype(float)
            df["SI_Minus_XX"] = (df["SI_XX"] == -1).astype(float)

            # P(SI=0|Z1) prelims
            df["S_IV_0_XX"] = 1.0 - df["SI_bis_XX"]

            if not exact_match:
                psiv0_formula = f"S_IV_0_XX ~ {IV_reg_pol_terms}"
                psiv0_model = stata_logit(psiv0_formula, df)
                df = lpredict(df, "PS_IV_0_Z_1_XX", psiv0_model, prob=True)
            else:
                esibis_formula = f"SI_bis_XX ~ {IV_reg_pol_terms}"
                esibis_model = smf.wls(esibis_formula, data=df, weights=df["weight_XX"]).fit()
                df = lpredict(df, "ES_I_bis_XX_Z_1", esibis_model)

                esi_formula = f"SI_XX ~ {IV_reg_pol_terms}"
                esi_model = smf.wls(esi_formula, data=df, weights=df["weight_XX"]).fit()
                df = lpredict(df, "ES_I_XX_Z_1", esi_model)

            scalars[f"PS_IV_0{pl}_XX"] = Mean("S_IV_0_XX", df)

            # P(SI=+1|Z1), P(SI=-1|Z1)
            for suffix in ("Minus", "Plus"):
                flag = "SI_Minus_XX" if suffix == "Minus" else "SI_Plus_XX"
                nb = float((df[flag] == 1).sum())
                scalars[f"nb_Switchers_I_{suffix}{pl}_XX"] = nb
                scalars[f"PS_I_{suffix}_1{pl}_XX"] = nb / scalars[f"N{pl}_XX"] if scalars[f"N{pl}_XX"] > 0 else 0.0

                if scalars[f"PS_I_{suffix}_1{pl}_XX"] == 0:
                    df[f"PS_I_{suffix}_1_Z_1_XX"] = 0.0
                else:
                    if not exact_match:
                        psis_formula = f"{flag} ~ {IV_reg_pol_terms}"
                        psis_model = stata_logit(psis_formula, df)
                        df = lpredict(df, f"PS_I_{suffix}_1_Z_1_XX", psis_model, prob=True)

            # RA for numerator: E(deltaY|Z1, SI=0)
            df_temp = df[df["SI_XX"] == 0].copy()
            mY_formula = f"delta_Y_XX ~ {IV_reg_pol_terms}"
            mY_model = smf.wls(mY_formula, data=df_temp, weights=df_temp["weight_XX"]).fit()
            df = lpredict(df, "mean_delta_Y_pred_IV_XX", mY_model)
            df["inner_sum_IV_num_XX"] = df["delta_Y_XX"] - df["mean_delta_Y_pred_IV_XX"]

            # RA for denominator: E(deltaD|Z1, SI=0)
            mD_formula = f"delta_D_XX ~ {IV_reg_pol_terms}"
            mD_model = smf.wls(mD_formula, data=df_temp, weights=df_temp["weight_XX"]).fit()
            df = lpredict(df, "mean_delta_D_pred_IV_XX", mD_model)
            df["inner_sum_IV_denom_XX"] = df["delta_D_XX"] - df["mean_delta_D_pred_IV_XX"]

            if estimation_method == "ra":
                df["inner_sum_IV_num_XX"] = df["inner_sum_IV_num_XX"] * df["SI_XX"]
                df["inner_sum_IV_denom_XX"] = df["inner_sum_IV_denom_XX"] * df["SI_XX"]
                scalars[f"num_delta_IV_{pairwise}{pl}_XX"] = Mean("inner_sum_IV_num_XX", df)
                scalars[f"denom_delta_IV_{pairwise}{pl}_XX"] = Mean("inner_sum_IV_denom_XX", df)

            if estimation_method == "ps":
                # Numerator
                df["delta_Y_P_IV_XX"] = (
                    df["delta_Y_XX"]
                    * ((df.get("PS_I_Plus_1_Z_1_XX", 0.0) - df.get("PS_I_Minus_1_Z_1_XX", 0.0)) / df["PS_IV_0_Z_1_XX"])
                    * scalars[f"PS_IV_0{pl}_XX"]
                )
                mean_delta_Y_P_IV = Mean("delta_Y_P_IV_XX", df[df["SI_bis_XX"] == 0])
                df["prod_sgn_delta_Z_delta_Y_XX"] = df["SI_XX"] * df["delta_Y_XX"]
                mean_sgn = Mean("prod_sgn_delta_Z_delta_Y_XX", df)
                scalars[f"num_delta_IV_{pairwise}{pl}_XX"] = mean_sgn - mean_delta_Y_P_IV

                # Denominator
                df["delta_D_P_IV_XX"] = (
                    df["delta_D_XX"]
                    * ((df.get("PS_I_Plus_1_Z_1_XX", 0.0) - df.get("PS_I_Minus_1_Z_1_XX", 0.0)) / df["PS_IV_0_Z_1_XX"])
                    * scalars[f"PS_IV_0{pl}_XX"]
                )
                mean_delta_D_P_IV = Mean("delta_D_P_IV_XX", df[df["SI_bis_XX"] == 0])
                df["prod_sgn_delta_Z_delta_D_XX"] = df["SI_XX"] * df["delta_D_XX"]
                mean_sgnD = Mean("prod_sgn_delta_Z_delta_D_XX", df)
                scalars[f"denom_delta_IV_{pairwise}{pl}_XX"] = mean_sgnD - mean_delta_D_P_IV

            if estimation_method == "dr":
                df["dr_IV_delta_Y_XX"] = (
                    (df["SI_XX"] - ((df.get("PS_I_Plus_1_Z_1_XX", 0.0) - df.get("PS_I_Minus_1_Z_1_XX", 0.0)) / df["PS_IV_0_Z_1_XX"]) * (1.0 - df["SI_bis_XX"]))
                    * df["inner_sum_IV_num_XX"]
                )
                scalars[f"num_delta_IV_{pairwise}{pl}_XX"] = Mean("dr_IV_delta_Y_XX", df)

                df["dr_IV_delta_D_XX"] = (
                    (df["SI_XX"] - ((df.get("PS_I_Plus_1_Z_1_XX", 0.0) - df.get("PS_I_Minus_1_Z_1_XX", 0.0)) / df["PS_IV_0_Z_1_XX"]) * (1.0 - df["SI_bis_XX"]))
                    * df["inner_sum_IV_denom_XX"]
                )
                scalars[f"denom_delta_IV_{pairwise}{pl}_XX"] = Mean("dr_IV_delta_D_XX", df)

            scalars[f"delta_3_{pairwise}{pl}_XX"] = (
                scalars[f"num_delta_IV_{pairwise}{pl}_XX"] / scalars[f"denom_delta_IV_{pairwise}{pl}_XX"]
                if scalars[f"denom_delta_IV_{pairwise}{pl}_XX"] != 0 else np.nan
            )

            scalars[f"denom_delta_IV_sum{pl}_XX"] = scalars.get(f"denom_delta_IV_sum{pl}_XX", 0.0) + scalars[f"denom_delta_IV_{pairwise}{pl}_XX"]

            # Build Phi_Y and Phi_D then Phi_3 as in R
            scalars[f"delta_Y{pl}_XX"] = Mean("inner_sum_IV_num_XX", df)

            # mean_pred_Y_IV_XX: regress deltaY on D1 polynomials among SI==0
            df_temp2 = df[df["SI_XX"] == 0].copy()
            mY2_formula = f"delta_Y_XX ~ {reg_pol_terms}"
            mY2_model = smf.wls(mY2_formula, data=df_temp2, weights=df_temp2["weight_XX"]).fit()
            df = lpredict(df, "mean_pred_Y_IV_XX", mY2_model)

            if not exact_match:
                df["Phi_Y_XX"] = (
                    (
                        df["SI_XX"]
                        - (df.get("PS_I_Plus_1_Z_1_XX", 0.0) - df.get("PS_I_Minus_1_Z_1_XX", 0.0))
                        * (1.0 - df["SI_bis_XX"]) / df["PS_IV_0_Z_1_XX"]
                    )
                    * (df["delta_Y_XX"] - df["mean_pred_Y_IV_XX"])
                    - scalars[f"delta_Y{pl}_XX"] * df["abs_delta_Z_XX"]
                ) / scalars[f"E_abs_delta_Z{pl}_XX"]
            else:
                df["Phi_Y_XX"] = (
                    (df["SI_XX"] - df["ES_I_XX_Z_1"] * ((1.0 - df["SI_bis_XX"]) / (1.0 - df["ES_I_bis_XX_Z_1"])))
                    * (df["delta_Y_XX"] - df["mean_pred_Y_IV_XX"])
                    - scalars[f"delta_Y{pl}_XX"] * df["abs_delta_Z_XX"]
                ) / scalars[f"E_abs_delta_Z{pl}_XX"]

            scalars[f"delta_D{pl}_XX"] = Mean("inner_sum_IV_denom_XX", df)

            # mean_pred_D_IV_XX: regress deltaD on D1 polynomials among SI==0
            mD2_formula = f"delta_D_XX ~ {reg_pol_terms}"
            mD2_model = smf.wls(mD2_formula, data=df_temp2, weights=df_temp2["weight_XX"]).fit()
            df = lpredict(df, "mean_pred_D_IV_XX", mD2_model)

            if not exact_match:
                df["Phi_D_XX"] = (
                    (
                        df["SI_XX"]
                        - (df.get("PS_I_Plus_1_Z_1_XX", 0.0) - df.get("PS_I_Minus_1_Z_1_XX", 0.0))
                        * (1.0 - df["SI_bis_XX"]) / df["PS_IV_0_Z_1_XX"]
                    )
                    * (df["delta_D_XX"] - df["mean_pred_D_IV_XX"])
                    - scalars[f"delta_D{pl}_XX"] * df["abs_delta_Z_XX"]
                ) / scalars[f"E_abs_delta_Z{pl}_XX"]
            else:
                df["Phi_D_XX"] = (
                    (df["SI_XX"] - df["ES_I_XX_Z_1"] * ((1.0 - df["SI_bis_XX"]) / (1.0 - df["ES_I_bis_XX_Z_1"])))
                    * (df["delta_D_XX"] - df["mean_pred_D_IV_XX"])
                    - scalars[f"delta_D{pl}_XX"] * df["abs_delta_Z_XX"]
                ) / scalars[f"E_abs_delta_Z{pl}_XX"]

            if scalars[f"delta_D{pl}_XX"] != 0:
                df[f"Phi_3_{pairwise}{pl}_XX"] = (
                    (df["Phi_Y_XX"] - scalars[f"delta_3_{pairwise}{pl}_XX"] * df["Phi_D_XX"])
                    / scalars[f"delta_D{pl}_XX"]
                )

                # Standard error for delta_3
                if cluster_col is not None:
                    phi = f"Phi_3_{pairwise}{pl}_XX"
                    df["_phi_c"] = df.groupby(cluster_col)[phi].transform(lambda s: np.nansum(s.to_numpy(dtype=float)))
                    df["_first_clus"] = df.groupby(cluster_col).cumcount().eq(0)
                    df["_phi_c"] = np.where(df["_first_clus"], df["_phi_c"], np.nan) / scalars[f"N_bar_c_{pairwise}{pl}_XX"]

                    nobs_c = wSum(df[~df["_phi_c"].isna()], w="weight_c_XX")
                    sd_phi = Sd("_phi_c", df, w="weight_c_XX") / np.sqrt(nobs_c) if nobs_c > 0 else np.nan
                    scalars[f"sd_delta_3_{pairwise}{pl}_XX"] = sd_phi

                    df.drop(columns=["_phi_c", "_first_clus"], inplace=True)
                else:
                    scalars[f"sd_delta_3_{pairwise}{pl}_XX"] = Sd(f"Phi_3_{pairwise}{pl}_XX", df) / np.sqrt(wSum(df))

                se = scalars[f"sd_delta_3_{pairwise}{pl}_XX"]
                scalars[f"LB_3_{pairwise}{pl}_XX"] = scalars[f"delta_3_{pairwise}{pl}_XX"] - 1.96 * se
                scalars[f"UB_3_{pairwise}{pl}_XX"] = scalars[f"delta_3_{pairwise}{pl}_XX"] + 1.96 * se
            else:
                df[f"Phi_3_{pairwise}{pl}_XX"] = np.nan
                scalars[f"sd_delta_3_{pairwise}{pl}_XX"] = np.nan
                scalars[f"LB_3_{pairwise}{pl}_XX"] = np.nan
                scalars[f"UB_3_{pairwise}{pl}_XX"] = np.nan

            # Keep inner_sum_IV_denom for aggregation
            df[f"inner_sum_IV_denom_{pairwise}{pl}_XX"] = df["inner_sum_IV_denom_XX"]

        scalars[f"non_missing_{pairwise}{pl}_XX"] = 1.0

    else:
        # ------------------------------------------------------------
        # 19) Not feasible => set outputs like the R fallback block
        # ------------------------------------------------------------
        for i in (1, 2, 3):
            scalars[f"delta_{i}_{pairwise}{pl}_XX"] = 0.0
            scalars[f"sd_delta_{i}_{pairwise}{pl}_XX"] = np.nan
            scalars[f"LB_{i}_{pairwise}{pl}_XX"] = np.nan
            scalars[f"UB_{i}_{pairwise}{pl}_XX"] = np.nan
            df[f"Phi_{i}_{pairwise}{pl}_XX"] = np.nan

        IVt = "" if (aoss == 1 or waoss == 1) else "_IV"

        if gap_XX != 0:
            scalars[f"N_Switchers{IVt}{pl}_XX"] = np.nan
            scalars[f"N_Stayers{IVt}{pl}_XX"] = np.nan

        if not np.isnan(scalars.get(f"N_Stayers{IVt}{pl}_XX", np.nan)) and scalars[f"N_Stayers{IVt}{pl}_XX"] < 2:
            scalars[f"N_Switchers{IVt}{pl}_XX"] = float(len(df))
            scalars[f"N_Stayers{IVt}{pl}_XX"] = 0.0

        if not np.isnan(scalars.get(f"N_Switchers{IVt}{pl}_XX", np.nan)) and scalars[f"N_Switchers{IVt}{pl}_XX"] == 0:
            scalars[f"N_Switchers{IVt}{pl}_XX"] = 0.0
            scalars[f"N_Stayers{IVt}{pl}_XX"] = float(len(df))

        df[f"abs_delta_D_{pairwise}{pl}_XX"] = np.nan
        df[f"S_{pairwise}{pl}_XX"] = np.nan

        if aoss == 1:
            scalars[f"P_{pairwise}{pl}_XX"] = 0.0
        if waoss == 1:
            scalars[f"E_abs_delta_D_{pairwise}{pl}_XX"] = 0.0
        if ivwaoss == 1:
            scalars[f"denom_delta_IV_{pairwise}{pl}_XX"] = 0.0

        scalars[f"non_missing_{pairwise}{pl}_XX"] = 0.0

    # ------------------------------------------------------------
    # 20) Prepare the "to_add" DataFrame to return upward for aggregation
    #     Keep the same columns as R does.
    # ------------------------------------------------------------
    df = df.sort_values(["ID_XX"], kind="mergesort").reset_index(drop=True)

    keep_cols = [
        "ID_XX",
        f"Phi_1_{pairwise}{pl}_XX",
        f"Phi_2_{pairwise}{pl}_XX",
        f"Phi_3_{pairwise}{pl}_XX",
        f"S_{pairwise}{pl}_XX",
        f"abs_delta_D_{pairwise}{pl}_XX",
        f"used_in_{pairwise}{pl}_XX",
        f"inner_sum_IV_denom_{pairwise}{pl}_XX",
    ]

    # Add cluster column if requested
    if cluster is not None:
        if "cluster_XX" in df.columns:
            keep_cols.append("cluster_XX")
        elif cluster in df.columns:
            keep_cols.append(cluster)

    # Select only those that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    out_df = df.loc[:, keep_cols].copy()

    # ------------------------------------------------------------
    # 21) Update the scalars dict with the per-pair outputs (mirrors R final block)
    # ------------------------------------------------------------
    if aoss == 1:
        scalars[f"P_{pairwise}{pl}_XX"] = scalars.get(f"P_{pairwise}{pl}_XX", 0.0)
    if waoss == 1:
        scalars[f"E_abs_delta_D_{pairwise}{pl}_XX"] = scalars.get(f"E_abs_delta_D_{pairwise}{pl}_XX", 0.0)
    if ivwaoss == 1:
        scalars[f"denom_delta_IV_{pairwise}{pl}_XX"] = scalars.get(f"denom_delta_IV_{pairwise}{pl}_XX", 0.0)

    scalars[f"non_missing_{pairwise}{pl}_XX"] = scalars.get(f"non_missing_{pairwise}{pl}_XX", 0.0)

    # Populate N_Switchers / N_Stayers into N_{Switchers,Stayers}_{1,2,3}_p
    if waoss == 1 or aoss == 1:
        scalars[f"N_Switchers_1_{pairwise}{pl}_XX"] = scalars.get(f"N_Switchers{pl}_XX", 0.0)
        scalars[f"N_Stayers_1_{pairwise}{pl}_XX"] = scalars.get(f"N_Stayers{pl}_XX", 0.0)
        scalars[f"N_Switchers_2_{pairwise}{pl}_XX"] = scalars.get(f"N_Switchers{pl}_XX", 0.0)
        scalars[f"N_Stayers_2_{pairwise}{pl}_XX"] = scalars.get(f"N_Stayers{pl}_XX", 0.0)

    if ivwaoss == 1:
        scalars[f"N_Switchers_3_{pairwise}{pl}_XX"] = scalars.get(f"N_Switchers_IV{pl}_XX", 0.0)
        scalars[f"N_Stayers_3_{pairwise}{pl}_XX"] = scalars.get(f"N_Stayers_IV{pl}_XX", 0.0)

    # For each active estimator i, store delta_i, sd, LB, UB
    for i, active in enumerate([aoss, waoss, ivwaoss], start=1):
        if active == 1:
            scalars[f"delta_{i}_{pairwise}{pl}_XX"] = scalars.get(f"delta_{i}_{pairwise}{pl}_XX", 0.0)
            scalars[f"sd_delta_{i}_{pairwise}{pl}_XX"] = scalars.get(f"sd_delta_{i}_{pairwise}{pl}_XX", np.nan)
            scalars[f"LB_{i}_{pairwise}{pl}_XX"] = scalars.get(f"LB_{i}_{pairwise}{pl}_XX", np.nan)
            scalars[f"UB_{i}_{pairwise}{pl}_XX"] = scalars.get(f"UB_{i}_{pairwise}{pl}_XX", np.nan)

    return {"scalars": scalars, "to_add": out_df}

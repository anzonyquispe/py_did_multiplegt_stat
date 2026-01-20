from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, Tuple
import warnings
from did_multiplegt_stat_main import did_multiplegt_stat_main_py
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None  # if scipy is missing, we will skip t-based p-values


# =============================================================================
# Helper objects to mimic R output structure (args + results + by/by_fd extras)
# =============================================================================

@dataclass
class DidMultiplegtStat:
    args: Dict[str, Any]
    by_levels: List[Any]
    results: Dict[str, Any]
    quantiles: Optional[np.ndarray] = None
    switchers_df: Optional[pd.DataFrame] = None
    cdf_plot: Optional[pd.DataFrame] = None
    by_graph: Optional[Any] = None
    by_fd_graph: Optional[Any] = None


# =============================================================================
# Validation helpers (mimic R checks)
# =============================================================================

def _is_single_string(x) -> bool:
    return isinstance(x, str)

def _as_list_of_strings(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        if not all(isinstance(v, str) for v in x):
            raise TypeError("Expected a list/tuple of strings.")
        return list(x)
    raise TypeError("Expected a string or a list/tuple of strings.")

def by_check(df: pd.DataFrame, id_col: str, by_col: str) -> bool:
    """
    R: by_check(df, ID, v) checks that ID is nested within by variable,
    i.e. by_col is time-invariant within each ID.
    """
    tmp = df.groupby(id_col)[by_col].nunique(dropna=False)
    return bool((tmp <= 1).all())

def _tsfill_panel(df: pd.DataFrame, id_col: str, t_col: str) -> pd.DataFrame:
    """
    Approximate R's tsfill/pdata.frame behavior:
    - Create full ID x time grid
    - Mark inserted rows with tsfilled_XX = 1
    This prevents computing diffs across gaps as if they were consecutive.
    """
    base = df.copy()
    base = base.sort_values([id_col, t_col]).reset_index(drop=True)

    ids = base[id_col].dropna().unique()
    times = np.sort(base[t_col].dropna().unique())

    full_index = pd.MultiIndex.from_product([ids, times], names=[id_col, t_col])
    base = base.set_index([id_col, t_col]).reindex(full_index).reset_index()

    # Mark rows that were not present originally (inserted)
    base["tsfilled_XX"] = base.isna().all(axis=1).astype(float)

    return base


# =============================================================================
# by_fd quantiles helper (mimics did_multiplegt_stat_quantiles)
# =============================================================================

def did_multiplegt_stat_quantiles_py(
    *,
    df: pd.DataFrame,
    ID: str,
    Time: str,
    D: str,
    Z: Optional[str],
    by_opt: int,
    quantiles: List[float],
    estimator: List[str],
) -> Dict[str, Any]:
    """
    Build partition_XX based on |ΔD| (or |ΔZ| for ivwaoss) across consecutive periods.
    Store partition_XX on the *current* period row (time t), so that pairwise can use
    partition_lead_XX on the baseline row (time t-1).

    This function returns:
    - df with new column partition_XX
    - val_quantiles: numeric cutpoints
    - quantiles: the probability cutpoints
    - switch_df: dataframe of switcher diffs used to build bins
    - quantiles_plot: data for an empirical CDF plot (optional)
    """
    out = df.copy()

    diff_var = Z if ("ivwaoss" in estimator) else D
    if diff_var is None:
        raise ValueError("diff_var is None but by_fd requested (need D or Z).")

    # Work on a sorted panel
    out = out.sort_values([ID, Time]).reset_index(drop=True)

    # Compute first difference at time t (current row minus lag)
    lag = out.groupby(ID)[diff_var].shift(1)
    delta = out[diff_var].astype(float) - lag.astype(float)
    abs_delta = delta.abs()

    # Switchers = abs_delta > 0 and finite
    m = np.isfinite(abs_delta.to_numpy()) & (abs_delta.to_numpy() > 0)

    switch_abs = abs_delta[m].to_numpy()
    if switch_abs.size == 0:
        # No switchers: partition all zeros
        out["partition_XX"] = 0
        val_q = np.array([np.nan] * len(quantiles), dtype=float)
    else:
        val_q = np.quantile(switch_abs, quantiles)

        # Assign bins 1..by_opt based on (val_q[k-1], val_q[k]] with first bin inclusive
        part = np.zeros(len(out), dtype=float)

        # For each row that is a switcher-diff row, assign bin
        x = abs_delta.to_numpy()
        for k in range(1, by_opt + 1):
            lo = val_q[k - 1]
            hi = val_q[k]
            if k == 1:
                in_bin = m & (x >= lo) & (x <= hi)
            else:
                in_bin = m & (x > lo) & (x <= hi)
            part[in_bin] = float(k)

        out["partition_XX"] = part

    # switchers_df: useful debug table
    switch_df = out.loc[m, [ID, Time, diff_var]].copy()
    switch_df["abs_delta"] = abs_delta[m].to_numpy()
    switch_df["partition_XX"] = out.loc[m, "partition_XX"].to_numpy()

    # quantiles_plot (empirical CDF table)
    if switch_abs.size > 0:
        xs = np.sort(switch_abs)
        ys = np.arange(1, xs.size + 1) / xs.size
        quantiles_plot = pd.DataFrame({"abs_delta": xs, "cdf": ys})
    else:
        quantiles_plot = pd.DataFrame({"abs_delta": [], "cdf": []})

    return {
        "df": out,
        "val_quantiles": val_q,
        "quantiles": np.array(quantiles, dtype=float),
        "switch_df": switch_df,
        "quantiles_plot": quantiles_plot,
    }


# =============================================================================
# Minimal graph placeholders (R creates nice plots; here we keep plot-ready data)
# =============================================================================

def by_graph(obj: DidMultiplegtStat) -> Any:
    """Placeholder: return something plot-ready if you want."""
    return None

def by_fd_graph(obj: DidMultiplegtStat) -> Any:
    """Placeholder: return something plot-ready if you want."""
    return None


# =============================================================================
# Main internal routine (best-effort, uses your did_multiplegt_stat_pairwise_py)
# =============================================================================

def did_multiplegt_stat_main_py(
    *,
    df: pd.DataFrame,
    Y: str,
    ID: str,
    Time: str,
    D: str,
    Z: Optional[str],
    estimator: List[str],
    estimation_method: str,
    order: int,
    noextrapolation: bool,
    placebo: bool,
    weight: Optional[str],
    switchers: Optional[str],
    disaggregate: bool,
    aoss_vs_waoss: bool,
    exact_match: bool,
    cluster: Optional[str],
    by_fd_opt: Optional[Any],
    other_treatments: Optional[List[str]],
) -> Dict[str, Any]:
    """
    Best-effort Python analog of did_multiplegt_stat_main().
    It:
      1) maps Time -> consecutive integers 1..T
      2) tsfills the panel (creates tsfilled_XX)
      3) loops over pairwise = 2..T and calls did_multiplegt_stat_pairwise_py()
      4) aggregates point estimates and (approx) SEs using weighted sums of pairwise IFs.

    NOTE:
    - Exact 1:1 match with R may require porting the exact R main file.
    - This is still a usable pipeline to get estimates and a table.
    """
    df0 = df.copy()

    other_treatments = other_treatments or []
    if isinstance(other_treatments, str):
        other_treatments = [other_treatments]

    # Map Time to 1..T (like cur_group_id in R)
    time_levels = np.sort(df0[Time].dropna().unique())
    time_map = {t: i + 1 for i, t in enumerate(time_levels)}
    df0["_T_int_XX"] = df0[Time].map(time_map).astype(float)

    # Fill panel grid and create tsfilled_XX
    df0 = _tsfill_panel(df0, id_col=ID, t_col="_T_int_XX")

    # Determine which estimators are requested
    aoss = 1 if "aoss" in estimator else 0
    waoss = 1 if "waoss" in estimator else 0
    ivwaoss = 1 if "ivwaoss" in estimator else 0

    # Accumulators (scalars dict is passed into pairwise and updated)
    scalars: Dict[str, Any] = {}

    # Collect pairwise IF chunks
    chunks: List[pd.DataFrame] = []

    T_max = int(np.nanmax(df0["_T_int_XX"].to_numpy())) if df0.shape[0] else 0
    pairs = list(range(2, T_max + 1))  # pairwise indices

    for pairwise in pairs:
        out = did_multiplegt_stat_pairwise_py(
            df0,
            Y=Y, ID=ID, Time="_T_int_XX", D=D, Z=Z,
            estimator=None,
            order=order,
            noextrapolation=noextrapolation,
            weight=weight,
            switchers=switchers,
            pairwise=pairwise,
            IDs=None,
            aoss=aoss,
            waoss=waoss,
            ivwaoss=ivwaoss,
            estimation_method=estimation_method,
            scalars=scalars,
            placebo=placebo,
            exact_match=exact_match,
            cluster=cluster,
            by_fd_opt=by_fd_opt,
            other_treatments=other_treatments,
        )
        scalars = out.scalars  # updated
        if out.to_add is not None:
            chunks.append(out.to_add)

    # Merge all IF chunks by ID_XX
    if chunks:
        if_df = chunks[0]
        for c in chunks[1:]:
            if_df = if_df.merge(c, on="ID_XX", how="outer", suffixes=("", "_dup"))
            # drop accidental duplicates
            dup_cols = [col for col in if_df.columns if col.endswith("_dup")]
            if dup_cols:
                if_df.drop(columns=dup_cols, inplace=True)
    else:
        if_df = pd.DataFrame({"ID_XX": df0[ID].dropna().unique()})

    # Helper to compute weighted overall estimate and IF (approx)
    def _overall_from_pairs(est_idx: int) -> Tuple[float, float, float, float, pd.Series]:
        """
        Return (estimate, se, lb, ub, phi_by_id).
        Uses pairwise Phi_{est_idx}_{pair}_XX and pair-weights:
          - AOSS: weight = P_{pair}_XX
          - WAOSS: weight = E_abs_delta_D_{pair}_XX
          - IV: weight = denom_delta_IV_{pair}_XX (approx)
        """
        # Collect pairwise estimates and weights from scalars
        deltas = []
        weights = []
        phi_cols = []

        for p in pairs:
            # only use non-missing pairs
            nm = scalars.get(f"non_missing_{p}_XX", 0)
            if nm != 1:
                continue

            delta_key = f"delta_{est_idx}_{p}_XX"
            if delta_key not in scalars:
                continue

            # pair weights
            if est_idx == 1:
                w_key = f"P_{p}_XX"
            elif est_idx == 2:
                w_key = f"E_abs_delta_D_{p}_XX"
            else:
                w_key = f"denom_delta_IV_{p}_XX"

            w = scalars.get(w_key, np.nan)
            d = scalars.get(delta_key, np.nan)
            if not np.isfinite(w) or w <= 0:
                continue
            if not np.isfinite(d):
                continue

            deltas.append(d)
            weights.append(w)
            phi_cols.append(f"Phi_{est_idx}_{p}_XX")

        if len(weights) == 0:
            return (np.nan, np.nan, np.nan, np.nan, pd.Series(dtype=float))

        W = np.array(weights, dtype=float)
        Dv = np.array(deltas, dtype=float)
        W_sum = W.sum()
        est = float((W * Dv).sum() / W_sum) if W_sum > 0 else np.nan

        # Build overall influence as weighted sum of pairwise influences (approx)
        # phi_overall_i = sum_p (w_p/W_sum) * Phi_{p,i}
        phi = pd.Series(0.0, index=if_df["ID_XX"])
        for w, col in zip(W, phi_cols):
            if col in if_df.columns:
                phi = phi.add((w / W_sum) * if_df[col].fillna(0.0).to_numpy(), fill_value=0.0)

        # Standard error:
        # - default: treat IDs as independent clusters (closest to your use case)
        # - if cluster is provided, cluster on that variable if it survived into IF df (pairwise keeps cluster_XX)
        if "cluster_XX" in if_df.columns and if_df["cluster_XX"].notna().any():
            g = if_df.groupby("cluster_XX", dropna=True)
            phi_g = g.apply(lambda x: np.nansum(x.reindex(columns=[]).to_numpy()) if False else np.nansum(phi.loc[x.index].to_numpy()))
            # phi_g is a Series indexed by cluster
            phi_vals = phi_g.to_numpy()
            G = phi_vals.size
            se = float(np.std(phi_vals, ddof=1) / np.sqrt(G)) if G >= 2 else np.nan
        else:
            # ID-cluster robust
            phi_vals = phi.to_numpy()
            G = phi_vals.size
            se = float(np.std(phi_vals, ddof=1) / np.sqrt(G)) if G >= 2 else np.nan

        lb = est - 1.96 * se if np.isfinite(se) else np.nan
        ub = est + 1.96 * se if np.isfinite(se) else np.nan
        return (est, se, lb, ub, phi)

    # Build results table
    rows = []

    # Aggregate row per estimator
    if aoss == 1:
        est, se, lb, ub, phi1 = _overall_from_pairs(1)
        rows.append({"estimator": "aoss", "pair": "overall", "estimate": est, "se": se, "lb": lb, "ub": ub})
    if waoss == 1:
        est, se, lb, ub, phi2 = _overall_from_pairs(2)
        rows.append({"estimator": "waoss", "pair": "overall", "estimate": est, "se": se, "lb": lb, "ub": ub})
    if ivwaoss == 1:
        est, se, lb, ub, phi3 = _overall_from_pairs(3)
        rows.append({"estimator": "ivwaoss", "pair": "overall", "estimate": est, "se": se, "lb": lb, "ub": ub})

    # Disaggregated pairwise rows (optional)
    if disaggregate:
        for p in pairs:
            if scalars.get(f"non_missing_{p}_XX", 0) != 1:
                continue
            # map p -> original time labels (t-1, t)
            t0 = time_levels[p - 2] if (p - 2) < len(time_levels) else (p - 1)
            t1 = time_levels[p - 1] if (p - 1) < len(time_levels) else p
            pair_label = f"{t0}→{t1}"

            if aoss == 1:
                rows.append({
                    "estimator": "aoss",
                    "pair": pair_label,
                    "estimate": scalars.get(f"delta_1_{p}_XX", np.nan),
                    "se": scalars.get(f"sd_delta_1_{p}_XX", np.nan),
                    "lb": scalars.get(f"LB_1_{p}_XX", np.nan),
                    "ub": scalars.get(f"UB_1_{p}_XX", np.nan),
                })
            if waoss == 1:
                rows.append({
                    "estimator": "waoss",
                    "pair": pair_label,
                    "estimate": scalars.get(f"delta_2_{p}_XX", np.nan),
                    "se": scalars.get(f"sd_delta_2_{p}_XX", np.nan),
                    "lb": scalars.get(f"LB_2_{p}_XX", np.nan),
                    "ub": scalars.get(f"UB_2_{p}_XX", np.nan),
                })
            if ivwaoss == 1:
                rows.append({
                    "estimator": "ivwaoss",
                    "pair": pair_label,
                    "estimate": scalars.get(f"delta_3_{p}_XX", np.nan),
                    "se": scalars.get(f"sd_delta_3_{p}_XX", np.nan),
                    "lb": scalars.get(f"LB_3_{p}_XX", np.nan),
                    "ub": scalars.get(f"UB_3_{p}_XX", np.nan),
                })

    table = pd.DataFrame(rows, columns=["estimator", "pair", "estimate", "se", "lb", "ub"])

    # Optional aoss_vs_waoss test (ID-cluster robust, t with df=G-1)
    test = None
    if aoss_vs_waoss and (aoss == 1) and (waoss == 1):
        # rebuild phi1/phi2 from helper (already computed above if aoss/waoss ran)
        # If not in scope (because not requested), compute now
        est1, _, _, _, phi1 = _overall_from_pairs(1)
        est2, _, _, _, phi2 = _overall_from_pairs(2)
        diff = est1 - est2

        phi_diff = phi1.reindex(phi2.index).fillna(0.0) - phi2.fillna(0.0)
        G = phi_diff.size
        se_diff = float(np.std(phi_diff.to_numpy(), ddof=1) / np.sqrt(G)) if G >= 2 else np.nan

        t_stat = diff / se_diff if np.isfinite(se_diff) and se_diff > 0 else np.nan
        df_t = G - 1

        if stats is not None and np.isfinite(t_stat) and df_t >= 1:
            p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df_t)))
        else:
            p_val = np.nan

        test = {"diff": diff, "se": se_diff, "t": t_stat, "df": df_t, "p_value": p_val}

    return {
        "table": table,
        "pairs": len(pairs),
        "time_levels": time_levels,
        "scalars": scalars,
        "aoss_vs_waoss_test": test,
    }


# =============================================================================
# Public wrapper (translation of did_multiplegt_stat.R you pasted)
# =============================================================================

def did_multiplegt_stat_py(
    df: pd.DataFrame,
    Y: str,
    ID: str,
    Time: str,
    D: str,
    Z: Optional[str] = None,
    estimator: Optional[Union[str, List[str]]] = None,
    estimation_method: Optional[str] = None,
    order: int = 1,
    noextrapolation: bool = False,
    placebo: bool = False,
    switchers: Optional[str] = None,
    disaggregate: bool = False,
    aoss_vs_waoss: bool = False,
    exact_match: bool = False,
    by: Optional[Union[str, List[str]]] = None,
    by_fd: Optional[int] = None,
    other_treatments: Optional[Union[str, List[str]]] = None,
    cluster: Optional[str] = None,
) -> DidMultiplegtStat:
    """
    Python translation of the R wrapper did_multiplegt_stat().

    The core ideas:
      - Validate and normalize options exactly like R.
      - Handle `by` (time-invariant grouping) and `by_fd` (quantile bins for |ΔD| or |ΔZ|).
      - Loop over by-levels and call did_multiplegt_stat_main_py().
      - Return an object with args + results (+ by/by_fd extras).

    IMPORTANT:
      - As in the R code you pasted, we set weight = None (option shut down).
      - For exact 1:1 replication of R, porting the exact did_multiplegt_stat_main.R
        is recommended. This wrapper is faithful; the internal main here is best-effort.
    """

    # -------------------------------------------------
    # Build args dict (like R does via match.call())
    # -------------------------------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Syntax error in df option. Dataframe required.")

    args: Dict[str, Any] = {
        "df": df, "Y": Y, "ID": ID, "Time": Time, "D": D, "Z": Z,
        "estimator": estimator, "estimation_method": estimation_method,
        "order": order, "noextrapolation": noextrapolation, "placebo": placebo,
        "switchers": switchers, "disaggregate": disaggregate, "aoss_vs_waoss": aoss_vs_waoss,
        "exact_match": exact_match, "by": by, "by_fd": by_fd,
        "other_treatments": other_treatments, "cluster": cluster,
    }

    # Shut down weight option (like R code comment)
    weight = None

    # -------------------------------------------------
    # Validate single-string options
    # -------------------------------------------------
    for vname, v in [("Y", Y), ("ID", ID), ("Time", Time), ("D", D)]:
        if not _is_single_string(v):
            raise TypeError(f"Syntax error in {vname} option. The option requires a single string.")

    if Z is not None and not _is_single_string(Z):
        raise TypeError("Syntax error in Z option. The option requires a single string.")

    if estimation_method is not None and not _is_single_string(estimation_method):
        raise TypeError("Syntax error in estimation_method option. The option requires a single string.")

    if switchers is not None and not _is_single_string(switchers):
        raise TypeError("Syntax error in switchers option. The option requires a single string.")

    if cluster is not None and not _is_single_string(cluster):
        raise TypeError("Syntax error in cluster option. The option requires a single string.")

    # Boolean options
    for vname, v in [
        ("noextrapolation", noextrapolation),
        ("placebo", placebo),
        ("disaggregate", disaggregate),
        ("aoss_vs_waoss", aoss_vs_waoss),
        ("exact_match", exact_match),
    ]:
        if not isinstance(v, bool):
            raise TypeError(f"Syntax error in {vname} option. Logical required.")

    # Integer options
    if not (isinstance(order, int) and order >= 1):
        raise TypeError("Syntax error in order option. Integer required (>=1).")

    if by_fd is not None:
        if not (isinstance(by_fd, int) and by_fd >= 1):
            raise TypeError("Syntax error in by_fd option. Integer required (>=1).")

    # by option
    by_list = _as_list_of_strings(by) if by is not None else None
    other_treat_list = _as_list_of_strings(other_treatments) if other_treatments is not None else None

    # -------------------------------------------------
    # Default estimator logic
    # -------------------------------------------------
    if estimator is None and Z is None:
        estimator_list = ["aoss", "waoss"]
    elif estimator is None and Z is not None:
        estimator_list = ["ivwaoss"]
    else:
        estimator_list = _as_list_of_strings(estimator)

    allowed = {"aoss", "waoss", "ivwaoss"}
    if any(e not in allowed for e in estimator_list):
        raise ValueError("Syntax error in estimator option: only aoss, waoss and ivwaoss allowed.")

    if switchers is not None and switchers not in {"up", "down"}:
        raise ValueError("Switchers could be either NULL/None, 'up' or 'down'.")

    # -------------------------------------------------
    # Default estimation_method logic
    # -------------------------------------------------
    if estimation_method is None:
        if exact_match is False:
            estimation_method = "dr"
        else:
            estimation_method = "ra"

    # If only aoss requested -> force ra
    if len(estimator_list) == 1 and estimator_list[0] == "aoss":
        estimation_method = "ra"

    # exact_match constraints
    if exact_match is True:
        if estimation_method != "ra":
            raise ValueError("The exact_match option is only compatible with regression adjustment estimation method.")
        if noextrapolation is True:
            warnings.warn("When exact_match and noextrapolation are both specified, only exact_match will be used.")
            noextrapolation = False
        if order != 1:
            raise ValueError("The order option is not compatible with exact_match.")
        order = 1

    if estimation_method not in {"ra", "dr", "ps"}:
        raise ValueError("Syntax error in estimation_method option.")

    if len(estimator_list) == 1 and estimator_list[0] == "aoss" and estimation_method in {"dr", "ps"}:
        raise ValueError("The propensity score-based approach is only available for waoss and ivwaoss.")

    # ivwaoss cannot be combined with aoss/waoss
    if ("ivwaoss" in estimator_list) and any(e in estimator_list for e in ["aoss", "waoss"]):
        raise ValueError("AOSS/WAOSS cannot be combined with IV-WAOSS (see helpfile).")

    if aoss_vs_waoss and (sum(e in estimator_list for e in ["aoss", "waoss"]) != 2):
        raise ValueError("To test equality between AOSS and WAOSS, specify both in estimator option.")

    if ("ivwaoss" in estimator_list) and (Z is None):
        raise ValueError("To compute ivwaoss you must specify the IV variable Z.")

    # Store normalized args
    args["estimator"] = estimator_list
    args["estimation_method"] = estimation_method
    args["order"] = order
    args["noextrapolation"] = noextrapolation

    # -------------------------------------------------
    # Handle by / by_fd
    # -------------------------------------------------
    df_work = df.copy()
    by_levels = ["_no_by"]
    quantiles_mat = None
    switch_df = None
    quantiles_plot = None

    by_fd_opt = None
    by_str = None

    if (by_list is not None and len(by_list) > 0) or (by_fd is not None):
        if by_list is not None and len(by_list) > 0:
            df_work["by_total"] = ""
            for j, v in enumerate(by_list, start=1):
                if not by_check(df_work, ID, v):
                    raise ValueError("The ID variable should be nested within the by variable.")
                if j == 1:
                    df_work["by_total"] = df_work[v].astype(str)
                else:
                    df_work["by_total"] = df_work["by_total"] + "," + df_work[v].astype(str)

            by_levels = sorted(df_work["by_total"].dropna().unique().tolist())
            by_str = ",".join(by_list)

        else:
            # by_fd case
            if 100 % by_fd != 0:
                raise ValueError(
                    "Syntax error in by option. When by_fd is integer, it must divide 100 "
                    "to allow for an integer subsetting of the quantiles."
                )

            q_levels = [0.0]
            for _ in range(by_fd):
                q_levels.append(q_levels[-1] + 1.0 / by_fd)

            qset = did_multiplegt_stat_quantiles_py(
                df=df_work,
                ID=ID, Time=Time, D=D, Z=Z,
                by_opt=by_fd,
                quantiles=q_levels,
                estimator=estimator_list,
            )
            df_work = qset["df"]
            val_quantiles = qset["val_quantiles"]
            quantiles = qset["quantiles"]
            switch_df = qset["switch_df"]
            quantiles_plot = qset["quantiles_plot"]

            # levels among switchers (partition_XX != 0)
            by_levels = sorted(
                df_work.loc[df_work["partition_XX"] != 0, "partition_XX"].dropna().unique().tolist()
            )

            quantiles_mat = np.vstack([quantiles, val_quantiles])

            if len(by_levels) != by_fd:
                warnings.warn(
                    f"Point mass > {100/by_fd:.0f}% detected. {by_fd - len(by_levels):.0f} bin(s) collapsed."
                )

    # -------------------------------------------------
    # Run main for each by level
    # -------------------------------------------------
    results_dict: Dict[str, Any] = {}

    df_main = df_work
    for i, lev in enumerate(by_levels, start=1):
        obj_name = "results"
        by_fd_opt = None

        if lev != "_no_by" and by_list is not None and len(by_list) > 0:
            df_main = df_work[df_work["by_total"] == lev].copy()
            obj_name = f"results_by_{i}"
            print(f"Running did_multiplegt_stat with {by_str} = {lev}")

        elif lev != "_no_by" and by_fd is not None:
            obj_name = f"results_by_{i}"
            diff_var = "Z" if ("ivwaoss" in estimator_list) else "D"

            # For display messages we need val_quantiles; if by_fd, we created them above
            # When bins collapse, val_quantiles may have repeats
            # lev is the partition id
            k = int(lev)
            sep = "[" if i == 1 else "("
            lo = float(quantiles_mat[1, k - 1]) if quantiles_mat is not None else np.nan
            hi = float(quantiles_mat[1, k]) if quantiles_mat is not None else np.nan
            qlo = float(quantiles_mat[0, k - 1]) * 100 if quantiles_mat is not None else np.nan
            qhi = float(quantiles_mat[0, k]) * 100 if quantiles_mat is not None else np.nan

            print(f"Running did_multiplegt_stat with switchers s.t. Δ{diff_var} ∈ {sep}{lo:.3f},{hi:.3f}] <{qlo:.0f}%-{qhi:.0f}% quantiles>.")

            if np.isfinite(lo) and np.isfinite(hi) and lo == hi:
                warnings.warn(f"({qlo:.0f}%, {qhi:.0f}%) quantile bin dropped: upper and lower bounds are equal.")

            by_fd_opt = lev  # this is what pairwise filters on

        # Call main
        res = did_multiplegt_stat_main_py(
            df=df_main,
            Y=Y, ID=ID, Time=Time, D=D, Z=Z,
            estimator=estimator_list,
            estimation_method=estimation_method,
            order=order,
            noextrapolation=noextrapolation,
            placebo=placebo,
            weight=weight,
            switchers=switchers,
            disaggregate=disaggregate,
            aoss_vs_waoss=aoss_vs_waoss,
            exact_match=exact_match,
            cluster=cluster,
            by_fd_opt=by_fd_opt,
            other_treatments=other_treat_list,
        )
        results_dict[obj_name] = res

    # Optional graphs (placeholders)
    out_obj = DidMultiplegtStat(
        args=args,
        by_levels=by_levels,
        results=results_dict,
        quantiles=quantiles_mat,
        switchers_df=switch_df,
        cdf_plot=quantiles_plot,
        by_graph=None,
        by_fd_graph=None,
    )

    if by_list is not None and len(by_list) > 0:
        out_obj.by_graph = by_graph(out_obj)
    if quantiles_mat is not None:
        out_obj.by_fd_graph = by_fd_graph(out_obj)

    return out_obj

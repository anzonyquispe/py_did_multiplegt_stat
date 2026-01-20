from __future__ import annotations

# =============================================================================
# Python translation of did_multiplegt_stat_main.R (the "main" aggregation routine)
# This code is designed to work with your existing did_multiplegt_stat_pairwise_py()
# (the pairwise routine you posted). Comments are in English, as you requested.
# =============================================================================

from typing import Optional, Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from did_multiplegt_stat_pairwise import did_multiplegt_stat_pairwise_py



# -----------------------------------------------------------------------------
# Helper: balance a panel by filling missing (ID, Time) cells (like plm::make.pbalanced(..., fill))
# -----------------------------------------------------------------------------
def balance_panel_fill(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
) -> pd.DataFrame:
    """
    Create a balanced (ID x Time) panel by inserting missing rows.
    New rows will have NaNs for all non-index columns (matching R's 'fill').
    """
    # Collect unique IDs and Time values
    ids = df[id_col].dropna().unique()
    times = np.sort(df[time_col].dropna().unique())

    # Build the full cartesian product index (ID x Time)
    full_index = pd.MultiIndex.from_product([ids, times], names=[id_col, time_col])

    # Reindex to full grid; missing combinations become NaN rows
    out = (
        df.set_index([id_col, time_col])
          .reindex(full_index)
          .reset_index()
    )
    return out


# -----------------------------------------------------------------------------
# Helper: remap time values to consecutive integers 1..T in sorted order
# (like dplyr::cur_group_id() after grouping by the original time)
# -----------------------------------------------------------------------------
def remap_time_to_consecutive(
    df: pd.DataFrame,
    time_col: str,
) -> Tuple[pd.DataFrame, Dict[Any, int]]:
    """
    Replace df[time_col] values by 1..K (sorted by original time).
    Returns (df, mapping).
    """
    # Extract sorted unique time values
    uniq_t = np.sort(df[time_col].dropna().unique())

    # Map each unique time to an integer 1..K
    mapping = {t: i + 1 for i, t in enumerate(uniq_t)}

    # Apply mapping (keep NaNs as NaN)
    df[time_col] = df[time_col].map(mapping)

    return df, mapping


# -----------------------------------------------------------------------------
# Helper: safe sample sd with ddof=1 (matches R's stats::sd behavior)
# -----------------------------------------------------------------------------
def sd_sample(x: Union[pd.Series, np.ndarray]) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return float(np.std(x, ddof=1))


# =============================================================================
# Main routine: did_multiplegt_stat_main_py
# =============================================================================
def did_multiplegt_stat_main_py(
    df: pd.DataFrame,
    *,
    Y: str,
    ID: str,
    Time: str,
    D: str,
    Z: Optional[str],
    estimator: Union[str, List[str], Tuple[str, ...], set],
    estimation_method: str,           # "ra" / "ps" / "dr"
    order: int,
    noextrapolation: bool,
    placebo: bool,
    switchers: Optional[str],         # None / "up" / "down"
    disaggregate: Optional[Any],      # kept for signature parity; not used in your R snippet
    aoss_vs_waoss: bool,
    exact_match: bool,
    weight: Optional[str],
    cluster: Optional[str],
    by_fd_opt: Optional[Any],
    other_treatments: Optional[List[str]],
) -> Dict[str, Any]:
    """
    Python translation of did_multiplegt_stat_main().

    It:
      1) keeps only needed columns for speed,
      2) creates internal *_XX columns,
      3) checks cluster nesting,
      4) drops missing mandatory values,
      5) balances the panel (fills missing ID-time rows),
      6) loops over pairwise indices and calls did_multiplegt_stat_pairwise_py(),
      7) aggregates pairwise deltas into overall estimands,
      8) builds influence functions and standard errors,
      9) returns a display-ready table (and optionally placebo table, AOSS vs WAOSS test, #clusters).

    IMPORTANT:
    - This function assumes you already have did_multiplegt_stat_pairwise_py available in scope.
    """

    # -------------------------
    # 0) Convert estimator argument into a set for easy membership checks
    # -------------------------
    if isinstance(estimator, str):
        estimator_set = {estimator}
    else:
        estimator_set = set(estimator)

    # -------------------------
    # 1) Preallocation of estimator flags (aoss_XX / waoss_XX / ivwaoss_XX)
    # -------------------------
    aoss_XX = 1 if "aoss" in estimator_set else 0
    waoss_XX = 1 if "waoss" in estimator_set else 0
    ivwaoss_XX = 1 if "ivwaoss" in estimator_set else 0

    # -------------------------
    # 2) Keep only variables of interest (speed layer)
    # -------------------------
    other_treatments = other_treatments or []

    # Build the variable list (unique, preserving order)
    varlist: List[str] = []
    for v in [Y, ID, Time, D, Z, weight, cluster, *other_treatments]:
        if v is None:
            continue
        if v not in varlist:
            varlist.append(v)

    # If partition_XX exists in df, keep it (R does this)
    if "partition_XX" in df.columns and "partition_XX" not in varlist:
        varlist.append("partition_XX")

    # Subset df to only varlist columns
    df = df[varlist].copy()

    # -------------------------
    # 3) Create internal copies with *_XX names
    # (R: df[[paste0(names(df_base)[i],"_XX")]] <- df[[col]])
    # -------------------------
    # Create internal canonical columns (these are the ones the algorithm uses)
    df["Y_XX"] = df[Y]
    df["ID_XX"] = df[ID]
    df["T_XX"] = df[Time]
    df["D_XX"] = df[D]

    # Instrument column Z_XX only matters if ivwaoss is requested
    if ivwaoss_XX == 1:
        if Z is None:
            raise ValueError("ivwaoss requested but Z is None.")
        df["Z_XX"] = df[Z]
    else:
        df["Z_XX"] = np.nan

    # Weight column: if missing, set later
    if weight is not None:
        df["weight_XX"] = df[weight]
    else:
        df["weight_XX"] = np.nan  # we overwrite with 1 below

    # Cluster column: if provided, copy
    if cluster is not None:
        df["cluster_XX"] = df[cluster]
    else:
        df["cluster_XX"] = np.nan

    # -------------------------
    # 4) Cluster checks: cluster must be coarser than ID, and ID must be nested in cluster
    # -------------------------
    n_clus_XX = None  # will be set only if clustering is valid

    if cluster is not None:
        # If cluster == ID, R ignores cluster and prints a message
        if cluster == ID:
            cluster = None
            df["cluster_XX"] = np.nan
            print("The cluster option should be different from (and coarser than) the ID variable. "
                  "The command will ignore the cluster option.")
        else:
            # Check nesting: each ID must belong to a single cluster
            # R uses sd(cluster_XX) by ID; we use nunique (works for numeric or string)
            clus_nuniq = df.groupby("ID_XX")["cluster_XX"].nunique(dropna=True)
            if (clus_nuniq > 1).any():
                raise ValueError("The ID variable should be nested within the clustering variable.")
            else:
                n_clus_XX = int(df["cluster_XX"].nunique(dropna=True))

    # -------------------------
    # 5) Drop observations with missing mandatory fields
    # (R: to_drop_XX = is.na(T) | is.na(D) | is.na(ID), plus Z if IV)
    # -------------------------
    to_drop = df["T_XX"].isna() | df["D_XX"].isna() | df["ID_XX"].isna()

    IV_req_XX = 0
    if ivwaoss_XX == 1:
        to_drop = to_drop | df["Z_XX"].isna()
        IV_req_XX = 1

    df = df.loc[~to_drop].copy()

    # If nothing remains, return empty-ish output
    if df.shape[0] == 0:
        return {"table": pd.DataFrame(), "pairs": 0}

    # -------------------------
    # 6) Balance the panel and create tsfilled_XX (like plm::make.pbalanced(..., fill))
    # -------------------------
    # R sets tsfilled_XX=0 before balancing, then flags filled rows with is.na(tsfilled_XX)
    df["tsfilled_XX"] = 0.0

    # Fill missing (ID_XX, T_XX) combinations
    df = balance_panel_fill(df, id_col="ID_XX", time_col="T_XX")

    # Mark which rows were newly created by the balancing step
    # (original rows had tsfilled_XX=0, filled rows have tsfilled_XX=NaN after reindex)
    df["tsfilled_XX"] = df["tsfilled_XX"].isna().astype(float)

    # -------------------------
    # 7) Remap time to consecutive 1..T (R: group_by(T_temp) mutate(T_XX=cur_group_id()))
    # -------------------------
    df, _time_map = remap_time_to_consecutive(df, time_col="T_XX")

    # -------------------------
    # 8) Weights: if no weight option, set weight_XX=1 and weight_c_XX=1
    # If weights exist, NA weights become 0 (R behavior)
    # -------------------------
    if weight is None:
        df["weight_XX"] = 1.0
        df["weight_c_XX"] = 1.0
    else:
        df["weight_XX"] = df["weight_XX"].astype(float)
        df["weight_XX"] = df["weight_XX"].fillna(0.0)

        # Cluster-time summed weights (R: group_by(cluster, T) sum(weight))
        if cluster is not None:
            df["weight_c_XX"] = (
                df.groupby(["cluster_XX", "T_XX"])["weight_XX"]
                  .transform(lambda s: float(np.nansum(s.to_numpy(dtype=float))))
            )
        else:
            df["weight_c_XX"] = 1.0

    # -------------------------
    # 9) IDs_XX data frame: one row per ID (and cluster if any), used to store IFs
    # -------------------------
    IDs_XX = pd.DataFrame({"ID_XX": np.sort(df["ID_XX"].dropna().unique())})

    if cluster is not None:
        # Each ID has one cluster (we already checked nesting), so take first/mean
        clus_by_id = df.groupby("ID_XX")["cluster_XX"].first().reset_index()
        IDs_XX = IDs_XX.merge(clus_by_id, on="ID_XX", how="left")

    # -------------------------
    # 10) Scalars initialization (matches the R list `scalars <- list(...)`)
    # -------------------------
    max_T_XX = int(np.nanmax(df["T_XX"].to_numpy(dtype=float)))

    scalars: Dict[str, Any] = {
        "PS_sum_XX": 0.0,
        "delta_1_1_XX": 0.0,
        "E_abs_delta_D_sum_XX": 0.0,
        "delta_2_1_XX": 0.0,
        "denom_delta_IV_sum_XX": 0.0,
        "delta_3_1_XX": 0.0,

        "N_Switchers_1_1_XX": 0.0,
        "N_Stayers_1_1_XX": 0.0,
        "N_Switchers_2_1_XX": 0.0,
        "N_Stayers_2_1_XX": 0.0,
        "N_Switchers_3_1_XX": 0.0,
        "N_Stayers_3_1_XX": 0.0,

        "N_drop_total_XX": 0.0,
        "N_drop_total_C_XX": 0.0,
        "IV_req_XX": IV_req_XX,
    }

    # Placebo scalars (R adds these when placebo==TRUE)
    if placebo:
        scalars.update({
            "PS_sum_pl_XX": 0.0,
            "delta_1_1_pl_XX": 0.0,
            "E_abs_delta_D_sum_pl_XX": 0.0,
            "delta_2_1_pl_XX": 0.0,
            "denom_delta_IV_sum_pl_XX": 0.0,
            "delta_3_1_pl_XX": 0.0,

            "N_Switchers_1_1_pl_XX": 0.0,
            "N_Stayers_1_1_pl_XX": 0.0,
            "N_Switchers_2_1_pl_XX": 0.0,
            "N_Stayers_2_1_pl_XX": 0.0,
            "N_Switchers_3_1_pl_XX": 0.0,
            "N_Stayers_3_1_pl_XX": 0.0,
        })

    # -------------------------
    # 11) Main loop over pairwise indices p = 2..max_T_XX
    # Calls your did_multiplegt_stat_pairwise_py() and merges results into IDs_XX.
    # -------------------------
    for p in range(2, max_T_XX + 1):

        # IMPORTANT LINE:
        # In your R snippet the call is Y="Y_ID". If your pipeline truly creates Y_ID,
        # replace Y="Y_XX" by Y="Y_ID" here.
        est_out = did_multiplegt_stat_pairwise_py(
            df,
            Y="Y_XX",
            ID="ID_XX",
            Time="T_XX",
            D="D_XX",
            Z="Z_XX",
            estimator=estimator,
            order=order,
            noextrapolation=noextrapolation,
            weight="weight_XX",
            switchers=switchers,
            pairwise=p,
            IDs=None,
            aoss=aoss_XX,
            waoss=waoss_XX,
            ivwaoss=ivwaoss_XX,
            estimation_method=estimation_method,
            scalars=scalars,
            placebo=False,
            exact_match=exact_match,
            cluster=("cluster_XX" if cluster is not None else None),
            by_fd_opt=by_fd_opt,
            other_treatments=other_treatments,
        )

        # Merge the pairwise ID-level additions (Phi_*, S_*, etc.) into IDs_XX
        if est_out.to_add is not None:
            IDs_XX = IDs_XX.merge(est_out.to_add, on="ID_XX", how="left")

        # Keep IDs sorted
        IDs_XX = IDs_XX.sort_values("ID_XX").reset_index(drop=True)

        # Update scalars from pairwise output
        scalars = est_out.scalars

        # -------------------------
        # Accumulate the overall (p=1) estimand numerators and counts
        # (these reproduce the R updates inside the loop)
        # -------------------------
        if aoss_XX == 1:
            # delta_1_1 += P_p * delta_1_p
            scalars["delta_1_1_XX"] += scalars.get(f"P_{p}_XX", 0.0) * scalars.get(f"delta_1_{p}_XX", 0.0)

            # Accumulate switchers/stayers totals only if enough stayers/switchers exist
            n_stayers = scalars.get(f"N_Stayers_1_{p}_XX", np.nan)
            n_switch = scalars.get(f"N_Switchers_1_{p}_XX", np.nan)
            if np.isfinite(n_stayers) and n_stayers > 1:
                scalars["N_Switchers_1_1_XX"] += (n_switch if np.isfinite(n_switch) else 0.0)
            if np.isfinite(n_switch) and n_switch > 0:
                scalars["N_Stayers_1_1_XX"] += (n_stayers if np.isfinite(n_stayers) else 0.0)

        if waoss_XX == 1:
            # delta_2_1 += E_abs_delta_D_p * delta_2_p
            scalars["delta_2_1_XX"] += scalars.get(f"E_abs_delta_D_{p}_XX", 0.0) * scalars.get(f"delta_2_{p}_XX", 0.0)

            n_stayers = scalars.get(f"N_Stayers_2_{p}_XX", np.nan)
            n_switch = scalars.get(f"N_Switchers_2_{p}_XX", np.nan)
            if np.isfinite(n_stayers) and n_stayers > 1:
                scalars["N_Switchers_2_1_XX"] += (n_switch if np.isfinite(n_switch) else 0.0)
            if np.isfinite(n_switch) and n_switch > 0:
                scalars["N_Stayers_2_1_XX"] += (n_stayers if np.isfinite(n_stayers) else 0.0)

        if ivwaoss_XX == 1:
            # delta_3_1 += denom_delta_IV_p * delta_3_p
            scalars["delta_3_1_XX"] += scalars.get(f"denom_delta_IV_{p}_XX", 0.0) * scalars.get(f"delta_3_{p}_XX", 0.0)

            n_stayers = scalars.get(f"N_Stayers_3_{p}_XX", np.nan)
            n_switch = scalars.get(f"N_Switchers_3_{p}_XX", np.nan)
            if np.isfinite(n_stayers) and n_stayers > 1:
                scalars["N_Switchers_3_1_XX"] += (n_switch if np.isfinite(n_switch) else 0.0)
            if np.isfinite(n_switch) and n_switch > 0:
                scalars["N_Stayers_3_1_XX"] += (n_stayers if np.isfinite(n_stayers) else 0.0)

    # -------------------------
    # 12) Placebo loop (p = 3..max_T_XX) if placebo==True
    # -------------------------
    if placebo:
        for p in range(3, max_T_XX + 1):

            est_out = did_multiplegt_stat_pairwise_py(
                df,
                Y="Y_XX",
                ID="ID_XX",
                Time="T_XX",
                D="D_XX",
                Z="Z_XX",
                estimator=estimator,
                order=order,
                noextrapolation=noextrapolation,
                weight="weight_XX",
                switchers=switchers,
                pairwise=p,
                IDs=None,
                aoss=aoss_XX,
                waoss=waoss_XX,
                ivwaoss=ivwaoss_XX,
                estimation_method=estimation_method,
                scalars=scalars,
                placebo=True,
                exact_match=exact_match,
                cluster=("cluster_XX" if cluster is not None else None),
                by_fd_opt=by_fd_opt,
                other_treatments=other_treatments,
            )

            if est_out.to_add is not None:
                IDs_XX = IDs_XX.merge(est_out.to_add, on="ID_XX", how="left")

            IDs_XX = IDs_XX.sort_values("ID_XX").reset_index(drop=True)
            scalars = est_out.scalars

            # Accumulate placebo aggregated numerators
            if aoss_XX == 1:
                scalars["delta_1_1_pl_XX"] += scalars.get(f"P_{p}_pl_XX", 0.0) * scalars.get(f"delta_1_{p}_pl_XX", 0.0)

                n_stayers = scalars.get(f"N_Stayers_1_{p}_pl_XX", np.nan)
                n_switch = scalars.get(f"N_Switchers_1_{p}_pl_XX", np.nan)
                if np.isfinite(n_stayers) and n_stayers > 1:
                    scalars["N_Switchers_1_1_pl_XX"] += (n_switch if np.isfinite(n_switch) else 0.0)
                if np.isfinite(n_switch) and n_switch > 0:
                    scalars["N_Stayers_1_1_pl_XX"] += (n_stayers if np.isfinite(n_stayers) else 0.0)

            if waoss_XX == 1:
                scalars["delta_2_1_pl_XX"] += scalars.get(f"E_abs_delta_D_{p}_pl_XX", 0.0) * scalars.get(f"delta_2_{p}_pl_XX", 0.0)

                n_stayers = scalars.get(f"N_Stayers_2_{p}_pl_XX", np.nan)
                n_switch = scalars.get(f"N_Switchers_2_{p}_pl_XX", np.nan)
                if np.isfinite(n_stayers) and n_stayers > 1:
                    scalars["N_Switchers_2_1_pl_XX"] += (n_switch if np.isfinite(n_switch) else 0.0)
                if np.isfinite(n_switch) and n_switch > 0:
                    scalars["N_Stayers_2_1_pl_XX"] += (n_stayers if np.isfinite(n_stayers) else 0.0)

            if ivwaoss_XX == 1:
                scalars["delta_3_1_pl_XX"] += scalars.get(f"denom_delta_IV_{p}_pl_XX", 0.0) * scalars.get(f"delta_3_{p}_pl_XX", 0.0)

                n_stayers = scalars.get(f"N_Stayers_3_{p}_pl_XX", np.nan)
                n_switch = scalars.get(f"N_Switchers_3_{p}_pl_XX", np.nan)
                if np.isfinite(n_stayers) and n_stayers > 1:
                    scalars["N_Switchers_3_1_pl_XX"] += (n_switch if np.isfinite(n_switch) else 0.0)
                if np.isfinite(n_switch) and n_switch > 0:
                    scalars["N_Stayers_3_1_pl_XX"] += (n_stayers if np.isfinite(n_stayers) else 0.0)

    # -------------------------
    # 13) Compute aggregated estimators (divide by accumulated denominators)
    # -------------------------
    if aoss_XX == 1:
        scalars["delta_1_1_XX"] = scalars["delta_1_1_XX"] / scalars.get("PS_sum_XX", np.nan)
        if placebo:
            scalars["delta_1_1_pl_XX"] = scalars["delta_1_1_pl_XX"] / scalars.get("PS_sum_pl_XX", np.nan)

    if waoss_XX == 1:
        scalars["delta_2_1_XX"] = scalars["delta_2_1_XX"] / scalars.get("E_abs_delta_D_sum_XX", np.nan)
        if placebo:
            scalars["delta_2_1_pl_XX"] = scalars["delta_2_1_pl_XX"] / scalars.get("E_abs_delta_D_sum_pl_XX", np.nan)

    if ivwaoss_XX == 1:
        scalars["delta_3_1_XX"] = scalars["delta_3_1_XX"] / scalars.get("denom_delta_IV_sum_XX", np.nan)
        if placebo:
            scalars["delta_3_1_pl_XX"] = scalars["delta_3_1_pl_XX"] / scalars.get("denom_delta_IV_sum_pl_XX", np.nan)

    # -------------------------
    # 14) Compute influence functions for the aggregated estimators (Phi_i_XX)
    # -------------------------
    # Initialize aggregated IF containers
    for i in [1, 2, 3]:
        IDs_XX[f"Phi_{i}_XX"] = 0.0
        IDs_XX[f"Phi_{i}_pl_XX"] = 0.0

    # Loop over pairwise periods and add contributions
    for p in range(2, max_T_XX + 1):

        # AOSS IF aggregation
        if aoss_XX == 1 and scalars.get(f"non_missing_{p}_XX", 0) == 1:
            Pp = scalars.get(f"P_{p}_XX", np.nan)
            d1p = scalars.get(f"delta_1_{p}_XX", np.nan)
            d1 = scalars.get("delta_1_1_XX", np.nan)
            PSsum = scalars.get("PS_sum_XX", np.nan)

            # Adjusted pairwise IF: (Pp*Phi_p + (d_p - d_agg)*(S_p - Pp)) / PSsum
            col_phi = f"Phi_1_{p}_XX"
            col_S = f"S_{p}_XX"
            if col_phi in IDs_XX.columns and col_S in IDs_XX.columns and np.isfinite(PSsum) and PSsum != 0:
                phi_adj = (Pp * IDs_XX[col_phi] + (d1p - d1) * (IDs_XX[col_S] - Pp)) / PSsum
                IDs_XX[col_phi] = phi_adj
                IDs_XX.loc[phi_adj.notna(), "Phi_1_XX"] += phi_adj[phi_adj.notna()]

        # Placebo AOSS IF aggregation (p>2 in R)
        if placebo and aoss_XX == 1 and p > 2 and scalars.get(f"non_missing_{p}_pl_XX", 0) == 1:
            Pp = scalars.get(f"P_{p}_pl_XX", np.nan)
            d1p = scalars.get(f"delta_1_{p}_pl_XX", np.nan)
            d1 = scalars.get("delta_1_1_pl_XX", np.nan)
            PSsum = scalars.get("PS_sum_pl_XX", np.nan)

            col_phi = f"Phi_1_{p}_pl_XX"
            col_S = f"S_{p}_pl_XX"
            if col_phi in IDs_XX.columns and col_S in IDs_XX.columns and np.isfinite(PSsum) and PSsum != 0:
                phi_adj = (Pp * IDs_XX[col_phi] + (d1p - d1) * (IDs_XX[col_S] - Pp)) / PSsum
                IDs_XX[col_phi] = phi_adj
                IDs_XX.loc[phi_adj.notna(), "Phi_1_pl_XX"] += phi_adj[phi_adj.notna()]

        # WAOSS IF aggregation
        if waoss_XX == 1 and scalars.get(f"non_missing_{p}_XX", 0) == 1:
            Ep = scalars.get(f"E_abs_delta_D_{p}_XX", np.nan)
            d2p = scalars.get(f"delta_2_{p}_XX", np.nan)
            d2 = scalars.get("delta_2_1_XX", np.nan)
            Esum = scalars.get("E_abs_delta_D_sum_XX", np.nan)

            col_phi = f"Phi_2_{p}_XX"
            col_abs = f"abs_delta_D_{p}_XX"
            if col_phi in IDs_XX.columns and col_abs in IDs_XX.columns and np.isfinite(Esum) and Esum != 0:
                phi_adj = (Ep * IDs_XX[col_phi] + (d2p - d2) * (IDs_XX[col_abs] - Ep)) / Esum
                IDs_XX[col_phi] = phi_adj
                IDs_XX.loc[phi_adj.notna(), "Phi_2_XX"] += phi_adj[phi_adj.notna()]

        # Placebo WAOSS IF aggregation
        if placebo and waoss_XX == 1 and p > 2 and scalars.get(f"non_missing_{p}_pl_XX", 0) == 1:
            Ep = scalars.get(f"E_abs_delta_D_{p}_pl_XX", np.nan)
            d2p = scalars.get(f"delta_2_{p}_pl_XX", np.nan)
            d2 = scalars.get("delta_2_1_pl_XX", np.nan)
            Esum = scalars.get("E_abs_delta_D_sum_pl_XX", np.nan)

            col_phi = f"Phi_2_{p}_pl_XX"
            col_abs = f"abs_delta_D_{p}_pl_XX"
            if col_phi in IDs_XX.columns and col_abs in IDs_XX.columns and np.isfinite(Esum) and Esum != 0:
                phi_adj = (Ep * IDs_XX[col_phi] + (d2p - d2) * (IDs_XX[col_abs] - Ep)) / Esum
                IDs_XX[col_phi] = phi_adj
                IDs_XX.loc[phi_adj.notna(), "Phi_2_pl_XX"] += phi_adj[phi_adj.notna()]

        # IVWAOSS IF aggregation
        if ivwaoss_XX == 1 and scalars.get(f"non_missing_{p}_XX", 0) == 1:
            Dp = scalars.get(f"denom_delta_IV_{p}_XX", np.nan)
            d3p = scalars.get(f"delta_3_{p}_XX", np.nan)
            d3 = scalars.get("delta_3_1_XX", np.nan)
            Dsum = scalars.get("denom_delta_IV_sum_XX", np.nan)

            col_phi = f"Phi_3_{p}_XX"
            col_in = f"inner_sum_IV_denom_{p}_XX"
            if col_phi in IDs_XX.columns and col_in in IDs_XX.columns and np.isfinite(Dsum) and Dsum != 0:
                phi_adj = (Dp * IDs_XX[col_phi] + (d3p - d3) * (IDs_XX[col_in] - Dp)) / Dsum
                IDs_XX[col_phi] = phi_adj
                IDs_XX.loc[phi_adj.notna(), "Phi_3_XX"] += phi_adj[phi_adj.notna()]

        # Placebo IVWAOSS IF aggregation
        if placebo and ivwaoss_XX == 1 and p > 2 and scalars.get(f"non_missing_{p}_pl_XX", 0) == 1:
            Dp = scalars.get(f"denom_delta_IV_{p}_pl_XX", np.nan)
            d3p = scalars.get(f"delta_3_{p}_pl_XX", np.nan)
            d3 = scalars.get("delta_3_1_pl_XX", np.nan)
            Dsum = scalars.get("denom_delta_IV_sum_pl_XX", np.nan)

            col_phi = f"Phi_3_{p}_pl_XX"
            col_in = f"inner_sum_IV_denom_{p}_pl_XX"
            if col_phi in IDs_XX.columns and col_in in IDs_XX.columns and np.isfinite(Dsum) and Dsum != 0:
                phi_adj = (Dp * IDs_XX[col_phi] + (d3p - d3) * (IDs_XX[col_in] - Dp)) / Dsum
                IDs_XX[col_phi] = phi_adj
                IDs_XX.loc[phi_adj.notna(), "Phi_3_pl_XX"] += phi_adj[phi_adj.notna()]

    # -------------------------
    # 15) Compute standard errors for the aggregated estimators using influence functions
    # -------------------------
    # If clustering: compute average cluster size N_bar_c, then sum IF within cluster and scale
    if cluster is not None:
        ids_per_cluster = IDs_XX.groupby("cluster_XX")["ID_XX"].nunique()
        N_bar_c_XX = float(ids_per_cluster.mean()) if len(ids_per_cluster) else np.nan
    else:
        N_bar_c_XX = np.nan

    def se_from_if(phi: pd.Series, cluster_col: Optional[str]) -> float:
        """
        Compute SE = sd(IF)/sqrt(n_obs), with cluster aggregation if cluster_col is not None.
        This mirrors the R logic: sum IF within cluster then divide by N_bar_c.
        """
        if cluster_col is not None:
            # Sum IF over clusters (one number per cluster)
            phi_c = IDs_XX.groupby(cluster_col)[phi.name].sum(min_count=1)
            # Scale by average cluster size (as in the R code)
            if np.isfinite(N_bar_c_XX) and N_bar_c_XX != 0:
                phi_c = phi_c / N_bar_c_XX
            x = phi_c.to_numpy(dtype=float)
            n = np.sum(np.isfinite(x))
            return sd_sample(x) / np.sqrt(n) if n > 0 else np.nan
        else:
            x = phi.to_numpy(dtype=float)
            n = np.sum(np.isfinite(x))
            return sd_sample(x) / np.sqrt(n) if n > 0 else np.nan

    # AOSS aggregated SE + CI
    if aoss_XX == 1:
        scalars["sd_delta_1_1_XX"] = se_from_if(IDs_XX["Phi_1_XX"], "cluster_XX" if cluster is not None else None)
        scalars["LB_1_1_XX"] = scalars["delta_1_1_XX"] - 1.96 * scalars["sd_delta_1_1_XX"]
        scalars["UB_1_1_XX"] = scalars["delta_1_1_XX"] + 1.96 * scalars["sd_delta_1_1_XX"]

        if placebo:
            scalars["sd_delta_1_1_pl_XX"] = se_from_if(IDs_XX["Phi_1_pl_XX"], "cluster_XX" if cluster is not None else None)
            scalars["LB_1_1_pl_XX"] = scalars["delta_1_1_pl_XX"] - 1.96 * scalars["sd_delta_1_1_pl_XX"]
            scalars["UB_1_1_pl_XX"] = scalars["delta_1_1_pl_XX"] + 1.96 * scalars["sd_delta_1_1_pl_XX"]

    # WAOSS aggregated SE + CI
    if waoss_XX == 1:
        scalars["sd_delta_2_1_XX"] = se_from_if(IDs_XX["Phi_2_XX"], "cluster_XX" if cluster is not None else None)
        scalars["LB_2_1_XX"] = scalars["delta_2_1_XX"] - 1.96 * scalars["sd_delta_2_1_XX"]
        scalars["UB_2_1_XX"] = scalars["delta_2_1_XX"] + 1.96 * scalars["sd_delta_2_1_XX"]

        if placebo:
            scalars["sd_delta_2_1_pl_XX"] = se_from_if(IDs_XX["Phi_2_pl_XX"], "cluster_XX" if cluster is not None else None)
            scalars["LB_2_1_pl_XX"] = scalars["delta_2_1_pl_XX"] - 1.96 * scalars["sd_delta_2_1_pl_XX"]
            scalars["UB_2_1_pl_XX"] = scalars["delta_2_1_pl_XX"] + 1.96 * scalars["sd_delta_2_1_pl_XX"]

    # IVWAOSS aggregated SE + CI
    if ivwaoss_XX == 1:
        scalars["sd_delta_3_1_XX"] = se_from_if(IDs_XX["Phi_3_XX"], "cluster_XX" if cluster is not None else None)
        scalars["LB_3_1_XX"] = scalars["delta_3_1_XX"] - 1.96 * scalars["sd_delta_3_1_XX"]
        scalars["UB_3_1_XX"] = scalars["delta_3_1_XX"] + 1.96 * scalars["sd_delta_3_1_XX"]

        if placebo:
            scalars["sd_delta_3_1_pl_XX"] = se_from_if(IDs_XX["Phi_3_pl_XX"], "cluster_XX" if cluster is not None else None)
            scalars["LB_3_1_pl_XX"] = scalars["delta_3_1_pl_XX"] - 1.96 * scalars["sd_delta_3_1_pl_XX"]
            scalars["UB_3_1_pl_XX"] = scalars["delta_3_1_pl_XX"] + 1.96 * scalars["sd_delta_3_1_pl_XX"]

    # -------------------------
    # 16) AOSS vs WAOSS test (optional)
    # -------------------------
    aoss_vs_waoss_table = None
    if aoss_vs_waoss and aoss_XX == 1 and waoss_XX == 1:
        diff = scalars["delta_1_1_XX"] - scalars["delta_2_1_XX"]
        IDs_XX["diff_Phi_1_2_XX"] = IDs_XX["Phi_1_XX"] - IDs_XX["Phi_2_XX"]

        if cluster is not None:
            # Cluster-level IF sums
            diff_c = IDs_XX.groupby("cluster_XX")["diff_Phi_1_2_XX"].sum(min_count=1)
            x = diff_c.to_numpy(dtype=float)
            se = sd_sample(x) / np.sqrt(np.sum(np.isfinite(x)))
            n_eff = int(np.sum(np.isfinite(x)))
        else:
            x = IDs_XX["diff_Phi_1_2_XX"].to_numpy(dtype=float)
            se = sd_sample(x) / np.sqrt(np.sum(np.isfinite(x)))
            n_eff = int(np.sum(np.isfinite(x)))

        t_stat = diff / se if (np.isfinite(se) and se != 0) else np.nan

        # R uses normal CDF; here we try to use Student-t if scipy is available (more Stata-like when clustering)
        try:
            from scipy import stats as st
            df_dof = max(n_eff - 1, 1)
            pval = 2 * (1 - st.t.cdf(np.abs(t_stat), df=df_dof)) if np.isfinite(t_stat) else np.nan
            crit = st.t.ppf(0.975, df=df_dof)
        except Exception:
            # Fallback: normal approximation
            from math import erf, sqrt
            def norm_cdf(z):  # standard normal CDF
                return 0.5 * (1.0 + erf(z / sqrt(2.0)))
            pval = 2 * (1 - norm_cdf(np.abs(t_stat))) if np.isfinite(t_stat) else np.nan
            crit = 1.96

        LB = diff - crit * se
        UB = diff + crit * se

        aoss_vs_waoss_table = pd.DataFrame(
            [[diff, se, LB, UB, t_stat, pval]],
            index=["Diff."],
            columns=["Estimate", "SE", "LB CI", "UB CI", "t stat.", "pval."],
        )

    # -------------------------
    # 17) Build the returned display table(s) (like ret_mat_XX in R)
    # -------------------------
    estims = ["aoss", "waoss", "ivwaoss"]
    flags = {"aoss": aoss_XX, "waoss": waoss_XX, "ivwaoss": ivwaoss_XX}

    # Prepare row names
    rown: List[str] = []
    for j, name in enumerate(estims, start=1):
        for p in range(1, max_T_XX + 1):
            if flags[name] == 1:
                rown.append(name.upper() if p == 1 else f"{name}_{p}")

    # Allocate output matrices
    ret = np.full((len(rown), 6), np.nan, dtype=float)
    ret_pl = np.full((len(rown), 6), np.nan, dtype=float) if placebo else None

    # Fill row by row
    r = 0
    for j, name in enumerate(estims, start=1):
        if flags[name] != 1:
            continue

        for p in range(1, max_T_XX + 1):
            # Pull scalars with the same naming pattern as R:
            # delta_{j}_{p}_XX, sd_delta_{j}_{p}_XX, LB_{j}_{p}_XX, UB_{j}_{p}_XX, N_Switchers_{j}_{p}_XX, N_Stayers_{j}_{p}_XX
            d_key = f"delta_{j}_{p}_XX"
            sd_key = f"sd_delta_{j}_{p}_XX"
            lb_key = f"LB_{j}_{p}_XX"
            ub_key = f"UB_{j}_{p}_XX"
            ns_key = f"N_Switchers_{j}_{p}_XX"
            nt_key = f"N_Stayers_{j}_{p}_XX"

            # Apply the same "set delta to NA if too few stayers/switchers" rule for p != 1
            delta_val = scalars.get(d_key, np.nan)
            n_stayers = scalars.get(nt_key, np.nan)
            n_switch = scalars.get(ns_key, np.nan)
            if p != 1:
                bad = (
                    ((not np.isfinite(n_stayers)) and (not np.isfinite(n_switch)))
                    or (np.isfinite(n_stayers) and n_stayers < 2)
                    or (np.isfinite(n_switch) and n_switch == 0)
                )
                if bad:
                    delta_val = np.nan

            ret[r, 0] = delta_val
            ret[r, 1] = scalars.get(sd_key, np.nan)
            ret[r, 2] = scalars.get(lb_key, np.nan)
            ret[r, 3] = scalars.get(ub_key, np.nan)
            ret[r, 4] = scalars.get(ns_key, np.nan)
            ret[r, 5] = scalars.get(nt_key, np.nan)

            # Placebo table: in R they skip p==2 (placebo not defined there)
            if placebo and p != 2:
                d_key = f"delta_{j}_{p}_pl_XX"
                sd_key = f"sd_delta_{j}_{p}_pl_XX"
                lb_key = f"LB_{j}_{p}_pl_XX"
                ub_key = f"UB_{j}_{p}_pl_XX"
                ns_key = f"N_Switchers_{j}_{p}_pl_XX"
                nt_key = f"N_Stayers_{j}_{p}_pl_XX"

                delta_val_pl = scalars.get(d_key, np.nan)
                n_stayers_pl = scalars.get(nt_key, np.nan)
                n_switch_pl = scalars.get(ns_key, np.nan)

                if p != 1:
                    bad_pl = (
                        ((not np.isfinite(n_stayers_pl)) and (not np.isfinite(n_switch_pl)))
                        or (np.isfinite(n_stayers_pl) and n_stayers_pl < 2)
                        or (np.isfinite(n_switch_pl) and n_switch_pl == 0)
                    )
                    if bad_pl:
                        delta_val_pl = np.nan

                ret_pl[r, 0] = delta_val_pl
                ret_pl[r, 1] = scalars.get(sd_key, np.nan)
                ret_pl[r, 2] = scalars.get(lb_key, np.nan)
                ret_pl[r, 3] = scalars.get(ub_key, np.nan)
                ret_pl[r, 4] = scalars.get(ns_key, np.nan)
                ret_pl[r, 5] = scalars.get(nt_key, np.nan)

            r += 1

    # Convert matrices to DataFrames
    table = pd.DataFrame(
        ret,
        index=rown,
        columns=["Estimate", "SE", "LB CI", "UB CI", "Switchers", "Stayers"],
    )

    out: Dict[str, Any] = {
        "table": table,
        "pairs": max_T_XX,
    }

    if placebo and ret_pl is not None:
        out["table_placebo"] = pd.DataFrame(
            ret_pl,
            index=rown,
            columns=["Estimate", "SE", "LB CI", "UB CI", "Switchers", "Stayers"],
        )

    if aoss_vs_waoss_table is not None:
        out["aoss_vs_waoss"] = aoss_vs_waoss_table

    if n_clus_XX is not None:
        out["n_clusters"] = n_clus_XX

    return out

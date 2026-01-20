# -*- coding: utf-8 -*-
"""
Replica did_multiplegt_stat (AS/WAS + placebos + test AS-WAS) en Python

CORREGIDO (para el test AS vs WAS):
- VCE cluster-robusta por id (CR1) sumando influencias por cluster
- Var(AS−WAS) incluye covarianza
- Inferencia: t(df = #clusters − 1)  [DEFAULT]
  (si quieres replicar “ado con z/normal”, setea INFERENCE="normal")

NOTA importante (como ya tenías):
- La Phi de AS NO lleva (w_t/n_t) en el término demeaned.
"""

import warnings
from pathlib import Path
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ------------------------------------------------------------
# Warnings
# ------------------------------------------------------------
try:
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
    warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_PATH = HERE / "gazoline_did_multiplegt_stat.dta"  # o .csv

ORDER = 1
PLACEBOS = [1, 2, 3]

ID_COL = "id"
T_COL  = "year"
Y_COL  = "lngpinc"
D_COL  = "tau"

# INFERENCE: "t" (corregido tipo Stata cluster-t) o "normal" (z)
INFERENCE = "t"

# ------------------------------------------------------------
# Normal helpers (z)
# ------------------------------------------------------------
_Z975 = 1.959963984540054  # invnormal(0.975)

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def normal_inference(est: float, se: float):
    """(stat, pval, lo, hi) con normal."""
    if not np.isfinite(se) or se <= 0:
        return np.nan, np.nan, np.nan, np.nan
    stat = est / se
    pval = 2.0 * (1.0 - _norm_cdf(abs(stat)))
    lo = est - _Z975 * se
    hi = est + _Z975 * se
    return stat, pval, lo, hi

# ------------------------------------------------------------
# t helpers
# ------------------------------------------------------------
def t_inference(est: float, se: float, df: int):
    """(t, pval, lo, hi) con t(df). Requiere scipy (viene con statsmodels)."""
    if not np.isfinite(se) or se <= 0 or df is None or df <= 0:
        return np.nan, np.nan, np.nan, np.nan

    try:
        from scipy.stats import t as student_t
    except Exception:
        # fallback: si no hay scipy, usa normal (aprox)
        return normal_inference(est, se)

    tstat = est / se
    crit = float(student_t.ppf(0.975, df))
    pval = float(2.0 * (1.0 - student_t.cdf(abs(tstat), df)))
    lo = est - crit * se
    hi = est + crit * se
    return tstat, pval, lo, hi

def do_inference(est: float, se: float, df: int, mode: str):
    mode = (mode or "").strip().lower()
    if mode == "normal":
        return normal_inference(est, se)
    # default: t
    return t_inference(est, se, df)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def poly_matrix(x, order: int):
    x = np.asarray(x, dtype=float)
    cols = [np.ones_like(x)]
    for k in range(1, order + 1):
        cols.append(x ** k)
    return np.column_stack(cols)

def glm_logit_prob(y01, X, maxiter=200):
    """
    p = P(y01=1 | X) por logit.
    Si falla, usa fit_regularized para estabilizar.
    """
    y01 = np.asarray(y01, dtype=float)
    X = np.asarray(X, dtype=float)
    try:
        res = sm.GLM(y01, X, family=sm.families.Binomial()).fit(maxiter=maxiter, disp=0)
        p = res.predict(X)
    except Exception:
        res = sm.GLM(y01, X, family=sm.families.Binomial()).fit_regularized(alpha=1e-8, L1_wt=0.0)
        p = res.predict(X)
    p = np.asarray(p, dtype=float)
    # permitir 0 exacto (como Stata cuando hay separación / predicción extrema)
    p = np.where(p <= 1e-12, 0.0, p)
    return p

def build_full_panel_and_transitions(df):
    years_full = np.arange(int(df[T_COL].min()), int(df[T_COL].max()) + 1)
    ids = np.sort(df[ID_COL].unique())

    full_index = pd.MultiIndex.from_product([ids, years_full], names=[ID_COL, T_COL])
    df_full = df.set_index([ID_COL, T_COL]).reindex(full_index).reset_index()
    df_full = df_full.sort_values([ID_COL, T_COL]).reset_index(drop=True)

    df_full["Y_l1"] = df_full.groupby(ID_COL)[Y_COL].shift(1)
    df_full["D_l1"] = df_full.groupby(ID_COL)[D_COL].shift(1)

    df_full["dY"] = df_full[Y_COL] - df_full["Y_l1"]
    df_full["dD"] = df_full[D_COL] - df_full["D_l1"]
    df_full["D1"] = df_full["D_l1"]

    df_full["Ht0"] = (
        df_full[Y_COL].notna()
        & df_full["Y_l1"].notna()
        & df_full[D_COL].notna()
        & df_full["D_l1"].notna()
    ).astype(int)

    return df_full

def make_placebo_cols(df_full, p):
    g = df_full.groupby(ID_COL)
    dY_p = g[Y_COL].shift(p) - g[Y_COL].shift(p + 1)
    dD_p = g[D_COL].shift(p) - g[D_COL].shift(p + 1)

    # Mantener el filtro base Ht0==1 (como tu versión), y además exigir placebo definido y dD_p==0
    Ht_p = ((df_full["Ht0"] == 1) & dY_p.notna() & dD_p.notna() & (dD_p == 0)).astype(int)
    df_full[f"dY_{p}"] = dY_p
    df_full[f"Ht_{p}"] = Ht_p
    return df_full

def usable_years(df_full, ht_col: str, order: int):
    d = df_full[df_full[ht_col] == 1]
    if d.empty:
        tmp = pd.DataFrame(columns=["year","n","switchers","stayers","usable_t"])
        return [], [], tmp

    tmp = (
        d.groupby(T_COL)["dD"]
         .agg(
             n="size",
             switchers=lambda s: int((s != 0).sum()),
             stayers=lambda s: int((s == 0).sum()),
         )
         .reset_index()
         .rename(columns={T_COL: "year"})
    )
    tmp["usable_t"] = (tmp["switchers"] > 0) & (tmp["stayers"] >= (order + 1))

    years_used = tmp.loc[tmp["usable_t"], "year"].tolist()
    years_excl = tmp.loc[(tmp["n"] > 0) & (~tmp["usable_t"]), "year"].tolist()
    return years_used, years_excl, tmp

def cluster_vcov_from_if(IF_mat, groups, df_correction=True):
    """
    IF_mat: (n,k) influencia por obs (en escala de influencia para el estimador)
    groups: (n,) cluster id

    Var(theta_hat) ≈ CR1 * (1/n^2) * sum_g (sum_{i in g} IF_i)(...)'
    CR1 = G/(G-1)
    + (opcional) (n-1)/(n-k) estilo Stata
    """
    IF_mat = np.asarray(IF_mat, float)
    groups = np.asarray(groups)

    # Reemplazar no-finitos por 0 para que el agregado por cluster no reviente
    IF_mat = np.where(np.isfinite(IF_mat), IF_mat, 0.0)

    n, k = IF_mat.shape
    df = pd.DataFrame({"g": groups})
    for j in range(k):
        df[f"IF{j}"] = IF_mat[:, j]

    S = df.groupby("g", sort=False)[[f"IF{j}" for j in range(k)]].sum().to_numpy()  # (G,k)
    G = int(S.shape[0])
    if G <= 1:
        return np.full((k, k), np.nan), G

    meat = S.T @ S

    cr1 = G / (G - 1.0)
    scale = cr1 / (n ** 2)

    if df_correction and n > k:
        scale *= (n - 1.0) / (n - k)

    V = scale * meat
    return V, G

# ------------------------------------------------------------
# Core estimator (p=0 o placebo p>=1) con Phi + VCE cluster
# ------------------------------------------------------------
def estimate_with_phi(df_full, p: int, order: int, inference_mode: str):
    if p == 0:
        ht_col = "Ht_0"
        dy_col = "dY_0"
    else:
        ht_col = f"Ht_{p}"
        dy_col = f"dY_{p}"

    years_used, years_excl, by_year = usable_years(df_full, ht_col, order)

    base = df_full[(df_full[ht_col] == 1) & (df_full[T_COL].isin(years_used))].copy()
    base = base.sort_values([T_COL, ID_COL]).reset_index(drop=True)
    nobs = len(base)

    # acumuladores por obs
    phiN_as = np.zeros(nobs, float)
    phiW_as = np.zeros(nobs, float)

    dr_was = np.zeros(nobs, float)
    absdD  = np.zeros(nobs, float)

    # loop por año
    for t in years_used:
        idx = np.where(base[T_COL].to_numpy() == t)[0]
        d = base.iloc[idx]

        y  = d[dy_col].to_numpy(float)
        dD = d["dD"].to_numpy(float)
        D1 = d["D1"].to_numpy(float)

        Sbis   = (dD != 0).astype(int)
        stayer = (Sbis == 0)
        X = poly_matrix(D1, order)

        # mean_pred en stayers
        mean_pred = sm.OLS(y[stayer], X[stayer, :]).fit().predict(X)
        inner = y - mean_pred

        # PS0 = P(stayer|D1)
        PS0 = glm_logit_prob(stayer.astype(int), X)
        PS0 = np.where(PS0 <= 1e-10, 0.0, PS0)

        # -------- AS --------
        S_over_dD = np.where(Sbis == 1, 1.0 / dD, 0.0)
        meanS_over = sm.OLS(S_over_dD, X).fit().predict(X)

        dr_as = np.where(
            Sbis == 1,
            S_over_dD * inner,
            np.where(PS0 > 0, -(meanS_over / PS0) * inner, np.nan),
        )

        delta1_t = float(np.nanmean(dr_as))

        # CLAVE (como tu nota): SIN (w_t/n_t) en (dr_as - delta1_t)
        phiN_as[idx] = delta1_t * Sbis + (dr_as - delta1_t)
        phiW_as[idx] = Sbis

        # -------- WAS --------
        S = np.sign(dD).astype(int)
        PS1Plus  = glm_logit_prob((S == 1).astype(int), X)
        PS1Minus = glm_logit_prob((S == -1).astype(int), X)
        PS1Plus  = np.where(PS1Plus  <= 1e-10, 0.0, PS1Plus)
        PS1Minus = np.where(PS1Minus <= 1e-10, 0.0, PS1Minus)

        ratio = np.where(PS0 > 0, (PS1Plus - PS1Minus) / PS0, np.nan)
        dr_t  = np.where(Sbis == 1, S * inner, -ratio * inner)

        dr_was[idx] = dr_t
        absdD[idx]  = np.abs(dD)

    # point estimates (ratio)
    AS  = float(np.nansum(phiN_as) / np.nansum(phiW_as))
    WAS = float(np.nansum(dr_was)  / np.nansum(absdD))
    DIFF = AS - WAS

    # influence functions (ratio form)
    mW_as  = float(np.nanmean(phiW_as))
    mW_was = float(np.nanmean(absdD))

    IF_as  = (phiN_as - AS  * phiW_as) / mW_as
    IF_was = (dr_was  - WAS * absdD)   / mW_was

    groups = base[ID_COL].to_numpy()
    V2, G = cluster_vcov_from_if(np.column_stack([IF_as, IF_was]), groups, df_correction=True)
    se_as  = float(np.sqrt(V2[0, 0]))
    se_was = float(np.sqrt(V2[1, 1]))
    cov_aw = float(V2[0, 1])
    se_diff = float(np.sqrt(se_as**2 + se_was**2 - 2.0 * cov_aw))

    # df para t cluster
    df_t = int(G - 1)

    # inferencia
    stat_as,  p_as,  lo_as,  hi_as  = do_inference(AS,   se_as,  df_t, inference_mode)
    stat_was, p_was, lo_was, hi_was = do_inference(WAS,  se_was, df_t, inference_mode)
    stat_d,   p_d,   lo_d,   hi_d   = do_inference(DIFF, se_diff, df_t, inference_mode)

    switchers = int(np.nansum(phiW_as))
    stayers   = int(nobs - switchers)

    return dict(
        p=p,
        nobs=nobs,
        years_used=years_used,
        years_excl=years_excl,
        by_year=by_year,
        G=G,
        df_t=df_t,

        AS=AS, se_as=se_as, lo_as=lo_as, hi_as=hi_as, stat_as=stat_as, p_as=p_as,
        WAS=WAS, se_was=se_was, lo_was=lo_was, hi_was=hi_was, stat_was=stat_was, p_was=p_was,
        DIFF=DIFF, se_diff=se_diff, lo_diff=lo_d, hi_diff=hi_d, stat_diff=stat_d, p_diff=p_d,

        switchers=switchers,
        stayers=stayers,
    )

# ------------------------------------------------------------
# Pretty printing (Stata-ish)
# ------------------------------------------------------------
def fmt_stata(x, nd=7):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "."
    s = f"{x:.{nd}f}"
    if s.startswith("0."):
        s = "." + s[2:]
    elif s.startswith("-0."):
        s = "-." + s[3:]
    return s

def print_header(nobs, order):
    print("                                  ----------------------------------------------")
    print(f"                                   Number of observations     = {nobs:>16d}")
    print("                                   Estimation method          =    doubly-robust")
    print(f"                                   Polynomial order           = {f'({order})':>16s}")
    print("                                  ----------------------------------------------")

def print_block(title, rows):
    print("--------------------------------------------------------------------------------")
    print(f"{title:^78s}")
    print("--------------------------------------------------------------------------------\n")
    print(f"{'':13s} |  {'Estimate':>8s} {'SE':>10s} {'LB CI':>10s} {'UB CI':>10s}  {'Switchers':>9s} {'Stayers':>9s}")
    print("-------------+------------------------------------------------------------------")
    for r in rows:
        print(
            f"{r['name']:13s} |"
            f"  {fmt_stata(r['est'], 7):>8s}"
            f" {fmt_stata(r['se'], 7):>10s}"
            f" {fmt_stata(r['lo'], 7):>10s}"
            f" {fmt_stata(r['hi'], 7):>10s}"
            f" {int(r['sw']):>10d}"
            f" {int(r['st']):>9d}"
        )
    print("--------------------------------------------------------------------------------")

def print_diff(res, inference_mode: str):
    stat_label = "t" if (inference_mode or "").lower() != "normal" else "z"
    print("\n--------------------------------------------------------------------------------\n")
    print("                          Test of difference between AS and WAS")
    print("H0: AS = WAS")
    print("--------------------------------------------------------------------------------\n")
    print(f"{'':13s} |  {'Diff.':>8s} {'SE':>10s} {'LB CI':>10s} {'UB CI':>10s}  {'pval.':>10s} {stat_label:>10s}")
    print("-------------+------------------------------------------------------------------")
    print(
        f"{'AS-WAS':13s} |"
        f"  {fmt_stata(res['DIFF'], 7):>8s}"
        f" {fmt_stata(res['se_diff'], 7):>10s}"
        f" {fmt_stata(res['lo_diff'], 7):>10s}"
        f" {fmt_stata(res['hi_diff'], 7):>10s}"
        f"  {fmt_stata(res['p_diff'], 7):>10s}"
        f" {fmt_stata(res['stat_diff'], 6):>10s}"
    )
    print("--------------------------------------------------------------------------------")

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No encuentro el archivo: {DATA_PATH}")

    if str(DATA_PATH).lower().endswith(".dta"):
        df = pd.read_stata(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)

    df = df[[ID_COL, T_COL, Y_COL, D_COL]].copy()
    df[ID_COL] = pd.to_numeric(df[ID_COL], errors="raise").astype(int)
    df[T_COL]  = pd.to_numeric(df[T_COL],  errors="raise").astype(int)

    df_full = build_full_panel_and_transitions(df)

    # base p=0 cols
    df_full["dY_0"] = df_full["dY"]
    df_full["Ht_0"] = df_full["Ht0"]

    # placebos: crear cols una sola vez
    for p in PLACEBOS:
        df_full = make_placebo_cols(df_full, p)

    # estimar una vez cada p y cachear (para no recalcular 2 veces)
    results = {}
    results[0] = estimate_with_phi(df_full, p=0, order=ORDER, inference_mode=INFERENCE)
    for p in PLACEBOS:
        results[p] = estimate_with_phi(df_full, p=p, order=ORDER, inference_mode=INFERENCE)

    res0 = results[0]
    print_header(res0["nobs"], ORDER)

    # AS
    print_block("Average Slope (AS)", [{
        "name": "AS",
        "est": res0["AS"], "se": res0["se_as"], "lo": res0["lo_as"], "hi": res0["hi_as"],
        "sw": res0["switchers"], "st": res0["stayers"]
    }])

    # Placebos AS
    print_block("Placebo(s) AS", [{
        "name": f"Placebo_{p}",
        "est": results[p]["AS"], "se": results[p]["se_as"], "lo": results[p]["lo_as"], "hi": results[p]["hi_as"],
        "sw": results[p]["switchers"], "st": results[p]["stayers"]
    } for p in PLACEBOS])

    # WAS
    print_block("Weighted Average Slope (WAS)", [{
        "name": "WAS",
        "est": res0["WAS"], "se": res0["se_was"], "lo": res0["lo_was"], "hi": res0["hi_was"],
        "sw": res0["switchers"], "st": res0["stayers"]
    }])

    # Placebos WAS
    print_block("Placebo(s) WAS", [{
        "name": f"Placebo_{p}",
        "est": results[p]["WAS"], "se": results[p]["se_was"], "lo": results[p]["lo_was"], "hi": results[p]["hi_was"],
        "sw": results[p]["switchers"], "st": results[p]["stayers"]
    } for p in PLACEBOS])

    # diff
    print_diff(res0, inference_mode=INFERENCE)

    # clusters
    print(f"\n(Number of clusters = {int(res0['G'])})")
    if INFERENCE.lower() != "normal":
        print(f"(t df = {int(res0['df_t'])})")

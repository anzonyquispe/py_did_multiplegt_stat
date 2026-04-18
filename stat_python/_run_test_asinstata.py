"""Run all 6 examples with asinstata=True and asinstata=False, compare results."""
import warnings
warnings.filterwarnings('ignore')

import sys
import pandas as pd
import numpy as np
from did_multiplegt_stat import did_multiplegt_stat

df = pd.read_stata('gazoline_did_multiplegt_stat.dta')
df_g = pd.read_stata('gentzkowetal_didtextbook.dta')
print(f'Gazoline: {df.shape}')
print(f'Gentzkow: {df_g.shape}')


def extract_estimates(result):
    out = {}
    tbl = result.get('results', {}).get('table', None)
    if isinstance(tbl, pd.DataFrame):
        for idx in tbl.index:
            out[idx] = tbl.iloc[tbl.index.get_loc(idx), 0]
    for pl in range(1, 5):
        pl_tbl = result.get('results', {}).get(f'table_placebo_{pl}', None)
        if isinstance(pl_tbl, pd.DataFrame):
            for idx in pl_tbl.index:
                out[f'Plac{pl}_{idx}'] = pl_tbl.iloc[pl_tbl.index.get_loc(idx), 0]
    return out


def compare_results(name, res_true, res_false):
    est_t = extract_estimates(res_true)
    est_f = extract_estimates(res_false)
    all_keys = sorted(set(list(est_t.keys()) + list(est_f.keys())))
    rows = []
    for k in all_keys:
        vt = est_t.get(k, np.nan)
        vf = est_f.get(k, np.nan)
        if pd.notna(vt) and pd.notna(vf) and vt != 0:
            pct = abs(vt - vf) / abs(vt) * 100
        else:
            pct = np.nan
        rows.append({'Estimator': k, 'True': vt, 'False': vf, 'Diff%': pct})
    comp = pd.DataFrame(rows)
    print(f'\n{"="*70}')
    print(f'  {name}')
    print(f'{"="*70}')
    print(comp.to_string(index=False))
    return comp


# V.01
print('\n>>> V.01 asinstata=True...')
r01_t = did_multiplegt_stat(df, 'lngca', 'id', 'year', 'tau',
                            order=1, aoss_vs_waoss=True, controls=['lngpinc'], asinstata=True)
print('>>> V.01 asinstata=False...')
r01_f = did_multiplegt_stat(df, 'lngca', 'id', 'year', 'tau',
                            order=1, aoss_vs_waoss=True, controls=['lngpinc'], asinstata=False)
comp01 = compare_results('V.01: AOSS+WAOSS, controls, Y=lngca', r01_t, r01_f)

# V.02
print('\n>>> V.02 asinstata=True...')
r02_t = did_multiplegt_stat(df, 'lngpinc', 'id', 'year', 'tau',
                            order=1, aoss_vs_waoss=True, controls=['lngpinc'], asinstata=True)
print('>>> V.02 asinstata=False...')
r02_f = did_multiplegt_stat(df, 'lngpinc', 'id', 'year', 'tau',
                            order=1, aoss_vs_waoss=True, controls=['lngpinc'], asinstata=False)
comp02 = compare_results('V.02: AOSS+WAOSS, controls, Y=lngpinc', r02_t, r02_f)

# V.03
print('\n>>> V.03 asinstata=True...')
r03_t = did_multiplegt_stat(df, 'lngca', 'id', 'year', 'lngpinc',
                            Z='tau', estimator='ivwaoss', order=1,
                            controls=['lngpinc'], bootstrap=5, seed=1, asinstata=True)
print('>>> V.03 asinstata=False...')
r03_f = did_multiplegt_stat(df, 'lngca', 'id', 'year', 'lngpinc',
                            Z='tau', estimator='ivwaoss', order=1,
                            controls=['lngpinc'], bootstrap=5, seed=1, asinstata=False)
comp03 = compare_results('V.03: IV-WAOSS, bootstrap=5', r03_t, r03_f)

# V.04
print('\n>>> V.04 asinstata=True...')
r04_t = did_multiplegt_stat(df, 'lngca', 'id', 'year', 'tau',
                            estimator='waoss', order=1, controls=['lngpinc'],
                            on_placebo_sample=True, asinstata=True)
print('>>> V.04 asinstata=False...')
r04_f = did_multiplegt_stat(df, 'lngca', 'id', 'year', 'tau',
                            estimator='waoss', order=1, controls=['lngpinc'],
                            on_placebo_sample=True, asinstata=False)
comp04 = compare_results('V.04: WAOSS, on_placebo_sample, Y=lngca', r04_t, r04_f)

# V.05
print('\n>>> V.05 asinstata=True...')
r05_t = did_multiplegt_stat(df, 'lngpinc', 'id', 'year', 'tau',
                            estimator='waoss', order=1, controls=['lngpinc'],
                            on_placebo_sample=True, asinstata=True)
print('>>> V.05 asinstata=False...')
r05_f = did_multiplegt_stat(df, 'lngpinc', 'id', 'year', 'tau',
                            estimator='waoss', order=1, controls=['lngpinc'],
                            on_placebo_sample=True, asinstata=False)
comp05 = compare_results('V.05: WAOSS, on_placebo_sample, Y=lngpinc', r05_t, r05_f)

# VI.01
print('\n>>> VI.01 asinstata=True...')
r06_t = did_multiplegt_stat(df_g, 'prestout', 'cnty90', 'year', 'numdailies',
                            placebo=1, exact_match=True, asinstata=True)
print('>>> VI.01 asinstata=False...')
r06_f = did_multiplegt_stat(df_g, 'prestout', 'cnty90', 'year', 'numdailies',
                            placebo=1, exact_match=True, asinstata=False)
comp06 = compare_results('VI.01: Gentzkow, exact_match, placebo=1', r06_t, r06_f)

# Summary
print('\n' + '='*70)
print('  SUMMARY: max |Diff%| per example')
print('='*70)
for name, comp in [('V.01', comp01), ('V.02', comp02), ('V.03', comp03),
                   ('V.04', comp04), ('V.05', comp05), ('VI.01', comp06)]:
    max_pct = comp['Diff%'].max()
    print(f'  {name}: max Diff% = {max_pct:.6f}%')

print('\nDONE.')

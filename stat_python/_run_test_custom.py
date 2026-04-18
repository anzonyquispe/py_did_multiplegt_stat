"""Run custom model examples and compare."""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from did_multiplegt_stat import did_multiplegt_stat

df = pd.read_stata('gazoline_did_multiplegt_stat.dta')
print(f'Gazoline: {df.shape}')


def extract_all(result):
    out = {}
    tbl = result.get('results', {}).get('table', None)
    if isinstance(tbl, pd.DataFrame):
        for idx in tbl.index:
            out[idx] = tbl.iloc[tbl.index.get_loc(idx), 0]
    return out


# Example 1: RandomForest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
print('\n>>> Example 1: RandomForest...')
r_rf = did_multiplegt_stat(df, 'lngca', 'id', 'year', 'tau',
                           order=1, controls=['lngpinc'],
                           model_deltay=RandomForestRegressor(n_estimators=100, random_state=42),
                           model_stayer=RandomForestClassifier(n_estimators=100, random_state=42))
print('Done.')

# Example 2: LassoCV
from sklearn.linear_model import LassoCV
print('\n>>> Example 2: LassoCV...')
r_lasso = did_multiplegt_stat(df, 'lngca', 'id', 'year', 'tau',
                              order=1, controls=['lngpinc'],
                              model_deltay=LassoCV())
print('Done.')

# Example 3: Default sklearn
print('\n>>> Example 3: Default sklearn...')
r_default = did_multiplegt_stat(df, 'lngca', 'id', 'year', 'tau',
                                order=1, controls=['lngpinc'])
print('Done.')

# Example 4: asinstata=True
print('\n>>> Example 4: asinstata=True...')
r_stata = did_multiplegt_stat(df, 'lngca', 'id', 'year', 'tau',
                              order=1, controls=['lngpinc'],
                              asinstata=True)
print('Done.')

# Comparison
est_rf = extract_all(r_rf)
est_lasso = extract_all(r_lasso)
est_default = extract_all(r_default)
est_stata = extract_all(r_stata)

all_keys = sorted(set(list(est_rf.keys()) + list(est_lasso.keys()) +
                      list(est_default.keys()) + list(est_stata.keys())))

rows = []
for k in all_keys:
    rows.append({
        'Estimator': k,
        'RandomForest': est_rf.get(k, np.nan),
        'LassoCV': est_lasso.get(k, np.nan),
        'Default(sklearn)': est_default.get(k, np.nan),
        'asinstata=True': est_stata.get(k, np.nan),
    })

comp = pd.DataFrame(rows)
print('\n' + '='*90)
print('  Comparison: 4 model configurations')
print('='*90)
print(comp.to_string(index=False))

# Diff from asinstata=True baseline
print('\n' + '='*90)
print('  Difference from asinstata=True (%)')
print('='*90)
for _, row in comp.iterrows():
    base = row['asinstata=True']
    if pd.notna(base) and base != 0:
        rf_d = abs(row['RandomForest'] - base) / abs(base) * 100 if pd.notna(row['RandomForest']) else np.nan
        la_d = abs(row['LassoCV'] - base) / abs(base) * 100 if pd.notna(row['LassoCV']) else np.nan
        sk_d = abs(row['Default(sklearn)'] - base) / abs(base) * 100 if pd.notna(row['Default(sklearn)']) else np.nan
        print(f"  {row['Estimator']:30s}  RF:{rf_d:8.4f}%  Lasso:{la_d:8.4f}%  Sklearn:{sk_d:8.4f}%")

print('\nDONE.')

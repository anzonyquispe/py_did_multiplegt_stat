"""Debug script for V.01 — Reduced Form with cross_fitting(10).
Ablation test: identifies which nuisance parameter (mu, p, g) drives the AS discrepancy.
"""
import os
import sys
import re
os.environ["DMS_DEBUG_CF"] = "ABLATION"

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from io import StringIO
from contextlib import redirect_stdout

from did_multiplegt_stat import did_multiplegt_stat, summary_did_multiplegt_stat

df = pd.read_stata(os.path.join(os.path.dirname(__file__), "gazoline_did_multiplegt_stat.dta"))

print("="*70)
print("V.01: Reduced Form, cross_fitting(10)")
print("Stata reference: AS=-0.0043121, WAS=-0.0035981")
print("="*70)

# Capture output
buf = StringIO()
import warnings
warnings.filterwarnings("ignore")

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

tee = Tee(sys.stdout, buf)
old_stdout = sys.stdout
sys.stdout = tee

r = did_multiplegt_stat(df, "lngca", "id", "year", "tau",
                        order=1, aoss_vs_waoss=True,
                        controls=["lngpinc"], cross_fitting=10)
summary_did_multiplegt_stat(r)

sys.stdout = old_stdout

# Parse output for ablation analysis
output = buf.getvalue()

# Parse per-pair data: P_t weights and ablation delta1 values
pairs = {}
for line in output.split('\n'):
    m = re.match(r'\s+\[CF-AS\] p=(\d+) plac=0: delta1=([\-\d.]+) Pt=([\-\d.]+)', line)
    if m:
        p = int(m.group(1))
        pairs[p] = {'delta1': float(m.group(2)), 'Pt': float(m.group(3)), 'ablation': {}}

    m = re.match(r'\s+ABLATION p=(\d+) ([\w_]+\s*): delta1=([\-\d.]+)', line)
    if m:
        p = int(m.group(1))
        name = m.group(2).strip()
        val = float(m.group(3))
        if p in pairs:
            pairs[p]['ablation'][name] = val

# Compute aggregate AS for each ablation scenario
print("\n" + "="*70)
print("ABLATION SUMMARY: Aggregate AS for each nuisance param combination")
print("="*70)
print(f"Stata reference: AS = -0.0043121")
print()

scenarios = ["all_CF", "full_p", "full_g", "full_mu", "full_p_g", "full_p_mu", "full_g_mu", "all_FULL"]
PS_sum = sum(d['Pt'] for d in pairs.values())

print(f"{'Scenario':<15} {'Agg AS':>12} {'vs Stata':>10} {'vs allCF':>10}")
print("-"*50)

for sc in scenarios:
    agg_num = 0.0
    for p, d in pairs.items():
        d1 = d['ablation'].get(sc, d['delta1'])
        agg_num += d['Pt'] * d1
    agg_as = agg_num / PS_sum if PS_sum != 0 else 0.0
    diff_stata = (agg_as - (-0.0043121)) / abs(-0.0043121) * 100
    all_cf_as = sum(d['Pt'] * d['ablation'].get('all_CF', d['delta1']) for d in pairs.values()) / PS_sum
    diff_cf = (agg_as - all_cf_as) / abs(all_cf_as) * 100 if all_cf_as != 0 else 0
    print(f"{sc:<15} {agg_as:>12.7f} {diff_stata:>9.1f}% {diff_cf:>9.1f}%")

print()
print("Key: full_X means using full-sample (non-CF) for parameter X")
print("  p = P(S=0|D1) from logit")
print("  g = E(S/ΔD|D1) from OLS")
print("  mu = E(ΔY|D1,S=0) from OLS")

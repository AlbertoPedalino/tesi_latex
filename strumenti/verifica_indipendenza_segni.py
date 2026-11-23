"""Are the sign tests of Sez. 5.5 affected by dependence between clusters? (check for Sez. 7.5.1)

For every comparison G_n - method whose cluster-by-cluster data are on disk (start and end: match files; along-motion
displacement: lead files, the same runs as figures/prepara_dati_moto.py), the clusters are ordered by number (= by
first instant, as in prepara_dati_moto.py) and the block length of Wilks (1997) for AR(2) data is computed on the series
of the differences and on the series of their signs, with the function of prepara_dati_moto.py. Where it is larger
than one, the 1% sign test is repeated as a block test: 99% interval of the share of 'G_n ahead' among the non-tied
pairs, from B = 4000 moving-block bootstrap samples of the clusters (blocks of that length, seed 0); the verdict is the
same when both say significant or both say not significant.

The match files without the closing and with the minimum of 10, 64, 1125 values were copied from the server on
23/11/2026 (outputs/concordate_5pc_senza_chiusura/match_r300.csv of 15/11, outputs/concordate_5pc_regola_tesi_min{K}/
match_r300_m{K}.csv of 18/11) into the same folders of log_server.
Output: log_server/verifica_indipendenza_segni.txt (Tesi folder).
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('moto', HERE.parent / 'figures' / 'prepara_dati_moto.py')
moto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moto)
LOGS = HERE.parent.parent / 'log_server'
OUT = LOGS / 'verifica_indipendenza_segni.txt'
OTHERS = ['pca', 'cblof', 'mcd', 'hbos', 'copod', 'iforest', 'normal', 'change_3h']
MATCH = [
    ('95-esimo', 'r300/outputs/concordate_5pc_regola_tesi/match_r300.csv'),
    ('97-esimo', 'r300/outputs/concordate_3pc_regola_tesi/match_r300.csv'),
    ('99-esimo', 'r300/outputs/concordate_1pc_regola_tesi/match_r300.csv'),
    ('100 km', 'sensibilita_1911/outputs/concordate_5pc_regola_tesi/match_r100.csv'),
    ('500 km', 'sensibilita_1911/outputs/concordate_5pc_regola_tesi/match_r500.csv'),
    ('Crawford per intero', 'sensibilita_1911/outputs/concordate_5pc_regola_tesi/match_r300_crawford500.csv'),
    ('96 h', 'sensibilita_1911/outputs/concordate_5pc_regola_tesi_96h/match_r300.csv'),
    ('senza chiusura', 'senza_chiusura/outputs/concordate_5pc_senza_chiusura/match_r300.csv'),
    ('minimo 10', 'minimo/outputs/concordate_5pc_regola_tesi_min10/match_r300_m10.csv'),
    ('minimo 64', 'minimo/outputs/concordate_5pc_regola_tesi_min64/match_r300_m64.csv'),
    ('minimo 1125', 'minimo/outputs/concordate_5pc_regola_tesi_min1125/match_r300_m1125.csv'),
]
B = 4000
rng = np.random.default_rng(0)
rows = []


def check(test, name, kind, measure, other, d):
    s = np.sign(d)
    ahead, behind = int((s > 0).sum()), int((s < 0).sum())
    p = binomtest(ahead, ahead + behind, 0.5).pvalue
    r1, _, _, l_diff = moto.wilks_length(d)
    l_sign = moto.wilks_length(s)[3]
    length = max(l_diff, l_sign)
    block = None
    if length > 1:
        shares = []
        for _ in range(B):
            t = s[moto.blocks(len(s), length, rng, circular=False)]
            t = t[t != 0]
            shares.append((t > 0).mean())
        low, high = np.percentile(shares, [0.5, 99.5])
        block = bool(low > 0.5 or high < 0.5)
    rows.append(dict(test=test, variant=name, kind=kind, measure=measure, other=other, n=len(d), r1=round(r1, 3),
                     l_diff=l_diff, l_sign=l_sign, p=f'{p:.3g}', sig_1pc=p < 0.01, sig_blocks=block,
                     same=(block is None) or (block == (p < 0.01))))


for name, rel in MATCH:
    frame = pd.read_csv(LOGS / rel)
    frame = frame[(frame['passes_crawford'].astype(str) == 'True') & (frame['border'].astype(str) == 'False')]
    for kind in sorted(frame['kind'].unique()):
        part = frame[frame['kind'] == kind]
        for column in ['start_lead_h', 'end_lead_h']:
            wide = part.pivot_table(index='cluster', columns='detector', values=column, aggfunc='first').sort_index()
            for other in OTHERS:
                if other in wide and 'Gn' in wide:
                    pair = wide[['Gn', other]].dropna()
                    if len(pair) >= 20:
                        check('start/end', name, kind, column, other, (pair['Gn'] - pair[other]).to_numpy())
for name, kind, rel in moto.RUNS:
    path = LOGS / rel
    if path.exists():
        wide = moto.table(path)
        for other in OTHERS:
            pair = wide[['Gn', other]].dropna()
            if len(pair) >= 20:
                check('along motion', name, kind, 'along_km', other, (pair['Gn'] - pair[other]).to_numpy())

out = pd.DataFrame(rows)
pd.set_option('display.width', 220)
lines = [
    __doc__.strip(), '',
    f'comparisons: {len(out)}; |r1| max {out.r1.abs().max():.3f}; Wilks l = 1 on the differences: {(out.l_diff == 1).sum()}; '
    f'on the signs: {(out.l_sign == 1).sum()}',
    f'with l > 1 (differences or signs): {(out.sig_blocks.notna()).sum()}; same 1% verdict with blocks: '
    f'{int(out[out.sig_blocks.notna()].same.sum())}; all comparisons with the same verdict: {int(out.same.sum())}', '',
    out.to_string(index=False),
]
OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
print('\n'.join(lines[:5]))
print('written', OUT)

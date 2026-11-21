"""Write the data of the along-motion displacement (Section 6.6.3, Appendix A) from the per-cluster files of
scripts/check_era5_concordant_lead.py of the code repository (column along_km; zone of 300 km).

Run from the thesis folder:
    python figures/prepara_dati_moto.py
Reads Tesi/log_server (the copies of the server folders listed in RUNS) and writes, in figures/dati:
- era5_concordate_moto_medie.csv: for every method and for the short and the long concordant clusters, the mean
  displacement, its 95% bootstrap interval, the share of clusters ahead, and the interval of the mean difference
  G_n - method, cluster by cluster;
- era5_concordate_moto_distribuzione.csv: the histogram (bars of 10 km between -150 and +150 km, percentage of the
  clusters), the 20th and 80th percentiles, the mean and the share beyond the bars of G_n, three-hour change and MCD;
- era5_concordate_moto_sensibilita.csv: mean and median displacement and share ahead of every method for every run of
  Appendix A, with the 95% bootstrap interval of G_n;
- prepara_dati_moto_log.txt: the block lengths and the check with circular blocks.

Rows: the matched concordant clusters (column matched) with a displacement (along_km not missing: a method with no flag
in the zone of the cluster has none); the rows of the probability levels (Gn_p..., Gn_rand...) are left out.

Summary, as Leonardo and Colle (2017, Sec. 2c and Fig. 4; 2020, Sec. 3 and Fig. 2): the mean over the clusters, with
the 95% interval of the percentile method from B = 10 000 bootstrap samples (2017: "resampled from the original data
10 000 times to obtain the 95th percentile confidence intervals around the sample mean"). Share ahead: clusters with a
positive displacement (2020: "ATEs that are positive"). Distribution: histogram with the 20th and 80th percentiles
(2017, Fig. 3a; 2020, Fig. 3).

Dependence (Wilks 1997; Gilleland et al. 2018): the clusters are ordered by their first instant (the cluster number of
the cube grows with it: checked on clusters.csv of the cube, column t0) and resampled in blocks of consecutive clusters,
the same blocks for every method (Wilks 1997, Sec. 3). The block length l is Wilks's rule for AR(2) data (eqs. 3, 6,
21-24), computed for every method and averaged (Wilks 1997, p. 77); l = 1 is the ordinary bootstrap. Check: circular
block bootstrap with blocks of sqrt(n) clusters (Gilleland et al. 2018, Sec. 2b). The difference G_n - method is the
one-sample case on the series of the differences (Wilks 1997, p. 69; Gilleland et al. 2018, Sec. 1: one interval
around the difference), with l from the rule on that series.
"""
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
LOGS = HERE.parent.parent / "log_server"
MAIN = LOGS / "r300" / "outputs" / "concordate_5pc_regola_tesi"
OUT = HERE / "dati"
B = 10000
SEED = 0
ORDER = ["Gn", "change_3h", "normal", "pca", "cblof", "mcd", "hbos", "copod", "iforest"]
SHOWN = ["Gn", "change_3h", "mcd"]  # the methods of the histogram: G_n and the two baselines ahead
EDGES = np.arange(-150, 151, 10)
# Appendix A: (column, kind, file under log_server)
RUNS = [
    ("95-esimo", "brevi", "r300/outputs/concordate_5pc_regola_tesi/lead_r300.csv"),
    ("95-esimo", "lunghi", "r300/outputs/concordate_5pc_regola_tesi/lead_r300_long.csv"),
    ("97-esimo", "brevi", "r300/outputs/concordate_3pc_regola_tesi/lead_r300.csv"),
    ("97-esimo", "lunghi", "r300/outputs/concordate_3pc_regola_tesi/lead_r300_long.csv"),
    ("99-esimo", "brevi", "r300/outputs/concordate_1pc_regola_tesi/lead_r300.csv"),
    ("99-esimo", "lunghi", "r300/outputs/concordate_1pc_regola_tesi/lead_r300_long.csv"),
    ("senza chiusura", "brevi", "senza_chiusura/outputs/concordate_5pc_senza_chiusura/lead_r300.csv"),
    ("senza chiusura", "lunghi", "esclusive_1911/outputs/concordate_5pc_senza_chiusura/lead_r300_long_excl.csv"),
    ("minimo 10", "lunghi", "minimo/outputs/concordate_5pc_regola_tesi_min10/lead_r300_long.csv"),
    ("minimo 64", "lunghi", "minimo/outputs/concordate_5pc_regola_tesi_min64/lead_r300_long.csv"),
    ("minimo 1125", "lunghi", "minimo/outputs/concordate_5pc_regola_tesi_min1125/lead_r300_long.csv"),
    ("96 h", "lunghi", "sensibilita_1911/outputs/concordate_5pc_regola_tesi_96h/lead_r300_long.csv"),
]
LOG_LINES = []


def log(text):
    print(text)
    LOG_LINES.append(text)


def table(path):
    """Clusters x methods, the displacement in km (NaN where the method has none), clusters by first instant."""
    frame = pd.read_csv(path, usecols=["cluster", "detector", "matched", "along_km"])
    frame = frame[(frame["matched"].astype(str) == "True") & frame["detector"].isin(ORDER)]
    return frame.pivot(index="cluster", columns="detector", values="along_km").sort_index()[ORDER]


def wilks_length(series):
    """Block length of Wilks (1997) for AR(2) data: eqs. (6), (21), (22), (3), (23), (24)."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    d = x - x.mean()
    r1 = (d[:-1] * d[1:]).sum() / (d ** 2).sum()
    a, b = x[:-2] - x[:-2].mean(), x[2:] - x[2:].mean()
    r2 = (a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum())
    phi1, phi2 = r1 * (1 - r2) / (1 - r1 ** 2), (r2 - r1 ** 2) / (1 - r1 ** 2)
    r = np.empty(n)
    r[0], r[1], r[2] = 1.0, r1, r2
    for k in range(3, n):
        r[k] = phi1 * r[k - 1] + phi2 * r[k - 2]
    k = np.arange(1, n)
    v = 1 + 2 * ((1 - k / n) * r[1:]).sum()
    v = v * np.exp(3 * v / n)
    length = np.sqrt(n)
    for _ in range(200):
        length = (n - length + 1) ** ((2 / 3) * (1 - 1 / v))
    return r1, r2, v, max(1, int(np.floor(length)))


def blocks(n, length, rng, circular):
    """Indices of one bootstrap sample of n values made of blocks of consecutive values."""
    count = int(np.ceil(n / length))
    starts = rng.integers(0, n if circular else n - length + 1, size=count)
    return ((starts[:, None] + np.arange(length)[None, :]) % n).ravel()[:n]


def interval(x, length, rng, circular=False):
    """95% interval of the mean of every column, percentile method, B samples, the same blocks for every column."""
    means = np.empty((B, x.shape[1]))
    for i in range(B):
        means[i] = np.nanmean(x[blocks(len(x), length, rng, circular)], axis=0)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def summarise(kind, path, rng):
    wide = table(path)
    x = wide.to_numpy()
    n = len(x)
    lengths = []
    for j, key in enumerate(ORDER):
        r1, r2, v, length = wilks_length(x[:, j])
        lengths.append(length)
        log(f"{kind} {key}: n {np.isfinite(x[:, j]).sum()}, r1 {r1:.3f}, r2 {r2:.3f}, V' {v:.2f}, Wilks l {length}")
    length = max(1, int(round(np.mean(lengths))))
    root = int(round(np.sqrt(n)))
    log(f"{kind}: {n} clusters, block length (mean of the rule) {length}; check with circular blocks of {root}")
    low, high = interval(x, length, rng)
    check_low, check_high = interval(x, root, rng, circular=True)
    log(f"{kind}: largest change of an end of the interval with circular blocks "
        f"{np.max(np.abs(np.r_[check_low - low, check_high - high])):.2f} km")
    rows = []
    for j, key in enumerate(ORDER):
        column = x[:, j]
        ok = np.isfinite(column)
        row = {"kind": kind, "detector": key, "clusters": int(ok.sum()), "mean_km": round(np.nanmean(column), 2),
               "ci_low_km": round(low[j], 2), "ci_high_km": round(high[j], 2),
               "ahead_pct": round(100 * np.mean(column[ok] > 0), 1)}
        if key != "Gn":
            diff = (x[:, 0] - column)[:, None]
            both = np.isfinite(diff[:, 0])
            diff_length = wilks_length(diff[both, 0])[3]
            d_low, d_high = interval(diff, diff_length, rng)
            c_low, c_high = interval(diff, root, rng, circular=True)
            row.update({"pairs": int(both.sum()), "diff_km": round(np.nanmean(diff), 2), "diff_low_km": round(d_low[0], 2),
                        "diff_high_km": round(d_high[0], 2)})
            log(f"{kind} Gn - {key}: {both.sum()} pairs, l {diff_length}, mean {np.nanmean(diff):.2f} "
                f"[{d_low[0]:.2f}, {d_high[0]:.2f}]; circular blocks [{c_low[0]:.2f}, {c_high[0]:.2f}]")
        rows.append(row)
    return rows, wide, (low[0], high[0], length)


def distribution(kind, wide):
    rows = []
    for key in SHOWN:
        x = wide[key].dropna().to_numpy()
        counts, _ = np.histogram(x[(x >= EDGES[0]) & (x <= EDGES[-1])], bins=EDGES)
        q20, q80 = np.percentile(x, [20, 80])
        for left, count in zip(EDGES[:-1], counts):
            rows.append({"kind": kind, "detector": key, "bin_from_km": int(left), "percent": round(100 * count / len(x), 3),
                         "q20_km": round(q20, 2), "q80_km": round(q80, 2), "mean_km": round(x.mean(), 2),
                         "below_pct": round(100 * np.mean(x < EDGES[0]), 1),
                         "above_pct": round(100 * np.mean(x > EDGES[-1]), 1)})
    return rows


def main():
    rng = np.random.default_rng(SEED)
    log(f"prepara_dati_moto.py: B = {B}, seed = {SEED}, percentile intervals at 95%")
    rows, short, short_gn = summarise("brevi", MAIN / "lead_r300.csv", rng)
    more, long, long_gn = summarise("lunghi", MAIN / "lead_r300_long.csv", rng)
    # the interval of G_n of the main run is computed once, in summarise, and reused for the column 95-esimo
    done = {MAIN / "lead_r300.csv": short_gn, MAIN / "lead_r300_long.csv": long_gn}
    pd.DataFrame(rows + more).to_csv(OUT / "era5_concordate_moto_medie.csv", index=False, lineterminator="\n")
    pd.DataFrame(distribution("brevi", short) + distribution("lunghi", long)).to_csv(
        OUT / "era5_concordate_moto_distribuzione.csv", index=False, lineterminator="\n")
    # G_n flags far from normal (an input beyond 2 standard deviations) and moderate (none), long clusters
    parts = pd.read_csv(MAIN / "lead_r300_long.csv", usecols=["detector", "matched", "along_far_km", "along_moderate_km"])
    parts = parts[(parts["matched"].astype(str) == "True") & (parts["detector"] == "Gn")]
    for name in ("along_far_km", "along_moderate_km"):
        values = parts[name].dropna()
        log(f"lunghi Gn {name}: {len(values)} clusters, mean {values.mean():.2f} km, median {values.median():.2f} km")
    runs = []
    for column, kind, relative in RUNS:
        wide = table(LOGS / relative)
        # interval of G_n, as for the main run: block length of Wilks's rule on its series, clusters by first instant
        if LOGS / relative in done:
            low, high, length = (np.array([value]) for value in done[LOGS / relative])
            length = int(length[0])
        else:
            length = wilks_length(wide["Gn"])[3]
            low, high = interval(wide[["Gn"]].to_numpy(), length, rng)
        for key in ORDER:
            x = wide[key].dropna()
            row = {"column": column, "kind": kind, "detector": key, "clusters": len(x),
                   "mean_km": round(x.mean(), 2), "median_km": round(x.median(), 2),
                   "ahead_pct": round(100 * (x > 0).mean(), 1), "file": relative}
            if key == "Gn":
                row.update({"ci_low_km": round(low[0], 2), "ci_high_km": round(high[0], 2)})
            runs.append(row)
        gn = wide["Gn"].dropna()
        log(f"Appendix A, {column}, {kind}: {len(gn)} clusters, mean {gn.mean():.2f} km [{low[0]:.2f}, {high[0]:.2f}], "
            f"Wilks l {length}, median {gn.median():.2f} km, ahead {100 * (gn > 0).mean():.1f}%; "
            f"largest mean of the nine: {wide.mean().idxmax()}")
    pd.DataFrame(runs).to_csv(OUT / "era5_concordate_moto_sensibilita.csv", index=False, lineterminator="\n")
    (OUT / "prepara_dati_moto_log.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf8", newline="\n")


if __name__ == "__main__":
    main()

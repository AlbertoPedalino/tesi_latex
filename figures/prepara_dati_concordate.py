"""Prepare the data of the figures and of the table of Section 6.7 from the outputs of the server runs, and check them
against the numbers already written in the thesis.

Run from the thesis folder:
    python figures/prepara_dati_concordate.py
which reads Tesi/log_server/r300/outputs (cube at the 95th percentile, zone of 300 km, run optuna-27, test years
2005-2025) and writes in figures/dati:
- era5_concordate_sovrapposizione.csv: median coverage, excess (eccedenza) and Jaccard index of every method in the
  zone of the short and long concordant clusters (boxes_r300.csv, boxes_r300_long.csv; eccedenza = excess /
  concordant, cluster by cluster);
- era5_concordate_rilevati.csv: share of the concordant clusters detected by every method (short, long) and share of
  its clusters without counterpart, as Bourdin and without the clusters near small ones only (match_r300.log);
- era5_concordate_giudici.csv: share of values in which each quantity that no score reads is unusual, for the
  non-concordant flags of every method ahead and behind the long concordant clusters, and for the reference
  (excess_r300_long_giudici.csv, timestamp of the value, ahead and behind weighted by their values).
Every number is compared with the one in the thesis (tables of Chapter 6 and risultati.md section 15): the script stops
if one differs by more than the rounding.
"""
import pathlib
import re

import pandas as pd

HERE = pathlib.Path(__file__).parent
RUNS = HERE.parents[1] / "log_server" / "r300" / "outputs"
CUBE = RUNS / "concordate_5pc_regola_tesi"
OUT = HERE / "dati"
METHODS = ("Gn", "change_3h", "normal", "pca", "cblof", "mcd", "hbos", "copod", "iforest")

# The numbers in the thesis before the change (Table of the overlap, 6.7.2; Table of the judges, 6.8; section 15).
OVERLAP_TABLE = {  # short: cop, ecc, jaccard; long: cop, ecc, jaccard
    "Gn": (0.69, 0.79, 0.30, 0.71, 0.36, 0.50), "normal": (0.64, 0.05, 0.47, 0.75, 0.12, 0.64),
    "pca": (0.97, 0.29, 0.71, 0.95, 0.17, 0.81), "cblof": (0.83, 0.21, 0.54, 0.80, 0.18, 0.65),
    "mcd": (0.96, 0.77, 0.48, 0.87, 0.30, 0.65), "hbos": (0.70, 0.37, 0.40, 0.85, 0.33, 0.61),
    "copod": (0.67, 0.11, 0.46, 0.82, 0.19, 0.65), "iforest": (0.87, 0.30, 0.56, 0.90, 0.25, 0.71),
    "change_3h": (0.77, 1.36, 0.24, 0.67, 0.53, 0.41)}
DETECTED_SECTION_15 = {  # short detected, long detected, without counterpart as Bourdin, without the near-small ones
    "Gn": (0.46, 0.91, 0.545, 0.225), "mcd": (0.44, 0.76, 0.537, 0.159), "change_3h": (0.32, 0.83, 0.725, 0.525),
    "pca": (0.65, 0.79, 0.254, 0.014), "cblof": (0.52, 0.80, 0.262, 0.059), "iforest": (0.55, 0.89, 0.267, 0.035),
    "hbos": (0.45, 0.89, 0.338, 0.084), "copod": (0.45, 0.92, 0.193, 0.015), "normal": (0.45, 0.81, 0.137, 0.038)}
JUDGES = ("cape highest 5%", "i10fg highest 5%", "cp highest 5%", "d2m highest 5%", "d2m lowest 5%", "kx highest 5%",
          "mn2t lowest 5%", "mx2t highest 5%", "tcwv highest 5%")
JUDGES_TABLE = {  # CAPE, gust, conv. precipitation, dew high, dew low, K, min low, max high, column vapour
    "Gn": (10.6, 7.5, 11.6, 10.5, 3.0, 15.2, 3.5, 9.6, 15.7), "normal": (6.9, 5.2, 4.5, 10.7, 10.9, 7.1, 12.0, 16.7, 7.5),
    "pca": (6.7, 18.3, 6.7, 7.8, 11.3, 8.0, 10.1, 11.3, 9.0), "cblof": (7.4, 7.3, 7.0, 10.8, 6.5, 8.2, 9.4, 10.3, 9.0),
    "mcd": (9.1, 7.2, 14.2, 9.0, 4.1, 14.3, 5.4, 7.3, 15.1), "hbos": (6.9, 21.5, 6.0, 9.3, 9.7, 8.4, 9.9, 14.0, 9.9),
    "copod": (7.8, 25.7, 9.1, 9.3, 7.1, 9.3, 10.2, 9.3, 11.5), "iforest": (6.9, 23.9, 7.2, 9.3, 8.4, 9.5, 8.3, 12.8, 11.7),
    "change_3h": (9.6, 7.1, 10.2, 9.0, 3.5, 12.3, 3.8, 8.3, 12.8),
    "reference": (5.2, 5.3, 3.8, 6.3, 5.4, 5.7, 5.5, 6.5, 6.6)}


def check(name, computed, written, places):
    if round(computed, places) != round(written, places) and abs(computed - written) > 0.5 * 10 ** -places + 1e-9:
        raise SystemExit(f"{name}: computed {computed:.4f}, written in the thesis {written}")


def overlap():
    rows = []
    for kind, path in (("brevi", CUBE / "boxes_r300.csv"), ("lunghi", CUBE / "boxes_r300_long.csv")):
        boxes = pd.read_csv(path)
        boxes["eccedenza"] = boxes.excess / boxes.concordant
        for method in METHODS:
            part = boxes[boxes.detector == method]
            rows.append({"cluster": kind, "metodo": method, "copertura": part.coverage.median(),
                         "eccedenza": part.eccedenza.median(), "jaccard": part.jaccard.median(), "cluster_n": len(part)})
    table = pd.DataFrame(rows)
    for _, row in table.iterrows():
        written = OVERLAP_TABLE[row.metodo][0 if row.cluster == "brevi" else 3:][:3]
        for value, mine, label in zip(written, (row.copertura, row.eccedenza, row.jaccard), ("cop", "ecc", "jaccard")):
            check(f"overlap {row.cluster} {row.metodo} {label}", mine, value, 2)
    table.to_csv(OUT / "era5_concordate_sovrapposizione.csv", index=False, float_format="%.4f")


def detected():
    text = (RUNS / "concordate_5pc_regola_tesi_match_r300.log").read_text(encoding="utf-8")
    rows = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) > 10 and parts[0] in METHODS and parts[1] in ("analysed", "long") and re.fullmatch(r"[\d.]+", parts[3]):
            if (parts[0], parts[1]) in rows:
                continue  # the first block is against the cube; the second, against the cube without the own vote
            rows[(parts[0], parts[1])] = (float(parts[3]), float(parts[-2]), float(parts[-1]))
    out = []
    for method in METHODS:
        short, long = rows[(method, "analysed")], rows[(method, "long")]
        record = {"metodo": method, "rilevati_brevi": short[0], "rilevati_lunghi": long[0],
                  "senza_corrispondente": short[2], "senza_corrispondente_lontani": short[1]}
        written = DETECTED_SECTION_15[method]
        check(f"detected short {method}", record["rilevati_brevi"], written[0], 2)
        check(f"detected long {method}", record["rilevati_lunghi"], written[1], 2)
        check(f"without counterpart {method}", record["senza_corrispondente"], written[2], 3)
        check(f"without counterpart, far {method}", record["senza_corrispondente_lontani"], written[3], 3)
        out.append(record)
    pd.DataFrame(out).to_csv(OUT / "era5_concordate_rilevati.csv", index=False, float_format="%.4f")


def judges():
    data = pd.read_csv(CUBE / "excess_r300_long_giudici.csv")
    data = data[data.hours == 0]
    out = []
    for method in METHODS + ("reference",):
        part = data[data.detector == method]
        record = {"metodo": method}
        for judge, written in zip(JUDGES, JUDGES_TABLE[method]):
            rows = part[part.judge == judge]
            share = (rows["values"] * rows["unusual %"]).sum() / rows["values"].sum()
            check(f"judge {method} {judge}", share, written, 1)
            record[judge] = share
        out.append(record)
    pd.DataFrame(out).to_csv(OUT / "era5_concordate_giudici.csv", index=False, float_format="%.3f")


AROUND_TABLE = {  # Table 6.6: over 2 short, long; behind the short clusters at 0 h, 24 h; jump in three hours far, moderate
    "Gn": (74.2, 78.4, 2.45, 1.91, 2.80, 2.78), "normal": (95.1, 95.0, 2.79, 2.49, 1.44, 1.42),
    "pca": (96.2, 96.0, 3.27, 2.22, 1.91, 2.46), "cblof": (90.2, 90.6, 2.76, 2.23, 1.62, 1.71),
    "mcd": (90.7, 90.3, 2.93, 2.05, 2.22, 2.28), "hbos": (94.2, 95.0, 2.66, 2.04, 1.58, 1.69),
    "copod": (92.0, 93.5, 2.61, 1.94, 1.66, 1.75), "iforest": (96.2, 95.9, 2.75, 2.02, 1.68, 1.93),
    "change_3h": (66.2, 72.1, 2.33, 1.89, 2.85, 2.71), "reference": (40.3, 47.9, 1.92, 1.86, 1.41, 1.28)}
HOURS = (0, 3, 6, 12, 24)


def weighted(frame, column):
    return (frame["values"] * frame[column]).sum() / frame["values"].sum()


def jumps(log):
    """Largest change of a standardised input in the three hours before the value, far and moderate values, ahead and
    behind weighted by their values (section 'the course before the value' of the excess log)."""
    sums = {}
    inside = False
    for line in log.read_text(encoding="utf-8").splitlines():
        if "the course before the value" in line:
            inside = True
            continue
        if inside and not line.strip():
            if sums:
                break
            continue
        parts = line.split()
        if inside and len(parts) >= 9 and parts[0] in METHODS + ("reference",) and parts[1] in ("ahead", "behind"):
            key = (parts[0], parts[2])
            total, count = sums.get(key, (0.0, 0))
            sums[key] = (total + int(parts[3]) * float(parts[4]), count + int(parts[3]))
    return {key: total / count for key, (total, count) in sums.items()}


def around():
    """Table 6.6 as data: share of values with a variable beyond two standard deviations, the largest standardised
    distance behind the short clusters from 0 to 24 hours after, and the jump in three hours before."""
    short, long = pd.read_csv(CUBE / "excess_r300_analysed.csv"), pd.read_csv(CUBE / "excess_r300_long.csv")
    jump = jumps(RUNS / "concordate_5pc_regola_tesi_excess_r300_analysed.log")
    rows = []
    for method in METHODS + ("reference",):
        def pick(frame):
            part = frame[(frame.detector == method) & frame.position.isin(["ahead", "behind"])]
            return part if method == "reference" else part[part.who.isin(["alone", "with others"])]
        s, l = pick(short), pick(long)
        record = {"metodo": method, "oltre2_brevi": weighted(s[s.hours == 0], "any > 2 %"),
                  "oltre2_lunghi": weighted(l[l.hours == 0], "any > 2 %"),
                  **{f"dietro_{h}h": weighted(s[(s.position == "behind") & (s.hours == h)], "max |z|") for h in HOURS},
                  "salto_lontani": jump[(method, "far")], "salto_moderati": jump[(method, "moderate")]}
        written = AROUND_TABLE[method]
        for value, key, places in zip(written, ("oltre2_brevi", "oltre2_lunghi", "dietro_0h", "dietro_24h",
                                                "salto_lontani", "salto_moderati"), (1, 1, 2, 2, 2, 2)):
            check(f"around {method} {key}", record[key], value, places)
        rows.append(record)
    pd.DataFrame(rows).to_csv(OUT / "era5_concordate_attorno.csv", index=False, float_format="%.4f")


LEVELS_TABLE = {  # Table 6.7: start (ahead/behind), end (later/earlier), motion (ahead/behind); short then long
    0.6: ((3, 8), (3, 11), (51, 46), (3, 12), (5, 16), (54, 46)),
    0.7: ((4, 13), (4, 18), (53, 46), (6, 19), (6, 28), (56, 44)),
    0.8: ((5, 18), (5, 25), (54, 46), (8, 26), (6, 37), (56, 44)),
    0.9: ((7, 22), (5, 32), (54, 45), (10, 32), (7, 46), (56, 44)),
    0.95: ((9, 24), (7, 36), (55, 45), (11, 34), (8, 50), (56, 44))}


def levels():
    """The flags of G_n with the probability of exceedance at least each level against their control (Table 6.7): start
    and end from match_r300_p2_levels.csv (run _p2, checked here), motion as written in the table (lead logs _p2,
    risultati.md section 16 bis)."""
    data = pd.read_csv(HERE.parents[1] / "log_server" / "r300_ns_p2" / "outputs" / "concordate_5pc_regola_tesi"
                       / "match_r300_p2_levels.csv")
    rows = []
    for level, written in LEVELS_TABLE.items():
        for k, kind in enumerate(("brevi", "lunghi")):
            part = data[(data.kind == ("analysed" if kind == "brevi" else "long")) & (data.level == f"Gn_p{level:g}")]
            start = part[part["level first"] == "starts earlier"].iloc[0]
            end = part[part["level first"] == "ends later"].iloc[0]
            start_w, end_w, motion_w = written[3 * k:3 * k + 3]
            check(f"level {level} {kind} start earlier", start["first ahead %"], start_w[0], 0)
            check(f"level {level} {kind} start later", start["first behind %"], start_w[1], 0)
            check(f"level {level} {kind} end later", end["first ahead %"], end_w[0], 0)
            check(f"level {level} {kind} end earlier", end["first behind %"], end_w[1], 0)
            rows.append({"livello": level, "cluster": kind, "inizio_prima": start["first ahead %"],
                         "inizio_dopo": start["first behind %"], "fine_dopo": end["first ahead %"],
                         "fine_prima": end["first behind %"], "moto_avanti": motion_w[0], "moto_dietro": motion_w[1]})
    pd.DataFrame(rows).to_csv(OUT / "era5_concordate_livelli.csv", index=False, float_format="%.2f")


if __name__ == "__main__":
    overlap()
    detected()
    judges()
    around()
    levels()
    print("all numbers match the thesis")

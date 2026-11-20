"""Write figures/dati/era5_concordate_moto_quartili.csv: for every method and for the short and the long concordant
clusters, the quartiles of the along-motion displacement of its flags over the matched concordant clusters.

Run from the thesis folder:
    python figures/prepara_dati_moto.py [OUTPUTS]
OUTPUTS defaults to ../log_server/r300/outputs/concordate_5pc_regola_tesi, the copy of the server folder with
lead_r300.csv (short clusters) and lead_r300_long.csv (long clusters), written by scripts/check_era5_concordant_lead.py
of the code repository (zone of 300 km, cube at the 95th percentile; run optuna-27, test years 2005-2025).

Rows: the matched concordant clusters (column matched) with a displacement (along_km not missing: a method with no flag
in the zone of the cluster has none); the rows of the probability levels (Gn_p..., Gn_rand...) are left out. The medians
and the share ahead are those of the former file era5_concordate_moto_r300.csv and of the text.
"""
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).parent
DEFAULT = HERE.parent.parent / "log_server" / "r300" / "outputs" / "concordate_5pc_regola_tesi"
OUT = HERE / "dati" / "era5_concordate_moto_quartili.csv"


def quartiles(path, kind):
    frame = pd.read_csv(path, usecols=["detector", "matched", "along_km"])
    frame = frame[frame["matched"].astype(str) == "True"]
    frame = frame[~frame["detector"].str.startswith("Gn_")].dropna(subset=["along_km"])
    rows = []
    for detector, part in frame.groupby("detector"):
        along = part["along_km"]
        rows.append({"kind": kind, "detector": detector, "clusters": len(along),
                     "q25_km": round(along.quantile(0.25), 2), "median_km": round(along.median(), 2),
                     "q75_km": round(along.quantile(0.75), 2), "ahead_pct": round(100 * (along > 0).mean(), 1)})
    return rows


def main():
    folder = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    rows = quartiles(folder / "lead_r300.csv", "brevi") + quartiles(folder / "lead_r300_long.csv", "lunghi")
    table = pd.DataFrame(rows)
    header = ("# Quartili sui cluster concordati abbinati di along_km (righe matched con along_km, esclusi Gn_p e Gn_rand)\n"
              "# da Tesi/log_server/r300/outputs/concordate_5pc_regola_tesi/lead_r300.csv (brevi) e lead_r300_long.csv\n"
              "# (lunghi), intorno di 300 km, 5%; scritto da figures/prepara_dati_moto.py.\n")
    OUT.write_text(header + table.to_csv(index=False), encoding="utf8", newline="\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

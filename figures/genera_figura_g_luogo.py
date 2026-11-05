"""Generate the figure of the climatology of the score: where and when the generator's score (G) flags, without
and with the climatology of the score.

Share of the flagged values of the score, by latitude band and calendar month, on the validation years
2000-2004 of the final run (figures/dati/era5_g_luogo.json, from scripts/check_era5_score_by_place.py of the code
repository, run by scripts/run_era5_score_behaviour.sh). The threshold of each score is its 95th percentile over the validation years, as everywhere in
the thesis, so that both scores flag 5% on average.

Run from the thesis folder. First, once, gather the data from the logs of the behaviour output (its folder logs/,
with luogo_g_grezzo_5pc.log and luogo_g_clim_5pc.log):
    python figures/genera_figura_g_luogo.py --prepare <behaviour output folder>
Then:
    python figures/genera_figura_g_luogo.py
which writes figures/era5_g_luogo.pdf.
"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

HERE = pathlib.Path(__file__).parent
DATA = HERE / "dati" / "era5_g_luogo.json"
WIDTH_CM = 14.0
TEXT, MUTED = "#0b0b0b", "#52514e"
MONTHS = ("gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "font.size": 9,
    "pdf.fonttype": 42,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


def read_log(path):
    """Bands, cells and the share of flagged values by band and month from a log of check_era5_score_by_place.py."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("latitude"))
    bands, cells, shares = [], [], []
    for line in lines[start + 1:]:
        fields = line.split()
        if len(fields) != 14:
            break
        bands.append(fields[0])
        cells.append(int(fields[1]))
        shares.append([float(value) for value in fields[2:]])
    return bands, cells, shares


def prepare(folder):
    folder = pathlib.Path(folder)
    bands, cells, raw = read_log(folder / "logs" / "luogo_g_grezzo_5pc.log")
    clim_bands, _, clim = read_log(folder / "logs" / "luogo_g_clim_5pc.log")
    if clim_bands != bands:
        raise ValueError("The two logs are not over the same bands")
    DATA.write_text(json.dumps({"source": f"scripts/check_era5_score_by_place.py, logs of {folder.name}", "top_percent": 5,
                                "bands": bands, "cells": cells, "G": raw, "G_clim": clim}, indent=2),
                    encoding="utf-8")
    print(f"written {DATA}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=pathlib.Path)
    args = parser.parse_args()
    if args.prepare is not None:
        prepare(args.prepare)
        return
    data = json.loads(DATA.read_text(encoding="utf-8"))
    top = data["top_percent"]
    panels = (("G", "G senza normalizzazione"), ("G_clim", "G con normalizzazione (giorno ±15 × ora)"))
    norm = TwoSlopeNorm(vmin=0, vcenter=top, vmax=15)
    width = WIDTH_CM / 2.54
    figure, axes = plt.subplots(2, 1, figsize=(width, 0.62 * width), constrained_layout=True)
    for axis, (key, title) in zip(axes, panels):
        values = np.array(data[key])
        mesh = axis.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
        for (row, column), value in np.ndenumerate(values):
            dark = abs(value - top) > 4
            axis.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=6.5,
                      color="white" if dark else TEXT)
        axis.set_xticks(range(12), MONTHS)
        axis.set_yticks(range(len(data["bands"])), [band.replace("N", "°N") for band in data["bands"]])
        axis.tick_params(length=0)
        axis.set_title(title, fontsize=9, loc="left", color=TEXT)
    bar = figure.colorbar(mesh, ax=axes, shrink=0.9, ticks=[0, top, 10, 15])
    bar.set_label("valori segnalati (%)")
    bar.outline.set_edgecolor(MUTED)
    figure.savefig(HERE / "era5_g_luogo.pdf")
    print(f"written {HERE / 'era5_g_luogo.pdf'}")


if __name__ == "__main__":
    main()

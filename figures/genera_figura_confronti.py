"""Generate the figure of the comparison of the detectors: the Jaccard index of the flags of every pair.

Highest 5% of each score, with its threshold from the validation years, on the test years 2005-2025 of
the final run (figures/dati/era5_confronti.json, from scripts/compare_era5_detectors.py of the code repository).

Run from the thesis folder:
    python figures/genera_figura_confronti.py
which writes figures/era5_confronti_jaccard.pdf.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
DATA = HERE / "dati" / "era5_confronti.json"
WIDTH_CM = 12.0
TEXT, MUTED = "#0b0b0b", "#52514e"
LABELS = {"G+clim": "G con norm.", "change_3h": "Variazione 3 ore", "normal": "Distanza dal normale", "cblof": "CBLOF",
          "mcd": "MCD", "pca": "PCA", "iforest": "Isolation Forest", "copod": "COPOD", "hbos": "HBOS"}

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


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    values = np.array(data["jaccard_5pc"], dtype=float)
    names = [LABELS[name] for name in data["detectors"]]
    width = WIDTH_CM / 2.54
    figure, axis = plt.subplots(figsize=(width, 0.8 * width), constrained_layout=True)
    mesh = axis.imshow(values, cmap="viridis", vmin=10, vmax=70)
    for (row, column), value in np.ndenumerate(values):
        if np.isfinite(value):
            axis.text(column, row, f"{np.floor(value + 0.5):.0f}", ha="center", va="center", fontsize=7.5,
                      color="white" if value < 45 else TEXT)
    axis.set_xticks(range(len(names)), names, rotation=40, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.tick_params(length=0)
    for side in axis.spines.values():
        side.set_visible(False)
    bar = figure.colorbar(mesh, ax=axis, shrink=0.8)
    bar.set_label("indice di Jaccard (%)")
    bar.outline.set_edgecolor(MUTED)
    figure.savefig(HERE / "era5_confronti_jaccard.pdf")
    print(f"written {HERE / 'era5_confronti_jaccard.pdf'}")


if __name__ == "__main__":
    main()

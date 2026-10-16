"""Generate the ERA5 result figures of chapter 6 from the values reported in the thesis.

Run from the thesis folder:
    python figures/genera_figure_era5.py

Writes figures/era5_sovrapposizione.pdf and figures/era5_eobs.pdf.

The values below were read on 15 October 2026 from the run
..._wdaily_seasonal_trend_ep3 (sources in chapters/risultati_fonti.tex): the
overlap from the notebook era5_stgan_ep3_risultati.ipynb, section 1 (one
timestamp every 29); the E-OBS ratios from the *_intervalli.log files of
scripts/check_era5_score_against_eobs.py. They are provisional in the same sense
as the numbers marked \\provv in the text: when the final run is scored, replace
the values here and run the script again.
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
WIDTH_CM = 13.2
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e2e1dc"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.5,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def save(figure, name):
    figure.savefig(HERE / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print("written", name)


# --------------------------------------------------------------------------
# Overlap between the four detectors: share (%) of the flags of the row that
# the column flags too, at 5% and at 1% of the observations.
# --------------------------------------------------------------------------
DETECTORS = ["detector", "distanza\ndal normale", "variazione\nin 3 h", "maggiore\ndei due"]
OVERLAP = {
    "5%": [[100, 50, 52, 58],
             [50, 100, 34, 69],
             [52, 34, 100, 62],
             [58, 69, 62, 100]],
    "1%": [[100, 43, 44, 51],
             [43, 100, 28, 57],
             [44, 28, 100, 69],
             [51, 57, 69, 100]],
}


def figure_overlap():
    figure, panels = plt.subplots(1, 2, figsize=(WIDTH_CM / 2.54, 6.6 / 2.54))
    colours = matplotlib.colors.LinearSegmentedColormap.from_list("blu", ["#ffffff", BLUE])
    for axes, (level, values) in zip(panels, OVERLAP.items()):
        matrix = np.array(values, dtype=float)
        axes.imshow(matrix, cmap=colours, vmin=0, vmax=100)
        for row in range(4):
            for column in range(4):
                value = matrix[row, column]
                axes.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=8,
                          color="white" if value > 65 else TEXT)
        axes.set_xticks(range(4), DETECTORS, fontsize=6.5, rotation=0)
        axes.set_yticks(range(4), DETECTORS if axes is panels[0] else [""] * 4, fontsize=6.5)
        axes.tick_params(length=0)
        for side in axes.spines.values():
            side.set_visible(False)
        axes.set_title(f"Segnalate al {level}", fontsize=9, color=TEXT)
    figure.tight_layout(w_pad=1.0)
    save(figure, "era5_sovrapposizione.pdf")


# --------------------------------------------------------------------------
# E-OBS confirmation: ratio R with its 95% interval over 2000 draws of whole
# years, for four indicators, the same day and the next day.
# --------------------------------------------------------------------------
INDICATORS = ["pioggia forte", "pressione bassa", "calo di pressione", "radiazione bassa"]
COMPARISONS = (  # label, colour, marker
    ("solo detector / tutte", BLUE, "o"),
    ("solo detector / anello", ORANGE, "s"),
    ("solo detector / solo rilevatore", AQUA, "^"),
    ("entrambi / tutte", PURPLE, "D"),
)
# RATIOS[comparison][indicator] = ((same day: value, low, high), (next day: value, low, high))
RATIOS = {
    "solo detector / tutte": [((1.17, 1.12, 1.21), (1.44, 1.40, 1.47)),
                              ((1.34, 1.27, 1.40), (1.37, 1.32, 1.43)),
                              ((1.16, 1.11, 1.22), (0.86, 0.83, 0.89)),
                              ((1.04, 1.00, 1.07), (1.18, 1.14, 1.21))],
    "solo detector / anello": [((1.58, 1.55, 1.62), (1.21, 1.19, 1.22)),
                               ((1.35, 1.32, 1.37), (1.19, 1.17, 1.22)),
                               ((1.27, 1.25, 1.29), (0.94, 0.92, 0.96)),
                               ((1.22, 1.20, 1.25), (1.08, 1.06, 1.10))],
    "solo detector / solo rilevatore": [((1.01, 0.98, 1.05), (1.25, 1.22, 1.28)),
                                        ((1.14, 1.09, 1.19), (1.13, 1.10, 1.17)),
                                        ((1.00, 0.97, 1.04), (0.83, 0.80, 0.87)),
                                        ((0.80, 0.76, 0.83), (1.06, 1.02, 1.11))],
    "entrambi / tutte": [((3.61, 3.48, 3.74), (2.11, 2.04, 2.17)),
                         ((3.04, 2.95, 3.12), (2.19, 2.11, 2.26)),
                         ((2.46, 2.39, 2.53), (0.79, 0.75, 0.83)),
                         ((2.25, 2.16, 2.34), (1.41, 1.36, 1.46))],
}


def figure_eobs():
    figure, panels = plt.subplots(1, 2, figsize=(WIDTH_CM / 2.54, 6.4 / 2.54), sharey=True)
    positions = np.arange(len(INDICATORS))
    offsets = np.linspace(-0.27, 0.27, len(COMPARISONS))
    for day, (axes, title) in enumerate(zip(panels, ("Stesso giorno", "Giorno successivo"))):
        axes.axvline(1, color=MUTED, linewidth=0.8)
        for offset, (label, colour, marker) in zip(offsets, COMPARISONS):
            values = np.array([RATIOS[label][index][day] for index in range(len(INDICATORS))])
            centre, low, high = values[:, 0], values[:, 1], values[:, 2]
            axes.errorbar(centre, positions + offset, xerr=[centre - low, high - centre], fmt=marker, color=colour,
                          markersize=3.8, markeredgecolor="white", markeredgewidth=0.5, elinewidth=0.9, capsize=1.5,
                          label=label)
        axes.set_xscale("log")
        axes.set_xticks([0.8, 1, 1.5, 2, 3], ["0.8", "1", "1.5", "2", "3"])
        axes.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes.set_xlabel("Rapporto $R$")
        axes.set_title(title, fontsize=9, color=TEXT)
        axes.set_yticks(positions, INDICATORS, fontsize=8)
        if day == 0:  # the axis is shared: invert it once, first indicator on top
            axes.invert_yaxis()
        axes.grid(axis="x", color=GRID, linewidth=0.5)
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
    handles, labels = panels[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=7.5, labelcolor=TEXT,
                  bbox_to_anchor=(0.55, -0.12), handlelength=1.2)
    figure.tight_layout(w_pad=1.2)
    save(figure, "era5_eobs.pdf")


if __name__ == "__main__":
    figure_overlap()
    figure_eobs()

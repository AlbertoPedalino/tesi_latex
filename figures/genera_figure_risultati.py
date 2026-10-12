"""Generate the result figures of chapter 6 from the values reported in the thesis.

Run from the thesis folder:
    python figures/genera_figure_risultati.py

Writes figures/errore_orizzonti.pdf, figures/sensibilita_soglia.pdf and
figures/quattro_casi.pdf.

The values below were read on the server on 11 October 2026; file names and
runs are listed in chapters/risultati_fonti.tex. They come from the existing
six-horizon SDE-Net run, so they are provisional in the same sense as the
numbers marked \\provv in the text: when the run is redone, replace the values
here and run the script again.
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
WIDTH_CM = 13.2
# Categorical colours checked for colour-vision deficiencies; markers differ too,
# so the series stay distinct in a greyscale print.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
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


def clean(axes, grid_axis="y"):
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.grid(axis=grid_axis, color=GRID, linewidth=0.5)
    axes.set_axisbelow(True)


def line(axes, x, y, colour, marker, label):
    axes.plot(x, y, color=colour, linewidth=1.4, marker=marker, markersize=4.5,
              markeredgecolor="white", markeredgewidth=0.7, label=label)


def save(figure, name):
    figure.savefig(HERE / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print("written", name)


# --------------------------------------------------------------------------
# Error by horizon, normal and flagged observations (MTGFlow), daytime hours.
# Source: multihorizon_anomaly_group_metrics.csv.
# --------------------------------------------------------------------------
HORIZONS = np.arange(1, 7)
MAE_NORMAL = [45.186, 57.078, 66.278, 72.406, 78.014, 81.836]
MAE_FLAGGED = [51.976, 63.902, 73.171, 79.428, 85.550, 88.910]
RMSE_NORMAL = [70.646, 85.859, 96.641, 104.545, 110.650, 114.775]
RMSE_FLAGGED = [82.279, 98.722, 108.852, 116.576, 122.021, 124.854]


def figure_horizons():
    figure, panels = plt.subplots(1, 2, figsize=(WIDTH_CM / 2.54, 5.6 / 2.54))
    for axes, normal, flagged, name in ((panels[0], MAE_NORMAL, MAE_FLAGGED, "MAE (W)"),
                                        (panels[1], RMSE_NORMAL, RMSE_FLAGGED, "RMSE (W)")):
        line(axes, HORIZONS, normal, BLUE, "o", "normali")
        line(axes, HORIZONS, flagged, ORANGE, "s", "segnalate da MTGFlow")
        axes.set_xlabel("Orizzonte (ore)")
        axes.set_ylabel(name)
        axes.set_xticks(HORIZONS)
        clean(axes)
    panels[1].legend(loc="lower right", handlelength=1.6, labelcolor=TEXT)
    figure.tight_layout(w_pad=2.0)
    save(figure, "errore_orizzonti.pdf")


# --------------------------------------------------------------------------
# Ratio between the error on flagged and on normal observations against the
# share of daytime observations flagged. Source:
# detector_threshold_sensitivity_metrics.csv of the two sensitivity analyses.
# --------------------------------------------------------------------------
MTGFLOW = {
    "share": [15.92, 9.83, 6.60, 4.69, 3.52, 2.76],          # k = 0.5 ... 3.0
    1: [1.1295, 1.1374, 1.1345, 1.1298, 1.1279, 1.1379],
    6: [1.0683, 1.0792, 1.0838, 1.0823, 1.0823, 1.0922],
    "reference": 2,                                           # k = 1.5
}
STGAN_GRAPH = {
    "share": [0.51, 1.01, 1.51, 2.00, 2.49, 2.96, 3.88, 5.49, 6.74, 7.66],   # top 0.25% ... 5%
    1: [1.6930, 1.5181, 1.4232, 1.3586, 1.3113, 1.2736, 1.2178, 1.1418, 1.0994, 1.0721],
    6: [1.2369, 1.1815, 1.1579, 1.1391, 1.1273, 1.1186, 1.1120, 1.1025, 1.0995, 1.0986],
    "reference": 3,                                           # top 1%
}
STGAN_CONVGRU = {
    "share": [0.51, 1.01, 1.50, 1.98, 2.45, 2.92, 3.86, 5.73, 7.58, 9.39],
    1: [1.1748, 1.0997, 1.0540, 1.0243, 1.0012, 0.9839, 0.9599, 0.9252, 0.8993, 0.8768],
    6: [0.9811, 0.9419, 0.9181, 0.9013, 0.8902, 0.8831, 0.8722, 0.8629, 0.8607, 0.8621],
    "reference": 3,
}
DETECTORS = (("MTGFlow", MTGFLOW, BLUE, "o"),
             ("STGAN su grafo", STGAN_GRAPH, ORANGE, "s"),
             ("STGAN con ConvGRU", STGAN_CONVGRU, AQUA, "^"))


def figure_sensitivity():
    figure, panels = plt.subplots(1, 2, figsize=(WIDTH_CM / 2.54, 6.2 / 2.54), sharey=True)
    for axes, horizon, title in ((panels[0], 1, "Previsione a un'ora"), (panels[1], 6, "Previsione a sei ore")):
        axes.axhline(1.0, color=MUTED, linewidth=0.7)
        for label, data, colour, marker in DETECTORS:
            line(axes, data["share"], data[horizon], colour, marker, label)
            reference = data["reference"]
            axes.plot(data["share"][reference], data[horizon][reference], marker=marker, markersize=7.5,
                      markerfacecolor="none", markeredgecolor=TEXT, markeredgewidth=0.8, linestyle="none")
        axes.set_title(title, fontsize=9, color=TEXT)
        axes.set_xlabel("Osservazioni diurne segnalate (%)")
        axes.set_xlim(0, 17)
        axes.set_xticks([0, 4, 8, 12, 16])
        clean(axes)
    panels[0].set_ylabel("MAE segnalate / MAE normali")
    panels[0].set_ylim(0.8, 1.75)
    panels[1].legend(loc="upper right", handlelength=1.6, labelcolor=TEXT)
    figure.tight_layout(w_pad=1.5)
    save(figure, "sensibilita_soglia.pdf")


# --------------------------------------------------------------------------
# The four input-target cases with STGAN decisions, daytime hours.
# Source: input_target_overall_metrics.csv.
# --------------------------------------------------------------------------
CASES = ["ingresso normale, target normale", "ingresso anomalo, target normale",
         "ingresso normale, target anomalo", "ingresso anomalo, target anomalo"]
CASE_MAE = {1: [42.902, 50.326, 59.421, 62.205], 6: [79.675, 90.199, 92.733, 95.190]}


def figure_cases():
    figure, panels = plt.subplots(1, 2, figsize=(WIDTH_CM / 2.54, 4.4 / 2.54), sharey=True)
    positions = np.arange(len(CASES))[::-1]
    for axes, horizon, title in ((panels[0], 1, "Previsione a un'ora"), (panels[1], 6, "Previsione a sei ore")):
        values = CASE_MAE[horizon]
        axes.barh(positions, values, height=0.55, color=BLUE)
        for position, value in zip(positions, values):
            axes.text(value + 1.5, position, "%.1f" % value, ha="left", va="center", color=TEXT, fontsize=8)
        axes.set_title(title, fontsize=9, color=TEXT)
        axes.set_xlabel("MAE (W)")
        axes.set_xlim(0, 112)
        clean(axes, grid_axis="x")
    panels[0].set_yticks(positions)
    panels[0].set_yticklabels(CASES, color=TEXT)
    figure.tight_layout(w_pad=1.5)
    save(figure, "quattro_casi.pdf")


# --------------------------------------------------------------------------
# One week of forecasts with the 95% interval, one location, one and six hours.
# Source: figures/dati/previsioni_settimana_maggio.csv (extract of predictions.csv).
# --------------------------------------------------------------------------
def figure_week():
    rows = [line.strip().split(",") for line in open(HERE / "dati" / "previsioni_settimana_maggio.csv", encoding="utf-8")
            if line[:2].isdigit()]
    figure, panels = plt.subplots(2, 1, figsize=(WIDTH_CM / 2.54, 9.4 / 2.54), sharex=True, sharey=True)
    for axes, horizon, title in ((panels[0], "1", "Previsione a un'ora"), (panels[1], "6", "Previsione a sei ore")):
        selected = [row for row in rows if row[1] == horizon]
        hours = np.arange(len(selected))
        observed, predicted, lower, upper = (np.array([float(row[column]) for row in selected]) for column in (2, 3, 4, 5))
        flagged = np.array([row[6] != "normal" for row in selected])
        axes.fill_between(hours, lower, upper, color=BLUE, alpha=0.16, linewidth=0, label="intervallo al 95%")
        axes.plot(hours, predicted, color=BLUE, linewidth=1.2, label="previsione")
        axes.plot(hours, observed, color=TEXT, linewidth=1.0, label="produzione osservata")
        axes.plot(hours[flagged], observed[flagged], linestyle="none", marker="s", markersize=4.5, color=ORANGE,
                  markeredgecolor="white", markeredgewidth=0.6, label="segnalata da MTGFlow", zorder=5)
        axes.axhline(0.0, color=MUTED, linewidth=0.5)
        axes.set_title(title, fontsize=9, color=TEXT)
        axes.set_ylabel("Produzione (W)")
        axes.set_xlim(0, len(selected) - 1)
        clean(axes)
    panels[1].set_xticks(np.arange(12, 168, 24))
    panels[1].set_xticklabels([str(day) for day in range(9, 16)])
    panels[1].tick_params(axis="x", length=0)
    panels[1].set_xlabel("Giorno di maggio 2019")
    for axes in panels:
        for boundary in range(24, 168, 24):
            axes.axvline(boundary, color=GRID, linewidth=0.5, zorder=0)
    handles, labels = panels[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, handlelength=1.4, columnspacing=2.0,
                  labelcolor=TEXT, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    figure.tight_layout(h_pad=1.0, rect=(0, 0, 1, 0.9))
    save(figure, "previsioni_settimana.pdf")


if __name__ == "__main__":
    figure_horizons()
    figure_sensitivity()
    figure_cases()
    figure_week()

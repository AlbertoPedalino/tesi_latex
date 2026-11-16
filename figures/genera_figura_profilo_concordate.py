"""Figure of Appendix B: the profile of every score across the concordant clusters, short and long.

The data are figures/dati/profilo_r6.csv, written on the server by
    scripts/check_era5_event_profile.py --cube outputs/concordate_5pc_regola_tesi --inputs-run outputs/stgan_optuna27_scored
        --limit 6 --csv outputs/concordate_5pc_regola_tesi/profilo_r6.csv
(physiq-pv a7dd5d69, test years 2005-2025): for every kind of concordant cluster and every position, in cells of the
grid from the border of the cluster (negative inside, the last position holding 6 cells or more), the median percentile
of every score among all its values of the test years. A description only: inside the cube at least five of the nine
methods are above their threshold by construction. Run from tesi_latex/figures:

    python genera_figura_profilo_concordate.py

which writes figures/era5_concordate_profilo.pdf. Order and colours of the methods from figures/stile.py.
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from genera_figure_eventi import position_axis, spaced
from genera_figure_g_d import TICKS, size
from stile import BLUE, COLOURS, GREEN_TEXT, MUTED, ORANGE, SHALLOW, SHALLOW_GREY, TEXT, key_of

HERE = pathlib.Path(__file__).parent
PROFILE = HERE / "dati" / "profilo_r6.csv"
LIMIT = 6
KINDS = (("short", "cluster concordati brevi"), ("long", "cluster concordati lunghi"))


def lines_of(table, kind):
    """The median percentile of every method, by position, for one kind of concordant cluster."""
    part = table[table["type"] == kind].set_index("position").sort_index()
    return {key_of(column[:-len(" percentile")]): part[column] for column in part.columns if column.endswith(" percentile")}


def stacked_labels(axis, bottom, labels, fontsize=7):
    """Names right of the points at six or more cells: the first (G_n, the lowest point) at the height of its point, the
    others stacked above it in the order given, so that none is pushed below its point."""
    figure = axis.figure
    figure.canvas.draw()
    low, high = axis.get_ylim()
    line_height = 1.2 * fontsize * figure.dpi / 72 / (axis.get_window_extent().height / (high - low))
    place = float(bottom) - line_height / 2  # the lower edge of the first name
    for label, colour in labels:
        height = (label.count("\n") + 1) * line_height
        axis.annotate(label, (1, place + height / 2), xycoords=("axes fraction", "data"), xytext=(9, 0),
                      textcoords="offset points", fontsize=fontsize, color=colour, ha="left", va="center",
                      linespacing=1.0, annotation_clip=False)
        place += height


def main():
    table = pd.read_csv(PROFILE)
    figure, axes = plt.subplots(1, 2, figsize=size(0.56), constrained_layout=True, sharey=True)
    spaced(figure, 0.18)
    for axis, (kind, title) in zip(axes, KINDS):
        lines = lines_of(table, kind)
        axis.axhline(50, color=MUTED, linewidth=0.8, linestyle="--")
        axis.axhline(95, color=MUTED, linewidth=0.8, linestyle=":")
        position_axis(axis)
        axis.set_xlim(-LIMIT - 1.3, LIMIT + 1.3)
        axis.set_xticks([-6, -4, -2, 2, 4, 6], ["≥6", "4", "2", "2", "4", "≥6"])
        axis.set_xlabel("celle dal bordo: dentro | fuori", fontsize=TICKS)
        axis.text(-LIMIT - 1.1, 51, "valore tipico", fontsize=TICKS - 1, color=MUTED, va="bottom")
        axis.text(LIMIT + 1.2, 96.5, "95-esimo\npercentile", fontsize=TICKS - 1, color=MUTED, ha="right", va="bottom",
                  linespacing=0.95)
        axis.text(0.15, 3, "bordo", fontsize=TICKS - 1, color=TEXT)
        axis.text(0.5, 1.02, title, transform=axis.transAxes, ha="center", va="bottom", fontsize=TICKS, color=TEXT)
        band = pd.concat([lines[key] for key in SHALLOW], axis=1)
        near = band.index < LIMIT  # the last position holds 6 cells or more: a point of its own, not joined
        axis.fill_between(band.index[near], band.min(axis=1)[near], band.max(axis=1)[near], color=SHALLOW_GREY,
                          alpha=0.55, linewidth=0, zorder=1)
        axis.vlines(LIMIT, band.min(axis=1).iloc[-1], band.max(axis=1).iloc[-1], color=SHALLOW_GREY, linewidth=4,
                    zorder=1)
        for key, width in (("change_3h", 1.6), ("normal", 1.6), ("Gn", 2.6)):  # G_n last, on top
            line = lines[key]
            axis.plot(line.index[line.index < LIMIT], line[line.index < LIMIT], color=COLOURS[key], linewidth=width,
                      zorder=3 + (key == "Gn"))
            axis.plot([LIMIT], [line.iloc[-1]], "o", color=COLOURS[key], markersize=4 + (key == "Gn"),
                      zorder=3 + (key == "Gn"))
        axis.set_ylim(0, 104)
        axis.set_yticks([0, 25, 50, 75, 100])
        last = lambda key: lines[key].iloc[-1]
        stacked_labels(axis, last("Gn"), ((r"$G_\mathrm{n}$", BLUE), ("sei modelli\nshallow", "#8a8880"),
                                          ("distanza\ndal normale", ORANGE), ("variazione\nin tre ore", GREEN_TEXT)))
    axes[0].set_ylabel("percentile mediano", fontsize=TICKS)
    figure.savefig(HERE / "era5_concordate_profilo.pdf")
    plt.close(figure)
    print(f"written era5_concordate_profilo.pdf in {HERE}")


if __name__ == "__main__":
    main()

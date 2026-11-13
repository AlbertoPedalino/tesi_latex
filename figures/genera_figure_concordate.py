"""Generate the figures of Section 6.8: the methods around the concordant clusters.

Run from the thesis folder:
    python figures/genera_figure_concordate.py
which writes figures/era5_concordate_moto_a.pdf and _b.pdf from figures/dati/era5_concordate_moto.csv.
and figures/era5_concordate_pr_a.pdf and _b.pdf (Appendix A.3) from figures/dati/era5_concordate_pr.csv: precision and
recall of every detector against the concordant cube, and against the cube without its own vote, as its own threshold
goes from the 80th to the 99.9th percentile of the validation years (scripts/check_era5_concordant_pr.py of the code
repository, commit 576547c, run of 13 November 2026; the cube fixed at the 95th percentile).

Data: the median along-track displacement (km, positive = ahead along the motion of the cluster) of the flags of the
overlapping cluster of each method, by class of the mean speed of the concordant cluster (cells of 0.5 degrees every
3 h), printed by scripts/check_era5_concordant_lead.py of the code repository ("median along-track error (km) by mean
speed of the cluster"), on outputs/concordate_5pc_regola_tesi with --kind analysed (short clusters: 8142 / 7646 / 7651 /
3043 clusters by class) and --kind long (long clusters: 916 / 670 / 508 / 48). Run optuna-27, test years 2005-2025,
thresholds at the 95th percentile of the validation years; logs of 12 November 2026.
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pannelli import save_panels

from genera_figure_eventi import (BLUE, GREEN, GREEN_TEXT, GRID, MUTED, ORANGE, SHALLOW_GREY, SHALLOW_KEYS, TEXT, TICKS,
                                  clean, end_labels, size, spaced)

HERE = pathlib.Path(__file__).parent
MOTION = HERE / "dati" / "era5_concordate_moto.csv"
PR = HERE / "dati" / "era5_concordate_pr.csv"
CLASSES = ("0-0.5", "0.5-1", "1-2", ">=2")
CLASS_LABELS = ("0–0.5", "0.5–1", "1–2", "2 o più")


def motion_figure():
    """Left: on the short clusters G_n is a little behind, more in the fast clusters, the change in three hours more.
    Right: on the long clusters G_n is ahead along the motion, more than every baseline but in the few fastest."""
    table = pd.read_csv(MOTION)
    figure, panels = plt.subplots(1, 2, figsize=size(0.5), constrained_layout=True, sharey=False)
    spaced(figure, 0.2)
    x = range(len(CLASSES))
    for axis, kind in zip(panels, ("brevi", "lunghi")):
        part = table[table["kind"] == kind].set_index("detector")[list(CLASSES)]
        axis.axhline(0, color=TEXT, linewidth=0.8)
        shallow = part.loc[list(SHALLOW_KEYS)]
        axis.fill_between(x, shallow.min(), shallow.max(), color=SHALLOW_GREY, alpha=0.55, linewidth=0, zorder=1)
        for key, colour, width in (("change_3h", GREEN, 1.6), ("normal", ORANGE, 1.6), ("Gn", BLUE, 2.6)):
            axis.plot(x, part.loc[key], color=colour, linewidth=width, marker="o", markersize=4 if key == "Gn" else 3,
                      zorder=3 if key != "Gn" else 4)
        axis.set_xticks(list(x), CLASS_LABELS)
        axis.set_xlim(-0.2, len(CLASSES) - 0.8)
        axis.set_ylim(-40, 40)
        axis.set_yticks([-40, -20, 0, 20, 40])
        axis.set_xlabel("velocità del cluster (celle ogni tre ore)", fontsize=TICKS)
        axis.set_ylabel("spostamento lungo il moto (km)", fontsize=TICKS)
        axis.tick_params(labelsize=TICKS, length=0)
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        clean(axis, "top", "right", "left")
        axis.text(-0.15, 38, "davanti", fontsize=TICKS - 1, color=MUTED, va="top")
        axis.text(-0.15, -38, "dietro", fontsize=TICKS - 1, color=MUTED, va="bottom")
        last = part[CLASSES[-1]]
        end_labels(axis, ((last["Gn"], r"$G_\mathrm{n}$", BLUE),
                          (last["normal"], "distanza\ndal normale", ORANGE),
                          (last["change_3h"], "variazione\nin tre ore", GREEN_TEXT),
                          (shallow[CLASSES[-1]].mean(), "sei modelli\nshallow", "#8a8880")))
    save_panels(figure, [[panels[0]], [panels[1]]], HERE, "era5_concordate_moto")
    plt.close(figure)


def pr_figure():
    """Precision against recall of every detector as its own threshold varies, against the concordant cube (left) and
    against the cube without its own vote (right); the dot is the 95th percentile, the threshold of the cube. G_n in
    blue, the two simple baselines in their colours, the six shallow models in grey."""
    table = pd.read_csv(PR, comment="#")
    figure, panels = plt.subplots(1, 2, figsize=size(0.5), constrained_layout=True)
    spaced(figure, 0.2)
    styles = [(key, SHALLOW_GREY, 1.0, 2) for key in SHALLOW_KEYS]
    styles += [("change_3h", GREEN, 1.6, 3), ("normal", ORANGE, 1.6, 3), ("Gn", BLUE, 2.6, 4)]
    for axis, suffix in zip(panels, ("", "_without_vote")):
        for key, colour, width, order in styles:
            part = table[table["detector"] == key].sort_values("percentile")
            axis.plot(part["recall" + suffix], part["precision" + suffix], color=colour, linewidth=width, zorder=order)
            at = part[part["percentile"] == 95]
            axis.plot(at["recall" + suffix], at["precision" + suffix], "o", color=colour, markersize=5 if key == "Gn" else 4,
                      markeredgecolor="white", markeredgewidth=0.6, zorder=order + 1)
        axis.set_xlim(0, 1.02)
        axis.set_ylim(0, 1.02)
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1])
        axis.set_xlabel("recall", fontsize=TICKS)
        axis.tick_params(labelsize=TICKS, length=0)
        axis.grid(color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        clean(axis, "top", "right")
    for axis in panels:  # each panel stands alone
        axis.set_ylabel("precision", fontsize=TICKS)
    handles = [plt.Line2D([], [], color=BLUE, linewidth=2.6, label=r"$G_\mathrm{n}$"),
               plt.Line2D([], [], color=ORANGE, linewidth=1.6, label="distanza dal normale"),
               plt.Line2D([], [], color=GREEN, linewidth=1.6, label="variazione in tre ore"),
               plt.Line2D([], [], color=SHALLOW_GREY, linewidth=1.0, label="sei modelli shallow"),
               plt.Line2D([], [], color=MUTED, marker="o", linestyle="none", markersize=4, label="95-esimo percentile")]
    legend = panels[0].legend(handles=handles, loc="lower left", fontsize=TICKS - 1, frameon=False, handlelength=1.6)
    save_panels(figure, [[panels[0], legend], [panels[1]]], HERE, "era5_concordate_pr")
    plt.close(figure)


if __name__ == "__main__":
    motion_figure()
    pr_figure()

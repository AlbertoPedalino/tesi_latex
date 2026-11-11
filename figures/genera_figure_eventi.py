"""Generate the figures of Sections 6.7 and 6.8: the events of G with the normalisation, by size and type, its zones and
the specific cases. Made for a reader who is not an expert of the domain: one message for every figure.

Run from the thesis folder. First, once, gather the data from the folder downloaded from the server
(outputs/finale_optuna27, with the cases of outputs/finale_optuna27/esempi, written by scripts/check_era5_event_zones.py, check_era5_event_profile.py and
check_era5_event_cases.py of the code repository):
    python figures/genera_figure_eventi.py --prepare <folder with the downloaded files>
which writes figures/dati/era5_eventi_zone.csv, era5_eventi_profilo.csv and era5_caso_NN.npz. Then:
    python figures/genera_figure_eventi.py --coastline <ne_50m_coastline.geojson>
which writes figures/era5_eventi_dimensione.pdf, era5_eventi_tipo.pdf,
era5_eventi_profilo.pdf, era5_eventi_profilo_tipi.pdf, era5_dropout.pdf, era5_dropout_mappe.pdf, era5_caso_NN.pdf, and figures/dati/era5_casi_zone.json with the medians of every case.
The coastline, from Natural Earth (public domain), is not kept in the repository.

Run: optuna-27 (outputs/finale_optuna27), test years 2005-2025, threshold from the validation years.
"""
import argparse
import json
import pathlib
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as dates
import matplotlib.patheffects as effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm
from scipy import ndimage

from pannelli import save_panels

from genera_figure_g_d import (BLUE, GRID, MUTED, ORANGE, TEXT, TICKS, VARIABLE_COLOURS, coastline_of, draw_map,
                               size)

HERE = pathlib.Path(__file__).parent
ZONES = HERE / "dati" / "era5_eventi_zone.csv"
PROFILE = HERE / "dati" / "era5_eventi_profilo.csv"
PROFILE_ALL = HERE / "dati" / "era5_eventi_profilo_tutti.csv"  # every score on the clusters of G_n, all types
SIZES = ("1", "2-4", "5-19", "20-99", "100-999", "1000-9999", ">9999")
SIZE_LABELS = ("1", "2–4", "5–19", "20–99", "100–999", "1000–\n9999", "oltre\n9999")
TYPES = {"vertical_velocity": "Velocità verticale", "total_precipitation": "Precipitazione",
         "mean_sea_level_pressure": "Pressione al livello del mare", "total_cloud_cover": "Copertura nuvolosa",
         "boundary_layer_height": "Strato limite", "specific_humidity": "Umidità specifica",
         "volumetric_soil_water_layer_1": "Umidità del suolo", "friction_velocity": "Velocità d'attrito",
         "surface_solar_radiation_downwards": "Radiazione solare", "v_component_of_wind": "Vento meridionale a 850 hPa",
         "u_component_of_wind": "Vento zonale a 850 hPa", "2m_temperature": "Temperatura a 2 m",
         "temperature": "Temperatura a 850 hPa", "10m_v_component_of_wind": "Vento meridionale a 10 m",
         "10m_u_component_of_wind": "Vento zonale a 10 m"}
RAMP = LinearSegmentedColormap.from_list("blu", ("#eef4fc", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"))
SCORE, NORMAL = "G+clim", "normale"
CORE = "core"
MIN_VALUES = 1000  # a position of a type with fewer values is not drawn
GREEN, GREEN_TEXT = "#1baf7a", "#12875d"  # the text is darker, to be read on white
HALO = [effects.withStroke(linewidth=2.5, foreground="white")]  # a label stays readable over the lines
# The baselines of the comparison section, each with its own events (scripts/run_era5_baseline_events.sh of the code
# repository: check_era5_event_zones.py and check_era5_event_profile.py with --events and --scores the baseline).
BASELINE_DIR = HERE / "dati" / "baseline"
BASELINES = {"normal": "Distanza dal normale", "change_3h": "Variazione in tre ore", "pca": "PCA", "mcd": "MCD",
             "cblof": "CBLOF", "copod": "COPOD", "hbos": "HBOS", "iforest": "Isolation Forest"}
SIMPLE = {"normal": ORANGE, "change_3h": GREEN}  # drawn in colour; the shallow models in grey
SHALLOW_GREY = "#b9b7b0"
EMPTY = "#d9d8d4"  # cells of the type tables with no event
BAND = "#cde2fb"  # the timestamps in which the cell of a case is in its cluster
FLAGGED = "#fbe1d8"  # above the threshold
SHALLOW_KEYS = ("pca", "mcd", "cblof", "copod", "hbos", "iforest")
SPREAD_COLUMN = 90.5  # Figure 6.13c: the x of the column of the differences
MONTHS_SHORT = ("gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic")


def prepare(folder):
    folder = pathlib.Path(folder)
    shutil.copyfile(folder / "zone_5pc.csv", ZONES)
    shutil.copyfile(folder / "profilo_5pc.csv", PROFILE)
    for path in sorted(folder.glob("case_*.npz")):
        shutil.copyfile(path, HERE / "dati" / f"era5_caso_{path.stem[-2:]}.npz")
    print(f"written {ZONES}")


def prepare_baselines(folder):
    """The tables of the baselines, from the output of run_era5_baseline_events.sh."""
    BASELINE_DIR.mkdir(exist_ok=True)
    for name in BASELINES:
        for stem in (f"zone_{name}_5pc", f"zone_{name}_5pc_types", f"profilo_{name}_5pc"):
            shutil.copyfile(pathlib.Path(folder) / f"{stem}.csv", BASELINE_DIR / f"{stem}.csv")
    print(f"written {BASELINE_DIR}")


def method_tables():
    """For G and every baseline: its name, the column prefix of its score, its zones and its profile."""
    zones = pd.read_csv(ZONES, dtype={"size": str})
    methods = [(r"$G_\mathrm{n}$", SCORE, zones, pd.read_csv(PROFILE))]
    for name, label in BASELINES.items():
        methods.append((label, name, pd.read_csv(BASELINE_DIR / f"zone_{name}_5pc.csv", dtype={"size": str}),
                        pd.read_csv(BASELINE_DIR / f"profilo_{name}_5pc.csv")))
    return methods


def baseline_summary():
    """The numbers of the table of G and the baselines: sizes, profile and main types of their own events."""
    rows = []
    for label, key, zones, profile in method_tables():
        by_size = zones[zones["type"] == "all"].set_index("size")
        values = by_size.loc[list(SIZES), f"{key} core values"].sum()
        events = by_size.loc[list(SIZES), "events"].sum()
        types = zones[(zones["type"] != "all") & ~zones["type"].str.startswith("far") & zones["size"].isin(SIZES)]
        types = types.groupby("type")[f"{key} core values"].sum().sort_values(ascending=False) / values * 100
        line = profile[profile["type"] == "all"].set_index("position")[f"{key} percentile"]
        rows.append({"metodo": label, "eventi": int(events), "valori": int(values),
                     "eventi sotto 20 %": 100 * by_size.loc[["1", "2-4", "5-19"], "events"].sum() / events,
                     "valori in eventi da 1000 %": 100 * by_size.loc[["1000-9999", ">9999"], f"{key} core values"].sum() / values,
                     **{f"percentile a {cells} fuori": line[cells] for cells in (1, 2, 3, 6)},
                     "lontano": zones[zones["type"].str.startswith("far")][f"{key} core percentile"].iloc[0],
                     **{f"tipo {rank + 1}": f"{name} {share:.1f}" for rank, (name, share) in enumerate(types.head(3).items())}})
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "dati" / "era5_eventi_baseline.csv", index=False, float_format="%.1f")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.round(1).to_string(index=False))


def read_zones():
    zones = pd.read_csv(ZONES, dtype={"size": str})
    return zones[~zones["type"].str.startswith("far")].set_index(["type", "size"])


def compact(value):
    """A number of values in few characters: 12.7 M, 450 k, 8 k."""
    if value >= 1e6:
        return f"{value / 1e6:.1f} M"
    return f"{value / 1e3:.0f} k" if value >= 1e3 else f"{value:.0f}"


def clean(axis, *sides):
    for side in sides:
        axis.spines[side].set_visible(False)


def sizes_figure(zones):
    """Almost all the clusters are small, almost all the flagged values are in a few huge clusters: two bars of 100,
    the clusters and the flagged values, cut in three bands of size (number of flagged values of the cluster)."""
    rows = [zones.loc[("all", name)] for name in SIZES]
    events = np.array([row["events"] for row in rows], dtype=float)
    values = np.array([row[f"{SCORE} {CORE} values"] for row in rows], dtype=float)
    bands = (("piccoli", "meno di 20 valori", (0, 1, 2), "#b7d3f6", TEXT),
             ("medi", "da 20 a 999 valori", (3, 4), "#5598e7", "white"),
             ("enormi", "1000 valori o più", (5, 6), "#184f95", "white"))
    figure, axis = plt.subplots(figsize=size(0.34), constrained_layout=True)
    for y, counts in ((1, events), (0, values)):
        shares = [100 * counts[list(index)].sum() / counts.sum() for _, _, index, _, _ in bands]
        left = 0.0
        for share, (name, _, _, colour, ink) in zip(shares, bands):
            axis.barh(y, share, left=left, height=0.62, color=colour, edgecolor="white", linewidth=0.8)
            text = f"{share:.0f}%" if share >= 1 else "<1%"
            if share >= 6:
                axis.text(left + share / 2, y, text, ha="center", va="center", fontsize=8, color=ink)
            else:  # too thin for its number: just after the end of the bar
                axis.text(101, y, text, ha="left", va="center", fontsize=8, color=colour if colour != "#b7d3f6" else TEXT)
            left += share
    axis.set_yticks([1, 0], ["% dei cluster", "% dei valori segnalati"])
    axis.set_xlim(0, 108)
    axis.set_ylim(-0.55, 2.05)
    axis.set_xticks([])
    axis.tick_params(length=0, labelsize=TICKS)
    clean(axis, "top", "right", "bottom", "left")
    # The names of the bands, in their colours, above the bars.
    x = 0.0
    for name, detail, _, colour, _ in bands:
        axis.text(x, 1.62, f"\u25a0 {name}\n    {detail}", fontsize=7.5, linespacing=1.1,
                  color=colour if colour != "#b7d3f6" else "#5f8fc9", ha="left", va="center")
        x += 36
    figure.savefig(HERE / "era5_eventi_dimensione.pdf")
    plt.close(figure)


def type_order(zones):
    totals = {name: sum(zones.loc[(name, band), f"{SCORE} {CORE} values"] for band in SIZES if (name, band) in zones.index)
              for name in TYPES}
    return sorted(TYPES, key=lambda name: -totals[name])


def table_axes(axis, order):
    axis.set_xticks(range(len(SIZES)), SIZE_LABELS)
    axis.set_yticks(range(len(order)), [TYPES.get(name, "Tutti i tipi") for name in order])
    axis.xaxis.tick_top()
    axis.xaxis.set_label_position("top")
    axis.set_xlabel("valori del cluster", fontsize=TICKS)
    axis.tick_params(length=0, labelsize=TICKS)
    for side in axis.spines.values():
        side.set_visible(False)


TYPE_COLUMNS = ("vertical_velocity", "total_precipitation", "total_cloud_cover", "volumetric_soil_water_layer_1",
                "friction_velocity", "boundary_layer_height")


def types_figure(zones):
    """Which variables dominate for G_n and for every baseline: the share of the flagged values by type of cluster, for
    all the types, on the rows in the order of G_n; the methods on the columns, G_n first."""
    columns, labels = [], []
    for label, key, table, _ in method_tables():
        table = table[(table["type"] != "all") & ~table["type"].str.startswith("far") & table["size"].isin(SIZES)]
        shares = table.groupby("type")[f"{key} core values"].sum()
        columns.append(100 * shares / shares.sum())
        labels.append(label)
    order = columns[0].reindex(list(TYPES)).fillna(0).sort_values(ascending=False).index  # by the share in G_n
    grid = np.array([[column.get(name, 0.0) for column in columns] for name in order])
    figure, axis = plt.subplots(figsize=size(0.92), constrained_layout=True)
    axis.imshow(grid, cmap=RAMP, vmin=0, vmax=70, aspect="auto", interpolation="nearest")
    for (row, column), value in np.ndenumerate(grid):
        text = f"{value:.0f}%" if value >= 0.5 else "<1%"
        axis.text(column, row, text, ha="center", va="center", fontsize=6.5, color="white" if value > 35 else TEXT)
    heads = {}  # one line each, slanted
    axis.set_xticks(range(len(labels)), [heads.get(label, label) for label in labels], fontsize=7)
    axis.xaxis.tick_top()
    plt.setp(axis.get_xticklabels(), rotation=30, ha="left", rotation_mode="anchor")
    axis.get_xticklabels()[0].set_color(BLUE)
    axis.get_xticklabels()[0].set_fontweight("bold")
    axis.set_yticks(range(len(order)), [TYPES[name] for name in order], fontsize=7)
    axis.axvline(0.5, color="white", linewidth=3)  # G_n apart from the baselines
    axis.tick_params(length=0)
    for side in axis.spines.values():
        side.set_visible(False)
    figure.savefig(HERE / "era5_eventi_tipo.pdf")
    plt.close(figure)



def spaced(figure, wspace=0.14):
    """More room between side-by-side panels, so that the names at the end of the lines of one panel do not seem to
    belong to the next."""
    figure.get_layout_engine().set(w_pad=0.08, h_pad=0.06, wspace=wspace)
    return figure

def position_axis(axis):
    """The horizontal axis of a profile: cells from the border of the event, inside on the left, outside on the right."""
    axis.axvline(0, color=TEXT, linewidth=0.8)
    axis.set_xlim(-8.3, 8.3)
    axis.set_xticks([-8, -6, -4, -2, -1, 1, 2, 4, 6, 8], ["8 o più", "6", "4", "2", "1", "1", "2", "4", "6", "8 o più"])
    axis.set_xlabel("celle dal bordo del cluster:  dentro  |  fuori", fontsize=TICKS)
    axis.tick_params(labelsize=TICKS, length=0)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    clean(axis, "top", "right", "left")


def profile_figure():
    """Left: each method across its own clusters, G falls more slowly than most baselines. Right: on the clusters of G,
    the distance from normal has the same profile as G. Only the positions up to 7 cells, joined by a line."""
    figure, (own, same) = plt.subplots(1, 2, figsize=size(0.56), constrained_layout=True)  # each panel stands alone
    spaced(figure, 0.18)
    for axis in (own, same):
        axis.axhline(50, color=MUTED, linewidth=0.8, linestyle="--")
        axis.axhline(95, color=MUTED, linewidth=0.8, linestyle=":")
        position_axis(axis)
        axis.set_xlim(-7.3, 7.3)
        axis.set_xticks([-6, -4, -2, 2, 4, 6], ["6", "4", "2", "2", "4", "6"])
        axis.set_xlabel("celle dal bordo: dentro | fuori", fontsize=TICKS)
    for axis in (own, same):
        axis.text(-7.1, 51, "valore tipico", fontsize=TICKS - 1, color=MUTED, va="bottom")
        axis.text(-7.1, 92, "soglia", fontsize=TICKS - 1, color=MUTED, va="top")
    shown = lambda table: table[(table["type"] == "all") & (table["position"].abs() < 8)].sort_values("position")
    labels, shallow = [], []
    for label, key, _, table in reversed(method_tables()):  # G last, on top
        table = shown(table)
        line = table.set_index("position")[f"{key} percentile"]
        if key == SCORE:
            own.plot(line.index, line, color=BLUE, linewidth=2.6, zorder=4)
            labels.append((line.iloc[-1], r"$G_\mathrm{n}$", BLUE))
        elif key in SIMPLE:
            own.plot(line.index, line, color=SIMPLE[key], linewidth=1.6, zorder=3)
            name = {"normal": "distanza\ndal normale", "change_3h": "variazione\nin tre ore"}[key]
            labels.append((line.iloc[-1], name, GREEN_TEXT if key == "change_3h" else ORANGE))
        else:
            shallow.append(line)
    band = pd.concat(shallow, axis=1)
    own.fill_between(band.index, band.min(axis=1), band.max(axis=1), color=SHALLOW_GREY, alpha=0.55, linewidth=0, zorder=1)
    labels.append((band.iloc[-1].mean(), "sei modelli\nshallow", "#8a8880"))
    own.set_title("Ogni metodo sui propri cluster", fontsize=9, loc="left", color=TEXT)
    for axis in (own, same):
        axis.set_ylim(0, 104)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.set_ylabel("percentile mediano", fontsize=TICKS)
    end_labels(own, labels)

    # Right: all the methods measured on the clusters of G_n (check_era5_event_profile.py, every score on G's events).
    table = pd.read_csv(PROFILE_ALL)
    table = table[(table["type"] == "all") & (table["position"].abs() < 8)].set_index("position")
    band = table[[f"{key} percentile" for key in ("pca", "mcd", "cblof", "copod", "hbos", "iforest")]]
    same.fill_between(band.index, band.min(axis=1), band.max(axis=1), color=SHALLOW_GREY, alpha=0.55, linewidth=0,
                      zorder=1)
    for column, colour, width in (("change_3h percentile", GREEN, 1.6), (f"{NORMAL} percentile", ORANGE, 1.6),
                                  (f"{SCORE} percentile", BLUE, 2.6)):
        same.plot(table.index, table[column], color=colour, linewidth=width, zorder=3)
    same.set_title(r"Tutti i metodi sui cluster di $G_\mathrm{n}$", fontsize=9, loc="left", color=TEXT)
    last = table.iloc[-1]
    end_labels(same, ((last[f"{SCORE} percentile"], r"$G_\mathrm{n}$", BLUE),
                      (last[f"{NORMAL} percentile"], "distanza\ndal normale", ORANGE),
                      (last["change_3h percentile"], "variazione\nin tre ore", GREEN_TEXT),
                      (band.iloc[-1].mean(), "sei modelli\nshallow", "#8a8880")))
    for axis in (own, same):
        axis.text(0.15, 3, "bordo", fontsize=TICKS - 1, color=TEXT)
    save_panels(figure, [[own], [same]], HERE, "era5_eventi_profilo")
    plt.close(figure)


def profile_types_figure():
    """G_n falls in the same way for every type of cluster (left), the variable of the type does not (right): some are
    back to normal one cell out of the core, others stay far from normal. Positions up to 7 cells, joined by a line."""
    table = pd.read_csv(PROFILE)
    table = table[(table["type"] != "all") & (table["values"] >= MIN_VALUES)]
    places = {"total_precipitation": (3.6, 0.42, -12), "vertical_velocity": (-7.0, 1.62, 0),
              "total_cloud_cover": (-5.3, 3.3, 0), "2m_temperature": (0.3, 2.4, 0)}
    chosen = {name: (*VARIABLE_COLOURS[name], *place) for name, place in places.items()}
    figure, (score, scarto) = plt.subplots(1, 2, figsize=size(0.56), constrained_layout=True)
    spaced(figure, 0.16)
    for name, group in table.groupby("type"):
        group = group.sort_values("position")
        joined = group["position"].abs() < 8
        colour, width, order = (chosen[name][0], 1.8, 2) if name in chosen else ("#c8c6bf", 0.9, 1)
        score.plot(group["position"][joined], group[f"{SCORE} percentile"][joined], color=colour, linewidth=width,
                   zorder=order)
        scarto.plot(group["position"][joined], group["variable distance"][joined], color=colour, linewidth=width + 0.2,
                    zorder=order)
    for name, (colour, label, x, y, shift) in chosen.items():
        scarto.annotate(TYPES[name], (x, y), xytext=(0, shift), textcoords="offset points", fontsize=TICKS - 0.5,
                        color=label, zorder=3, path_effects=HALO)
    scarto.annotate("altri tipi", (5.0, 1.75), fontsize=TICKS - 0.5, color=MUTED)
    score.annotate("le curve di tutti\ni tipi quasi coincidono", (-7.0, 72), fontsize=TICKS - 0.5, color=TEXT,
                   path_effects=HALO)
    score.axhline(50, color=MUTED, linewidth=0.8, linestyle="--")
    score.axhline(95, color=MUTED, linewidth=0.8, linestyle=":")
    score.set_ylim(0, 104)
    score.set_yticks([0, 50, 95], ["0", "50", "soglia"])
    score.set_ylabel(r"percentile mediano di $G_\mathrm{n}$", fontsize=TICKS)
    score.set_title(r"$G_\mathrm{n}$ per tipo di cluster", fontsize=9, loc="left", color=TEXT)
    scarto.set_ylim(0, 3.5)
    scarto.set_ylabel("|scarto standardizzato| (dev. std.)", fontsize=TICKS)
    scarto.set_title("La variabile del tipo", fontsize=9, loc="left", color=TEXT)
    for axis in (score, scarto):
        position_axis(axis)
        axis.set_xlim(-7.3, 7.3)
        axis.set_xticks([-6, -4, -2, 2, 4, 6], ["6", "4", "2", "2", "4", "6"])
        axis.set_xlabel("celle dal bordo: dentro | fuori", fontsize=TICKS)
    scarto.text(0.15, 0.08, "bordo", fontsize=TICKS - 1, color=TEXT)
    score.text(0.15, 3, "bordo", fontsize=TICKS - 1, color=TEXT)
    save_panels(figure, [[score], [scarto]], HERE, "era5_eventi_profilo_tipi")
    plt.close(figure)



def types_spread_figure():
    """Figure 6.13c: does every method fall in the same way for every type of cluster? For each method on its own
    clusters, the median percentile two cells out of the border for every type (a point per type), from the lowest
    type to the highest; the methods from the most uniform to the least."""
    rows = []
    for label, key, _, profile in method_tables():
        table = profile[(profile["type"] != "all") & (profile["position"] == 2) & (profile["values"] >= MIN_VALUES)]
        rows.append((label, key, table.set_index("type")[f"{key} percentile"]))
    rows.sort(key=lambda row: row[2].max() - row[2].min())
    coloured = ("vertical_velocity", "total_precipitation", "total_cloud_cover", "2m_temperature")  # as in 6.13a
    figure, axis = plt.subplots(figsize=size(0.42), constrained_layout=True)
    for y, (label, key, values) in enumerate(rows):
        mine = key == SCORE
        axis.plot([values.min(), values.max()], [y, y], color=BLUE if mine else "#d6d4cd", linewidth=6,
                  solid_capstyle="round", alpha=0.35 if mine else 1.0, zorder=1)
        others = values[[name for name in values.index if name not in coloured]]
        axis.scatter(others, np.full(len(others), y), s=11, color="#9a988f", linewidths=0, zorder=2)
        for name in coloured:
            if name in values.index:
                axis.scatter(values[name], y, s=24, color=VARIABLE_COLOURS[name][0], edgecolors="white",
                             linewidths=0.6, zorder=3)
        axis.text(SPREAD_COLUMN, y, f"{values.max() - values.min():.0f}", ha="center", va="center", fontsize=7.5,
                  color=BLUE if mine else TEXT)
    axis.set_yticks(range(len(rows)), [label for label, _, _ in rows], fontsize=TICKS)
    for tick, (_, key, _) in zip(axis.get_yticklabels(), rows):
        if key == SCORE:
            tick.set_color(BLUE)
            tick.set_fontweight("bold")
    axis.invert_yaxis()  # the most uniform on top
    axis.text(SPREAD_COLUMN, -0.95, "differenza fra\nil tipo più alto\ne il più basso", ha="center", va="bottom",
              fontsize=6.5, color=MUTED, linespacing=1.0)
    axis.axvline(SPREAD_COLUMN - 2.2, color=GRID, linewidth=0.8)
    axis.set_xlim(56, SPREAD_COLUMN + 2.5)
    axis.set_xticks([60, 65, 70, 75, 80, 85])
    axis.set_ylim(len(rows) - 0.4, -0.6)
    axis.set_xlabel("percentile mediano dello score a due celle fuori dal bordo (50 = valore tipico, 95 = soglia)",
                    fontsize=TICKS)
    axis.tick_params(labelsize=TICKS, length=0)
    axis.grid(axis="x", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    clean(axis, "top", "right", "left")
    handles = [plt.Line2D([], [], linestyle="", marker="o", markersize=5, color=VARIABLE_COLOURS[name][0],
                          markeredgecolor="white") for name in coloured]
    handles.append(plt.Line2D([], [], linestyle="", marker="o", markersize=3.5, color="#9a988f"))
    legend = axis.legend(handles, [TYPES[name] for name in coloured] + ["altri tipi"], loc="center left",
                         bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=6.5, handletextpad=0.3, labelspacing=0.5,
                         title="ogni pallino\nè un tipo di cluster", title_fontsize=6.5)
    legend._legend_box.align = "left"
    figure.savefig(HERE / "era5_eventi_profilo_tipi_c.pdf")
    plt.close(figure)

def dropout_figure():
    """The dispersion in proportion to the error: by size of the error, across the events, at equal error."""
    data = json.loads((HERE / "dati" / "era5_dropout.json").read_text(encoding="utf-8"))
    figure, (bands, across, classes) = plt.subplots(1, 3, figsize=size(0.5), constrained_layout=True, width_ratios=(1.3, 1.2, 0.95))
    spaced(figure, 0.12)
    numbers = data["ratio by band of the raw error"]
    place = np.arange(len(numbers["band"]))
    bands.bar(place, 100 * np.array(numbers["ratio"]), width=0.72, color=BLUE)
    for x, y in zip(place, numbers["ratio"]):
        bands.text(x, 100 * y + 0.25, f"{100 * y:.1f}", ha="center", fontsize=6.5, color=TEXT)
    bands.set_ylim(0, 13)
    bands.set_yticks([])
    bands.set_xticks(place, ["<50", "50–90", "90–95", "95–99", "99–99.9", ">99.9"], rotation=35, ha="right", rotation_mode="anchor")
    bands.tick_params(labelsize=6.5, length=0)
    bands.set_ylabel("rapporto (%)", fontsize=TICKS)
    bands.set_xlabel("percentile dell'errore", fontsize=TICKS)
    bands.set_title("Secondo l'errore", fontsize=8.5, loc="left", color=TEXT)
    clean(bands, "top", "right", "left")

    position = np.array(data["positions"]["position"])
    ratio = 100 * np.array(data["positions"]["ratio"])
    joined = np.abs(position) < 8
    across.plot(position[joined], ratio[joined], color=BLUE, linewidth=2, marker="o", markersize=4, markeredgecolor="white")
    across.plot(position[~joined], ratio[~joined], color=BLUE, linewidth=0, marker="o", markersize=4, markeredgecolor="white")
    position_axis(across)
    across.set_xticks([-8, -4, -1, 1, 4, 8], ["8+", "4", "1", "1", "4", "8+"])
    across.set_xlabel("celle dal bordo: dentro | fuori", fontsize=TICKS)
    across.set_ylabel("rapporto (%)", fontsize=TICKS)
    across.set_ylim(0, 13)  # from 0: the differences across the clusters are small, and must look small
    across.set_yticks([0, 5, 10])
    across.set_title("Attraverso i cluster", fontsize=8.5, loc="left", color=TEXT)

    numbers = data["classes of the ratio at equal raw error"]
    classes.bar(numbers["class"], numbers["normal flagged %"], width=0.72, color=ORANGE)
    for x, y in zip(numbers["class"], numbers["normal flagged %"]):
        classes.text(x, y + 1.5, f"{y:.0f}%", ha="center", fontsize=6.5, color=TEXT)
    classes.set_ylim(0, 72)
    classes.set_yticks([])
    classes.set_xticks(numbers["class"], ["1", "2", "3", "4", "5"])
    classes.tick_params(labelsize=TICKS, length=0)
    classes.set_xlabel("classi, dalle estrazioni\nconcordi alle discordi", fontsize=TICKS)
    classes.set_ylabel("oltre la soglia della\ndistanza dal normale", fontsize=TICKS)
    classes.set_title("A parità di errore", fontsize=8.5, loc="left", color=TEXT)
    clean(classes, "top", "right", "left")
    save_panels(figure, [[bands], [across], [classes]], HERE, "era5_dropout")
    plt.close(figure)


def dropout_maps_figure(coastline_path):
    """The standard deviation of the draws cell by cell, mean over the test years, and its mean ratio to the error."""
    maps = dict(np.load(HERE / "dati" / "era5_dropout_mappe.npz"))
    grid = {key: maps[key] for key in ("rows", "cols", "latitudes", "longitudes")}
    coastline = coastline_of(coastline_path, grid)
    figure, axes = plt.subplots(1, 2, figsize=size(0.56), constrained_layout=True)
    groups = []
    for axis, key, title, label, scale in ((axes[0], "mean grezza", "Deviazione standard, media", r"deviazione standard dell'errore, $\times 10^{-4}$", 1e4),
                                           (axes[1], "mean ratio", "Rapporto, medio", "rapporto (%)", 100)):
        values = maps[key] * scale
        low, high = np.nanpercentile(values, (1, 99))
        mesh = draw_map(axis, grid, values, coastline, left=axis is axes[0], cmap="viridis", vmin=low, vmax=high)
        axis.set_title(title, fontsize=9, loc="left", color=TEXT)
        bar = figure.colorbar(mesh, ax=axis, orientation="horizontal", shrink=0.9, pad=0.02, aspect=30)
        bar.set_label(label, fontsize=TICKS)
        bar.ax.tick_params(labelsize=TICKS)
        groups.append([axis, bar.ax])
    save_panels(figure, groups, HERE, "era5_dropout_mappe", dpi=300)
    plt.close(figure)


def percentile(values, quantiles):
    return np.interp(values, quantiles, np.linspace(0, 100, len(quantiles)))


def end_labels(axis, lines, fontsize=7):
    """Write the name of every line just right of its last value, moved up or down only as much as needed to keep the
    names apart. lines: (last value in the units of the axis, label, colour); a label may have two lines."""
    figure = axis.figure
    figure.canvas.draw()  # the height of the axes, to turn the size of the text into the units of the axis
    low, high = axis.get_ylim()
    per_unit = axis.get_window_extent().height / (high - low)
    line_height = 1.2 * fontsize * figure.dpi / 72 / per_unit
    entries = sorted(((float(value), label, colour, (label.count("\n") + 1) * line_height) for value, label, colour in lines))
    places = [value for value, _, _, _ in entries]
    for _ in range(50):  # push apart the neighbours that overlap, then keep them inside the axis
        for i in range(1, len(entries)):
            need = (entries[i - 1][3] + entries[i][3]) / 2
            if places[i] - places[i - 1] < need:
                middle = (places[i] + places[i - 1]) / 2
                places[i - 1], places[i] = middle - need / 2, middle + need / 2
        places = [min(max(place, low + height / 2), high - height / 2)
                  for place, (_, _, _, height) in zip(places, entries)]
    for (_, label, colour, _), place in zip(entries, places):
        axis.annotate(label, (1, place), xycoords=("axes fraction", "data"), xytext=(3, 0), textcoords="offset points",
                      fontsize=fontsize, color=colour, ha="left", va="center", linespacing=1.0, annotation_clip=False)


def case_figure(number, data, coastline):
    """A case: where it is, how unusual, when it starts and ends, and how long the chain it belongs to lasts."""
    record = json.loads(str(data["case"]))
    moment = pd.Timestamp(record["timestamp"])
    grid = {key: data[key] for key in ("rows", "cols", "latitudes", "longitudes")}
    longitudes, latitudes = data["longitudes"], data["latitudes"]
    picture = np.zeros((len(latitudes), len(longitudes)))
    picture[data["rows"], data["cols"]] = data["core"]
    figure = plt.figure(figsize=size(0.98), constrained_layout=True)
    # More room between the four panels, so that each reads on its own: maps above, time and profile below.
    figure.get_layout_engine().set(w_pad=0.08, h_pad=0.08, wspace=0.22, hspace=0.1)
    panels = figure.add_gridspec(2, 2, height_ratios=(1.35, 1.25))
    left, right = figure.add_subplot(panels[0, 0]), figure.add_subplot(panels[0, 1])
    limit = 4
    mesh = draw_map(left, grid, data["input"], coastline, cmap="RdBu_r", vmin=-limit, vmax=limit)
    left_bar = figure.colorbar(mesh, ax=left, orientation="horizontal", shrink=0.9, pad=0.02, aspect=30, extend="both")
    left_bar.set_label("scarto standardizzato\n(deviazioni standard dal normale)", fontsize=TICKS)
    left_bar.ax.tick_params(labelsize=TICKS)
    left.set_title(TYPES[record['type']], fontsize=9, loc="left", color=TEXT)
    mesh = draw_map(right, grid, percentile(data[f"score_{SCORE}"], data[f"quantiles_{SCORE}"]), coastline, left=False,
                    cmap="viridis", vmin=0, vmax=100)
    right_bar = figure.colorbar(mesh, ax=right, orientation="horizontal", shrink=0.9, pad=0.02, aspect=30,
                                ticks=[0, 25, 50, 75, 95])
    right_bar.set_label("percentile (95 circa = soglia)", fontsize=TICKS)
    right_bar.ax.tick_params(labelsize=TICKS)
    right.set_title(r"$G_\mathrm{n}$", fontsize=9, loc="left", color=TEXT)
    for axis in (left, right):
        # The core as in Figure 6.8: orange over a white halo, apart from the dark coastlines.
        axis.contour(longitudes, latitudes[::-1], picture[::-1], levels=[0.5], colors="white", linewidths=2.4)
        axis.contour(longitudes, latitudes[::-1], picture[::-1], levels=[0.5], colors=ORANGE, linewidths=1.2)
        # The cell of the series below: a ring with a white halo, visible on every colour of the maps.
        axis.plot(record["cell longitude"], record["cell latitude"], marker="o", markersize=9, markerfacecolor="none",
                  markeredgecolor="white", markeredgewidth=2.6, zorder=5)
        axis.plot(record["cell longitude"], record["cell latitude"], marker="o", markersize=9, markerfacecolor="none",
                  markeredgecolor=TEXT, markeredgewidth=1.2, zorder=6)

    # The profile of this cluster at the timestamp of the maps, as in Figure 6.12: every cell has its position from the
    # border of the core (negative inside, positive outside, chessboard distance as in check_era5_event_profile.py);
    # outside, only the cells whose nearest flagged cell is in this core, so that other clusters do not count here.
    profile = figure.add_subplot(panels[1, 1])
    shape_grid = (len(latitudes), len(longitudes))
    flagged = np.zeros(shape_grid, dtype=bool)
    flagged[data["rows"], data["cols"]] = data[f"score_{SCORE}"] > float(data[f"limit_{SCORE}"])
    own = np.zeros(shape_grid, dtype=bool)
    own[data["rows"], data["cols"]] = data["core"].astype(bool)
    flagged |= own
    inside = ndimage.distance_transform_cdt(own, metric="chessboard")
    outside, nearest = ndimage.distance_transform_cdt(~flagged, metric="chessboard", return_indices=True)
    position = np.where(own, -inside, outside)[data["rows"], data["cols"]]
    mine = (own | own[nearest[0], nearest[1]])[data["rows"], data["cols"]]
    level_map = percentile(data[f"score_{SCORE}"], data[f"quantiles_{SCORE}"])
    normal_map = percentile(data[f"score_{NORMAL}"], data[f"quantiles_{NORMAL}"])
    steps = [k for k in range(-7, 8) if k != 0]
    medians = {}
    maps = {name: percentile(data[f"score_{name}"], data[f"quantiles_{name}"])
            for name in (SCORE, NORMAL, "change_3h", *SHALLOW_KEYS)}
    for name, values in maps.items():
        medians[name] = [np.nanmedian(values[mine & (position == k)]) if (mine & (position == k)).sum() >= 5
                         else np.nan for k in steps]
    shallow = np.array([medians[name] for name in SHALLOW_KEYS])
    profile.fill_between(steps, np.nanmin(shallow, axis=0), np.nanmax(shallow, axis=0), color=SHALLOW_GREY, alpha=0.55,
                         linewidth=0, zorder=1)
    for name, colour, width in (("change_3h", GREEN, 1.4), (NORMAL, ORANGE, 1.6), (SCORE, BLUE, 2.2)):
        profile.plot(steps, medians[name], color=colour, linewidth=width, marker="o", markersize=2.5, zorder=3)
    profile.axhline(50, color=MUTED, linewidth=0.8, linestyle="--")
    profile.axhline(95, color=MUTED, linewidth=0.8, linestyle=":")
    profile.axvline(0, color=TEXT, linewidth=0.8)
    profile.text(0.15, 3, "bordo", fontsize=TICKS - 1, color=TEXT)
    profile.set_xlim(-5.5, 7.5)
    profile.set_xticks([-4, -2, 2, 4, 6], ["4", "2", "2", "4", "6"])
    profile.set_xlabel("celle dal bordo: dentro | fuori", fontsize=TICKS)
    profile.set_ylim(0, 104)
    profile.set_yticks([0, 50, 95], ["0", "50", "soglia"])
    profile.set_ylabel("percentile mediano", fontsize=TICKS)
    profile.tick_params(labelsize=TICKS, length=2)
    profile.set_title("Dal nucleo verso l'esterno", fontsize=9, loc="left", color=TEXT)
    clean(profile, "top", "right")
    last = lambda values: next((v for v in reversed(values) if np.isfinite(v)), 50.0)
    end_labels(profile, ((last(medians[SCORE]), r"$G_\mathrm{n}$", BLUE),
                         (last(medians[NORMAL]), "distanza\ndal normale", ORANGE),
                         (last(medians["change_3h"]), "variazione\nin tre ore", GREEN_TEXT),
                         (float(np.nanmean([last(medians[name]) for name in SHALLOW_KEYS])), "sei modelli\nshallow",
                          "#8a8880")))
    save_panels(figure, [[left, left_bar.ax], [right, right_bar.ax], [profile]], HERE, f"era5_caso_{number:02d}",
                letters="abd", dpi=300)
    plt.close(figure)
    case_series_figure(number, data)


def case_series_figure(number, data):
    """The circled cell in time, full width, on the days around the maps: above the percentiles of the scores, which
    read all the variables of the cell, with the flagged zone shaded; below the scarto of the variable of the type
    alone. The light blue bands are the timestamps in which the cell is in the cluster."""
    record = json.loads(str(data["case"]))
    moment = pd.Timestamp(record["timestamp"])
    times = pd.DatetimeIndex(data["series_times"])
    figure = plt.figure(figsize=size(0.48), constrained_layout=True)
    rows = figure.add_gridspec(2, 1, height_ratios=(1.9, 1))
    scores = figure.add_subplot(rows[0])
    alone = figure.add_subplot(rows[1], sharex=scores)
    flagged = data["series_core"]
    bands = list(zip(np.nonzero(np.diff(np.r_[0, flagged.astype(int)]) == 1)[0],
                     np.nonzero(np.diff(np.r_[flagged.astype(int), 0]) == -1)[0]))
    half = pd.Timedelta(hours=1.5)  # a band of one timestamp would have no width
    for axis in (scores, alone):
        for begin, end in bands:
            axis.axvspan(times[begin] - half, times[end] + half, color=BAND, linewidth=0, zorder=0)
        axis.axvline(moment, color=TEXT, linewidth=0.9, zorder=5)
        axis.margins(x=0)
        axis.tick_params(labelsize=TICKS, length=2)
        clean(axis, "top", "right")

    # Above: the scores, as percentiles of their validation values; above 95 the cell is flagged.
    scores.axhspan(95, 100, color=FLAGGED, linewidth=0, zorder=0.5)
    scores.axhline(95, color="#c8553d", linewidth=0.7, linestyle="--", zorder=0.6)
    scores.text(0.006, 97.5, "segnalato", transform=scores.get_yaxis_transform(), fontsize=TICKS - 1, color="#b3462a",
                va="center", zorder=6, path_effects=HALO)
    series = {name: percentile(data[f"series_{name}"], data[f"quantiles_{name}"])
              for name in (SCORE, NORMAL, "change_3h", *SHALLOW_KEYS)}
    shallow = np.array([series[name] for name in SHALLOW_KEYS])
    scores.fill_between(times, shallow.min(axis=0), shallow.max(axis=0), color=SHALLOW_GREY, alpha=0.55, linewidth=0,
                        zorder=1)
    for name, colour, width in (("change_3h", GREEN, 1.2), (NORMAL, ORANGE, 1.5), (SCORE, BLUE, 2.0)):
        scores.plot(times, series[name], color=colour, linewidth=width, zorder=3)
    scores.set_ylim(0, 100)
    scores.set_yticks([0, 50, 95], ["0", "50", "95"])
    scores.set_ylabel("percentile\ndello score", fontsize=TICKS)
    scores.tick_params(labelbottom=False)
    scores.set_title("Score della cella (tutte le variabili)", fontsize=8.5, loc="left", color=TEXT, pad=7)
    # A legend beside the panel: four names at the end of lines that cross each other would overlap.
    handles = [plt.Line2D([], [], color=BLUE, linewidth=2.0), plt.Line2D([], [], color=ORANGE, linewidth=1.5),
               plt.Line2D([], [], color=GREEN, linewidth=1.2),
               plt.Rectangle((0, 0), 1, 1, color=SHALLOW_GREY, alpha=0.55, linewidth=0)]
    scores.legend(handles, [r"$G_\mathrm{n}$", "distanza dal normale", "variazione in tre ore", "sei modelli shallow"],
                  loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=6.5, handlelength=1.4,
                  labelspacing=0.5)

    # Below: the variable of the type alone, in standard deviations from its seasonal normal.
    variable, variable_text = VARIABLE_COLOURS.get(record["type"], (MUTED, TEXT))
    alone.axhline(0, color=MUTED, linewidth=0.8)
    alone.annotate("normale", (1, 0), xycoords=("axes fraction", "data"), xytext=(3, 0), textcoords="offset points",
                   fontsize=6.5, color=MUTED, ha="left", va="center", annotation_clip=False)
    alone.plot(times, data["series_input"], color=variable, linewidth=1.5, zorder=3)
    reach = max(3.0, float(np.ceil(np.nanmax(np.abs(data["series_input"])))))
    alone.set_ylim(-reach, reach)
    alone.set_yticks([-reach, 0, reach], [f"{-reach:g}", "0", f"+{reach:g}"])
    alone.set_ylabel("scarto stand.\n(dev. std.)", fontsize=TICKS)
    # In ERA5 the vertical velocity is in Pa/s, negative when the air rises.
    note = " (negativo: l'aria sale più del solito)" if record["type"] == "vertical_velocity" else ""
    alone.set_title(f"{TYPES[record['type']]} da sola: scarto standardizzato{note}", fontsize=8, loc="left", color=TEXT,
                    pad=7)
    alone.annotate("istante delle mappe", (moment, 0.0), xycoords=("data", "axes fraction"), xytext=(3, 2),
                   textcoords="offset points", fontsize=TICKS - 1, color=TEXT, ha="left", va="bottom",
                   path_effects=HALO, zorder=6)
    widest = max(bands, key=lambda band: band[1] - band[0])
    middle = times[widest[0]] + (times[widest[1]] - times[widest[0]]) / 2
    inside = times[widest[0]] - half <= moment <= times[widest[1]] + half
    anchor, shift, side = (moment, -3, "right") if inside and moment >= middle else \
        ((moment, 3, "left") if inside else (middle, 0, "center"))
    alone.annotate("la cella è nel cluster", (anchor, 1), xycoords=("data", "axes fraction"), xytext=(shift, -1),
                   textcoords="offset points", fontsize=6.5, color="#3d6fa8", ha=side, va="top", path_effects=HALO,
                   zorder=6)
    alone.xaxis.set_major_locator(dates.DayLocator())
    alone.xaxis.set_major_formatter(lambda value, _: f"{dates.num2date(value).day} "
                                                     f"{MONTHS_SHORT[dates.num2date(value).month - 1]}")
    alone.set_xlabel(f"{moment.year}, tacche a mezzanotte UTC", fontsize=TICKS)
    east = record["cell longitude"]
    where = f"{east:g}°E" if east >= 0 else f"{-east:g}°W"
    figure.suptitle(f"La cella cerchiata nel tempo ({record['cell latitude']:g}°N, {where})", x=0.01, ha="left",
                    fontsize=9, color=TEXT)
    figure.savefig(HERE / f"era5_caso_{number:02d}_c.pdf")
    plt.close(figure)


def case_medians(data):
    """Median percentile of G with the normalisation and median |distance from normal| in the three zones of a case."""
    level = percentile(data[f"score_{SCORE}"], data[f"quantiles_{SCORE}"])
    return {zone: {"cells": int(data[zone].sum()), "G percentile": float(np.nanmedian(level[data[zone]])),
                   "distance": float(np.nanmedian(np.abs(data["input"][data[zone]])))} for zone in ("core", "edge", "far")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=pathlib.Path, help="Folder with the files downloaded from the server")
    parser.add_argument("--prepare-baselines", type=pathlib.Path, help="Output of run_era5_baseline_events.sh")
    parser.add_argument("--coastline", type=pathlib.Path, help="Natural Earth coastline, GeoJSON")
    args = parser.parse_args()
    if args.prepare is not None:
        prepare(args.prepare)
    if args.prepare_baselines is not None:
        prepare_baselines(args.prepare_baselines)
    zones = read_zones()
    baseline_summary()
    sizes_figure(zones)
    types_figure(zones)
    profile_figure()
    profile_types_figure()
    types_spread_figure()
    dropout_figure()
    dropout_maps_figure(args.coastline)
    medians = {}
    for path in sorted((HERE / "dati").glob("era5_caso_*.npz")):
        with np.load(path) as data:
            number = int(path.stem[-2:])
            arrays = {key: data[key] for key in data.files}
            coastline = coastline_of(args.coastline, arrays)
            case_figure(number, arrays, coastline)
            medians[json.loads(str(arrays["case"]))["type"]] = {"record": json.loads(str(arrays["case"])), **case_medians(arrays)}
    (HERE / "dati" / "era5_casi_zone.json").write_text(json.dumps(medians, indent=1), encoding="utf-8")
    print(json.dumps({name: {zone: value[zone] for zone in ("core", "edge", "far")} for name, value in medians.items()}, indent=1))


if __name__ == "__main__":
    main()

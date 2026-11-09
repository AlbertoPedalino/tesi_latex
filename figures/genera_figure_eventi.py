"""Generate the figures of Sections 6.7 and 6.8: the events of G with the normalisation, by size and type, its zones and
the specific cases. Made for a reader who is not an expert of the domain: one message for every figure.

Run from the thesis folder. First, once, gather the data from the folder downloaded from the server
(outputs/finale_optuna27, with the cases of outputs/finale_optuna27/esempi, written by scripts/check_era5_event_zones.py, check_era5_event_profile.py and
check_era5_event_cases.py of the code repository):
    python figures/genera_figure_eventi.py --prepare <folder with the downloaded files>
which writes figures/dati/era5_eventi_zone.csv, era5_eventi_profilo.csv and era5_caso_NN.npz. Then:
    python figures/genera_figure_eventi.py --coastline <ne_50m_coastline.geojson>
which writes figures/era5_eventi_dimensione.pdf, era5_eventi_tipo.pdf, era5_eventi_mappa_zone.pdf,
era5_eventi_profilo.pdf, era5_eventi_profilo_tipi.pdf, era5_dropout.pdf, era5_dropout_tipi.pdf, era5_dropout_mappe.pdf, era5_caso_NN.pdf, and figures/dati/era5_casi_zone.json with the medians of every case.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import ndimage

from genera_figure_g_d import BLUE, GRID, MUTED, ORANGE, TEXT, TICKS, YELLOW, coastline_of, draw_map, size

HERE = pathlib.Path(__file__).parent
ZONES = HERE / "dati" / "era5_eventi_zone.csv"
PROFILE = HERE / "dati" / "era5_eventi_profilo.csv"
SIZES = ("1", "2-4", "5-19", "20-99", "100-999", "1000-9999", ">9999")
SIZE_LABELS = ("1", "2–4", "5–19", "20–99", "100–999", "1000–\n9999", "oltre\n9999")
TYPES = {"vertical_velocity": "Velocità verticale", "total_precipitation": "Precipitazione",
         "mean_sea_level_pressure": "Pressione al livello del mare", "total_cloud_cover": "Copertura nuvolosa",
         "boundary_layer_height": "Strato limite", "specific_humidity": "Umidità specifica",
         "volumetric_soil_water_layer_1": "Umidità del suolo", "friction_velocity": "Velocità di attrito",
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
PURPLE = "#7b3fbf"  # the distance from normal read on the events of G, not on its own
EMPTY = "#d9d8d4"  # cells of the type tables with no event


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
    """Almost all the events are small, almost all the values are in a few huge events."""
    rows = [zones.loc[("all", name)] for name in SIZES]
    events = np.array([row["events"] for row in rows], dtype=float)
    values = np.array([row[f"{SCORE} {CORE} values"] for row in rows], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=size(0.42), sharey=True, constrained_layout=True)
    place = np.arange(len(SIZES))[::-1]
    for axis, shares, title in ((axes[0], 100 * events / events.sum(), "Quota dei cluster"),
                                (axes[1], 100 * values / values.sum(), "Quota dei valori segnalati")):
        axis.barh(place, shares, height=0.72, color=BLUE)
        for y, share in zip(place, shares):
            axis.text(share + 1, y, f"{share:.1f}%", va="center", fontsize=TICKS, color=TEXT)
        axis.set_xlim(0, 62)
        axis.set_xticks([])
        axis.set_title(title, fontsize=9, loc="left", color=TEXT)
        axis.tick_params(length=0, labelsize=TICKS)
        clean(axis, "top", "right", "bottom")
    axes[0].set_yticks(place, [label.replace("\n", "") if "–" in label else label.replace("\n", " ") for label in SIZE_LABELS])
    axes[0].set_ylabel("numero di valori del cluster", fontsize=TICKS)
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


def types_figure(zones):
    """Which variables dominate, and at which size, for G and for every baseline: the values flagged by type and size."""
    order = type_order(zones)
    norm = LogNorm(vmin=5e3, vmax=2.5e7)
    figure, axes = plt.subplots(3, 3, figsize=size(1.18), sharex=True, sharey=True, constrained_layout=True)
    for axis, (label, key, table, _) in zip(axes.flat, method_tables()):
        table = table.set_index(["type", "size"])
        grid = np.full((len(order), len(SIZES)), np.nan)
        for row, name in enumerate(order):
            for column, band in enumerate(SIZES):
                if (name, band) in table.index:
                    grid[row, column] = table.loc[(name, band), f"{key} core values"]
        axis.set_facecolor(EMPTY)  # a size with no event of that type
        mesh = axis.imshow(grid, cmap=RAMP, norm=norm, aspect="auto", interpolation="nearest")
        # The two long names on two lines: on one line they would run into the title of the next panel.
        title = {"Distanza dal normale": "Distanza\ndal normale", "Variazione in tre ore": "Variazione\nin tre ore"}.get(label, label)
        axis.set_title("G" if key == SCORE else title, fontsize=8.5, loc="left", color=BLUE if key == SCORE else TEXT,
                       fontweight="bold" if key == SCORE else "normal")
        axis.tick_params(length=0, labelsize=7.5)
        for side in axis.spines.values():
            side.set_visible(key == SCORE)
            side.set_color(BLUE)
    for axis in axes[-1]:
        axis.set_xticks(range(len(SIZES)), ("1", "2–4", "5–19", "20–99", "100–999", "1000–9999", "oltre 9999"),
                        fontsize=7.5, rotation=90)
    for axis in axes[:, 0]:
        axis.set_yticks(range(len(order)), [TYPES[name] for name in order], fontsize=7.5)
    figure.supxlabel("numero di valori del cluster", fontsize=TICKS)
    bar = figure.colorbar(mesh, ax=axes, shrink=0.5, pad=0.01, aspect=30)
    bar.set_label("valori segnalati (scala logaritmica)", fontsize=TICKS)
    bar.ax.tick_params(labelsize=TICKS)
    figure.savefig(HERE / "era5_eventi_tipo.pdf")
    plt.close(figure)


def zones_map_figure(data, coastline):
    """What the three zones are, on a map: G at one instant, and its zones. The zones are those of all the events."""
    grid = {key: data[key] for key in ("rows", "cols", "latitudes", "longitudes")}
    level = percentile(data[f"score_{SCORE}"], data[f"quantiles_{SCORE}"])
    core = np.zeros((len(data["latitudes"]), len(data["longitudes"])), dtype=bool)
    core[data["rows"], data["cols"]] = data[f"score_{SCORE}"] > data[f"limit_{SCORE}"]
    edge = ndimage.binary_dilation(core, structure=np.ones((3, 3), dtype=bool)) & ~core
    zone = np.where(data["far"], 0, 1)
    zone[edge[data["rows"], data["cols"]]] = 2
    zone[core[data["rows"], data["cols"]]] = 3
    figure, axes = plt.subplots(1, 2, figsize=size(0.56), constrained_layout=True)
    mesh = draw_map(axes[0], grid, level, coastline, cmap="viridis", vmin=0, vmax=100)
    bar = figure.colorbar(mesh, ax=axes[0], orientation="horizontal", shrink=0.9, pad=0.02, aspect=30, ticks=[0, 25, 50, 75, 95])
    bar.set_label("percentile (95 circa = soglia)", fontsize=TICKS)
    bar.ax.tick_params(labelsize=TICKS)
    axes[0].set_title(r"$G_\mathrm{n}$", fontsize=9, loc="left", color=TEXT)
    colours = ["#f4f3ef", "#cfcdc6", ORANGE, BLUE]
    mesh = draw_map(axes[1], grid, zone, coastline, left=False, cmap=ListedColormap(colours), vmin=-0.5, vmax=3.5)
    bar = figure.colorbar(mesh, ax=axes[1], orientation="horizontal", shrink=1.0, pad=0.02, aspect=30, ticks=range(4))
    bar.set_ticklabels(("lontana", "intermedia", "contorno", "nucleo"), fontsize=7.5)  # 7.5: at 8 "intermedia" runs into its neighbours
    bar.ax.tick_params(length=0)
    axes[1].set_title("Le zone", fontsize=9, loc="left", color=TEXT)
    figure.savefig(HERE / "era5_eventi_mappa_zone.pdf", dpi=300)
    plt.close(figure)


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
    """Each method across its own events: G falls more gradually than most baselines, all typical about six cells out."""
    figure, axis = plt.subplots(figsize=size(0.62), constrained_layout=True)
    axis.axhline(50, color=MUTED, linewidth=0.8, linestyle="--")
    axis.text(-8.2, 50, "valore tipico", fontsize=TICKS, color=MUTED, va="bottom")
    axis.axhline(95, color=MUTED, linewidth=0.8, linestyle=":")
    axis.text(8.2, 96, "soglia", fontsize=TICKS, color=MUTED, va="bottom", ha="right")
    for label, key, _, table in reversed(method_tables()):  # G last, on top
        table = table[table["type"] == "all"].sort_values("position")
        joined = table["position"].abs() < 8  # the last points gather all the cells beyond: not joined to the others
        if key == SCORE:
            style = dict(color=BLUE, linewidth=2.2, marker="o", markersize=5, markeredgecolor="white", zorder=4)
        elif key in SIMPLE:
            style = dict(color=SIMPLE[key], linewidth=1.6, marker="o", markersize=3.5, markeredgecolor="white", zorder=3)
        else:
            style = dict(color=SHALLOW_GREY, linewidth=0.9, marker="o", markersize=2, zorder=2)
        axis.plot(table["position"][joined], table[f"{key} percentile"][joined], **style)
        axis.plot(table["position"][~joined], table[f"{key} percentile"][~joined], **{**style, "linewidth": 0})
    # The distance from normal read on the events of G (the text of Section 6.7), not on its own events.
    table = pd.read_csv(PROFILE)
    table = table[table["type"] == "all"].sort_values("position")
    joined = table["position"].abs() < 8
    style = dict(color=PURPLE, linewidth=1.6, linestyle=(0, (4, 2)), marker="o", markersize=3.5, markeredgecolor="white",
                 zorder=3.5)
    axis.plot(table["position"][joined], table[f"{NORMAL} percentile"][joined], **style)
    axis.plot(table["position"][~joined], table[f"{NORMAL} percentile"][~joined], **{**style, "linestyle": "none"})
    axis.text(0.15, 3, "bordo del cluster", fontsize=TICKS, color=TEXT)
    keys = [Line2D([], [], color=BLUE, linewidth=2.2, marker="o", markersize=5, markeredgecolor="white",
                   label=r"$G_\mathrm{n}$"),
            Line2D([], [], color=ORANGE, linewidth=1.6, label="distanza dal normale, sui propri cluster"),
            Line2D([], [], color=PURPLE, linewidth=1.6, linestyle=(0, (4, 2)),
                   label=r"distanza dal normale, sui cluster di $G_\mathrm{n}$"),
            Line2D([], [], color=GREEN, linewidth=1.6, label="variazione in tre ore"),
            Line2D([], [], color=SHALLOW_GREY, linewidth=0.9, label="sei modelli shallow")]
    copod = method_tables()[list(BASELINES).index("copod") + 1][3]
    copod = copod[copod["type"] == "all"].set_index("position")["copod percentile"]
    axis.annotate("COPOD", (3, copod[3]), xytext=(4, 5), textcoords="offset points", fontsize=TICKS - 1, color=MUTED)
    axis.legend(handles=keys, frameon=False, fontsize=TICKS, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
                handlelength=1.8, columnspacing=1.5)
    axis.set_ylim(0, 106)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_ylabel("percentile mediano", fontsize=TICKS)
    position_axis(axis)
    figure.savefig(HERE / "era5_eventi_profilo.pdf")
    plt.close(figure)


def profile_types_figure():
    """One cell out of the core some variables are back to normal, others are not: the variable of each type."""
    table = pd.read_csv(PROFILE)
    table = table[(table["type"] != "all") & (table["values"] >= MIN_VALUES)]
    chosen = {"total_precipitation": (BLUE, BLUE, 3.6, 0.42, -12), "vertical_velocity": (GREEN, GREEN_TEXT, -7.6, 1.62, 0),
              "total_cloud_cover": (ORANGE, ORANGE, -5.3, 3.3, 0),
              "2m_temperature": (YELLOW, "#a87300", 0.3, 2.4, 0)}
    figure, axis = plt.subplots(figsize=size(0.56), constrained_layout=True)
    for name, group in table.groupby("type"):
        group = group.sort_values("position")
        if name not in chosen:
            joined = group["position"].abs() < 8
            axis.plot(group["position"][joined], group["variable distance"][joined], color="#c8c6bf", linewidth=0.9, zorder=1)
            axis.plot(group["position"][~joined], group["variable distance"][~joined], color="#c8c6bf", linewidth=0, marker="o", markersize=2.5, zorder=1)
    for name, (colour, label, x, y, shift) in chosen.items():
        group = table[table["type"] == name].sort_values("position")
        joined = group["position"].abs() < 8
        axis.plot(group["position"][joined], group["variable distance"][joined], color=colour, linewidth=2, zorder=2)
        axis.plot(group["position"][~joined], group["variable distance"][~joined], color=colour, linewidth=0, marker="o", markersize=5, markeredgecolor="white", zorder=2)
        axis.annotate(TYPES[name], (x, y), xytext=(0, shift), textcoords="offset points", fontsize=TICKS, color=label, zorder=3, path_effects=HALO)
    axis.annotate("altri tipi", (6.2, 1.62), fontsize=TICKS, color=MUTED)
    axis.text(0.15, 0.08, "bordo del cluster", fontsize=TICKS, color=TEXT)
    axis.set_ylim(0, 3.5)
    axis.set_ylabel("scarto della variabile (0 = normale)", fontsize=TICKS)
    position_axis(axis)
    figure.savefig(HERE / "era5_eventi_profilo_tipi.pdf")
    plt.close(figure)


def dropout_figure():
    """The dispersion in proportion to the error: by size of the error, across the events, at equal error."""
    data = json.loads((HERE / "dati" / "era5_dropout.json").read_text(encoding="utf-8"))
    figure, (bands, across, classes) = plt.subplots(1, 3, figsize=size(0.46), constrained_layout=True, width_ratios=(1.3, 1.2, 0.95))
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
    across.set_ylim(9, 11.5)
    across.set_yticks([9, 10, 11])
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
    figure.savefig(HERE / "era5_dropout.pdf")
    plt.close(figure)


def dropout_types_figure():
    """The dropout by type of event: the ratio across the events, and the classes at equal error."""
    positions = pd.read_csv(HERE / "dati" / "era5_dropout_posizioni.csv")
    positions = positions[(positions["type"] != "all") & (positions["values"] >= MIN_VALUES)]
    classes = pd.read_csv(HERE / "dati" / "era5_dropout_classi.csv")
    classes = classes[(classes["type"] != "all") & (classes["spread"] == "ratio") & (classes["bands"] == "raw error")]
    # The three types drawn in colour are named in a legend: the lines of precipitation and vertical velocity
    # overlap outside the events, and a label written next to them would be ambiguous.
    chosen = {"total_precipitation": BLUE, "vertical_velocity": GREEN, "total_cloud_cover": ORANGE}
    figure, (across, gap) = plt.subplots(1, 2, figsize=size(0.62), constrained_layout=True, width_ratios=(1.25, 1))
    for name, group in positions.groupby("type"):
        if name not in chosen:
            group = group.sort_values("position")
            joined = group["position"].abs() < 8
            across.plot(group["position"][joined], 100 * group["ratio value"][joined], color="#c8c6bf", linewidth=0.9, zorder=1)
    for name, colour in chosen.items():
        group = positions[positions["type"] == name].sort_values("position")
        joined = group["position"].abs() < 8
        across.plot(group["position"][joined], 100 * group["ratio value"][joined], color=colour, linewidth=2, zorder=2,
                    label=TYPES[name])
    across.set_ylim(4.6, None)  # room for the legend under the lines
    across.legend(frameon=False, fontsize=7, loc="lower right", handlelength=1.4)
    across.annotate("umidità del suolo", (-4.6, 12.9), fontsize=7.5, color=MUTED)
    across.annotate("radiazione solare", (-6.2, 7.0), fontsize=7.5, color=MUTED)
    position_axis(across)
    across.set_xticks([-6, -4, -2, 2, 4, 6], ["6", "4", "2", "2", "4", "6"])
    across.set_xlabel("celle dal bordo: dentro | fuori", fontsize=TICKS)
    across.set_ylabel("rapporto (%)", fontsize=TICKS)
    across.set_title("Il rapporto attraverso i cluster", fontsize=9, loc="left", color=TEXT)

    # Classes 1, 4 and 5: the share rises from class 1 to class 3 or 4 in every type, and in most types falls in class 5.
    table = classes.pivot(index="type", columns="class", values="normal flagged %")
    table = table.loc[table[4].sort_values().index]
    place = np.arange(len(table))
    gap.hlines(place, table[[1, 4, 5]].min(axis=1), table[[1, 4, 5]].max(axis=1), color=GRID, linewidth=1.5, zorder=1)
    gap.scatter(table[1], place, s=26, color=BLUE, edgecolor="white", linewidth=0.6, zorder=2, label="classe 1 (concordi)")
    gap.scatter(table[4], place, s=26, color=MUTED, edgecolor="white", linewidth=0.6, zorder=2, label="classe 4")
    gap.scatter(table[5], place, s=26, color=ORANGE, edgecolor="white", linewidth=0.6, zorder=3, label="classe 5 (discordi)")
    gap.set_yticks(place, [TYPES[name] for name in table.index], fontsize=7)
    gap.set_xlim(0, 100)
    gap.set_xlabel("oltre la soglia della distanza\ndal normale (%)", fontsize=TICKS)
    gap.tick_params(labelsize=7, length=0)
    gap.grid(axis="x", color=GRID, linewidth=0.6)
    gap.set_axisbelow(True)
    figure.legend(*gap.get_legend_handles_labels(), frameon=False, fontsize=7.5, ncol=3, loc="outside lower right",
                  columnspacing=0.8)
    gap.set_title("A parità di errore", fontsize=9, loc="left", color=TEXT)
    clean(gap, "top", "right", "left")
    figure.savefig(HERE / "era5_dropout_tipi.pdf")
    plt.close(figure)


def dropout_maps_figure(coastline_path):
    """The standard deviation of the draws cell by cell: its mean over the years, and at the instants of the cases."""
    maps = dict(np.load(HERE / "dati" / "era5_dropout_mappe.npz"))
    grid = {key: maps[key] for key in ("rows", "cols", "latitudes", "longitudes")}
    coastline = coastline_of(coastline_path, grid)
    figure = plt.figure(figsize=size(0.98), constrained_layout=True)
    # One grid per row: in a single grid the margins of the top maps would also narrow the small maps of the bottom row.
    rows = figure.add_gridspec(2, 1, height_ratios=(1.14, 1.0))
    top, bottom = rows[0].subgridspec(1, 2), rows[1].subgridspec(1, 3)
    for slot, key, title, label, scale in ((top[0, 0], "mean grezza", "Deviazione standard, media", r"deviazione standard dell'errore, $\times 10^{-4}$", 1e4),
                                           (top[0, 1], "mean ratio", "Rapporto, medio", "rapporto (%)", 100)):
        axis = figure.add_subplot(slot)
        values = maps[key] * scale
        low, high = np.nanpercentile(values, (1, 99))
        mesh = draw_map(axis, grid, values, coastline, left=slot == top[0, 0], cmap="viridis", vmin=low, vmax=high)
        axis.set_title(title, fontsize=9, loc="left", color=TEXT)
        bar = figure.colorbar(mesh, ax=axis, orientation="horizontal", shrink=0.9, pad=0.02, aspect=30)
        bar.set_label(label, fontsize=TICKS)
        bar.ax.tick_params(labelsize=TICKS)
    # The instants are those of the cases (check_era5_event_cases.py), given to check_era5_dropout_flags.py --instants
    # as the "timestamp" column of cases.csv.
    cases = []
    for path in sorted((HERE / "dati").glob("era5_caso_*.npz")):
        with np.load(path) as case:
            record = json.loads(str(case["case"]))
        moment = pd.Timestamp(record["timestamp"])
        cases.append((path.stem[-2:], record["timestamp"],
                      f"{TYPES[record['type']].replace(' al livello del mare', '')}\n{moment.day}/{moment.month}/{moment.year}"))
    axes = []
    high = max(np.nanpercentile(maps[f"instant {moment} grezza"], 99) for _, moment, _ in cases) * 1e4
    for column, (number, moment, title) in enumerate(cases):
        axis = figure.add_subplot(bottom[0, column])
        mesh = draw_map(axis, grid, maps[f"instant {moment} grezza"] * 1e4, coastline, left=column == 0, cmap="viridis", vmin=0, vmax=high)
        with np.load(HERE / "dati" / f"era5_caso_{number}.npz") as case:
            picture = np.zeros((len(grid["latitudes"]), len(grid["longitudes"])))
            picture[grid["rows"], grid["cols"]] = case["core"]
        axis.contour(grid["longitudes"], grid["latitudes"][::-1], picture[::-1], levels=[0.5], colors=ORANGE, linewidths=1.6)
        axis.set_title(title, fontsize=8, loc="left", color=TEXT)
        if column > 0:  # the latitudes are those of the first map: without them the small maps are wider
            axis.tick_params(labelleft=False)
        axes.append(axis)
    bar = figure.colorbar(mesh, ax=axes, orientation="horizontal", shrink=0.5, pad=0.02, aspect=40)
    bar.set_label(r"deviazione standard dell'errore, $\times 10^{-4}$", fontsize=TICKS)
    bar.ax.tick_params(labelsize=TICKS)
    figure.savefig(HERE / "era5_dropout_mappe.pdf", dpi=300)
    plt.close(figure)


def percentile(values, quantiles):
    return np.interp(values, quantiles, np.linspace(0, 100, len(quantiles)))


def case_figure(number, data, coastline):
    """A case: where it is, how unusual, when it starts and ends, and how long the chain it belongs to lasts."""
    record = json.loads(str(data["case"]))
    moment = pd.Timestamp(record["timestamp"])
    grid = {key: data[key] for key in ("rows", "cols", "latitudes", "longitudes")}
    longitudes, latitudes = data["longitudes"], data["latitudes"]
    picture = np.zeros((len(latitudes), len(longitudes)))
    picture[data["rows"], data["cols"]] = data["core"]
    figure = plt.figure(figsize=size(0.86), constrained_layout=True)
    panels = figure.add_gridspec(2, 2, height_ratios=(1.75, 1))
    left, right = figure.add_subplot(panels[0, 0]), figure.add_subplot(panels[0, 1])
    limit = 4
    mesh = draw_map(left, grid, data["input"], coastline, cmap="RdBu_r", vmin=-limit, vmax=limit)
    bar = figure.colorbar(mesh, ax=left, orientation="horizontal", shrink=0.9, pad=0.02, aspect=30, extend="both")
    bar.set_label("scarto (0 = normale, 1 = variabilità abituale)", fontsize=TICKS)
    bar.ax.tick_params(labelsize=TICKS)
    left.set_title(f"{TYPES[record['type']]}: scarto", fontsize=9, loc="left", color=TEXT)
    mesh = draw_map(right, grid, percentile(data[f"score_{SCORE}"], data[f"quantiles_{SCORE}"]), coastline, left=False,
                    cmap="viridis", vmin=0, vmax=100)
    bar = figure.colorbar(mesh, ax=right, orientation="horizontal", shrink=0.9, pad=0.02, aspect=30, ticks=[0, 25, 50, 75, 95])
    bar.set_label("percentile (95 circa = soglia)", fontsize=TICKS)
    bar.ax.tick_params(labelsize=TICKS)
    right.set_title(r"$G_\mathrm{n}$", fontsize=9, loc="left", color=TEXT)
    for axis in (left, right):
        axis.contour(longitudes, latitudes[::-1], picture[::-1], levels=[0.5], colors=TEXT, linewidths=0.6)

    times = pd.DatetimeIndex(data["series_times"])
    cell = figure.add_subplot(panels[1, 0])
    flagged = data["series_core"]
    for start, stop in zip(np.nonzero(np.diff(np.r_[0, flagged.astype(int)]) == 1)[0],
                           np.nonzero(np.diff(np.r_[flagged.astype(int), 0]) == -1)[0]):
        half = pd.Timedelta(hours=1.5)  # a band of one timestamp would have no width
        cell.axvspan(times[start] - half, times[stop] + half, color="#cde2fb", linewidth=0)
    cell.axvline(moment, color=TEXT, linewidth=0.8)
    cell.axhline(95, color=MUTED, linewidth=0.8, linestyle="--")
    cell.plot(times, percentile(data[f"series_{SCORE}"], data[f"quantiles_{SCORE}"]), color=BLUE, linewidth=1.6)
    cell.plot(times, percentile(data[f"series_{NORMAL}"], data[f"quantiles_{NORMAL}"]), color=ORANGE, linewidth=1.6)
    # The variable of the type in the same cell, on its own axis: the text says how far it is from its normal.
    gap = cell.twinx()
    gap.plot(times, data["series_input"], color=GREEN, linewidth=1.1)
    reach = max(3.0, float(np.ceil(np.nanmax(np.abs(data["series_input"])))))
    gap.set_ylim(-reach, reach)
    gap.set_ylabel("scarto", fontsize=TICKS, color=GREEN_TEXT)
    gap.tick_params(labelsize=TICKS, length=2, colors=GREEN_TEXT)
    gap.spines["top"].set_visible(False)
    keys = (Line2D([], [], color=BLUE, linewidth=1.6, label=r"$G_\mathrm{n}$"), Patch(color="#cde2fb", linewidth=0, label="cella nel cluster"),
            Line2D([], [], color=ORANGE, linewidth=1.6, label="distanza dal normale"), Line2D([], [], color=TEXT, linewidth=0.8, label="istante delle mappe"),
            Line2D([], [], color=GREEN, linewidth=1.1, label="scarto (asse destro)"))
    cell.legend(handles=keys, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, columnspacing=1.2)
    cell.set_ylim(0, 102)
    cell.set_yticks([0, 50, 95])
    cell.set_ylabel("percentile", fontsize=TICKS)
    cell.margins(x=0)
    cell.tick_params(labelsize=TICKS, length=2)
    cell.xaxis.set_major_locator(dates.DayLocator(interval=2))
    cell.xaxis.set_major_formatter(dates.DateFormatter("%d/%m"))
    east = record["cell longitude"]
    where = f"{east:g}°E" if east >= 0 else f"{-east:g}°W"
    cell.set_title(f"Una cella del nucleo ({record['cell latitude']:g}°N, {where})", fontsize=9, loc="left", color=TEXT)
    clean(cell, "top", "right")

    chain = figure.add_subplot(panels[1, 1])
    event_times = pd.DatetimeIndex(data["event_times"])
    chain.fill_between(event_times, data["event_cells"], step="mid", color=BLUE, linewidth=0)
    chain.axvline(moment, color=TEXT, linewidth=0.8)
    hours = record.get("hours", (event_times[-1] - event_times[0]) / pd.Timedelta(hours=1))
    chain.set_title(f"Il cluster dura {hours:.0f} ore", fontsize=9, loc="left", color=TEXT)
    chain.set_ylabel("celle del cluster", fontsize=TICKS)
    chain.set_ylim(0, None)
    chain.margins(x=0)
    chain.tick_params(labelsize=TICKS, length=2)
    # A tick at 00 UTC of every day, so that the date under it is the day it marks.
    chain.xaxis.set_major_locator(dates.DayLocator(interval=max(1, len(event_times) // 40)))
    chain.xaxis.set_major_formatter(dates.DateFormatter("%d/%m"))
    clean(chain, "top", "right")
    figure.savefig(HERE / f"era5_caso_{number:02d}.pdf", dpi=300)
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
    dropout_figure()
    dropout_types_figure()
    dropout_maps_figure(args.coastline)
    medians = {}
    for path in sorted((HERE / "dati").glob("era5_caso_*.npz")):
        with np.load(path) as data:
            number = int(path.stem[-2:])
            arrays = {key: data[key] for key in data.files}
            coastline = coastline_of(args.coastline, arrays)
            case_figure(number, arrays, coastline)
            if number == 0:
                zones_map_figure(arrays, coastline)
            medians[json.loads(str(arrays["case"]))["type"]] = {"record": json.loads(str(arrays["case"])), **case_medians(arrays)}
    (HERE / "dati" / "era5_casi_zone.json").write_text(json.dumps(medians, indent=1), encoding="utf-8")
    print(json.dumps({name: {zone: value[zone] for zone in ("core", "edge", "far")} for name, value in medians.items()}, indent=1))


if __name__ == "__main__":
    main()

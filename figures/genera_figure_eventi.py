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

from genera_figure_g_d import BLUE, GRID, MUTED, ORANGE, TEXT, TICKS, YELLOW, coastline_of, draw_map, size

HERE = pathlib.Path(__file__).parent
ZONES = HERE / "dati" / "era5_eventi_zone.csv"
PROFILE = HERE / "dati" / "era5_eventi_profilo.csv"
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


TYPE_COLUMNS = ("vertical_velocity", "total_precipitation", "total_cloud_cover", "volumetric_soil_water_layer_1",
                "friction_velocity", "boundary_layer_height")


def types_figure(zones):
    """Which variables dominate for G_n and for every baseline: the share of the flagged values by type of cluster."""
    rows, labels = [], []
    for label, key, table, _ in method_tables():
        table = table[(table["type"] != "all") & ~table["type"].str.startswith("far") & table["size"].isin(SIZES)]
        shares = table.groupby("type")[f"{key} core values"].sum()
        shares = 100 * shares / shares.sum()
        row = [shares.get(name, 0.0) for name in TYPE_COLUMNS]
        rows.append(row + [100 - sum(row)])
        labels.append(label)
    grid = np.array(rows)
    names = [TYPES[name] for name in TYPE_COLUMNS] + ["Altri tipi"]
    figure, axis = plt.subplots(figsize=size(0.5), constrained_layout=True)
    axis.imshow(grid, cmap=RAMP, vmin=0, vmax=70, aspect="auto", interpolation="nearest")
    for (row, column), value in np.ndenumerate(grid):
        axis.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=7.5,
                  color="white" if value > 35 else TEXT)
    heads = {"Precipitazione": "Precipi-\ntazione"}
    axis.set_xticks(range(len(names)), [heads.get(name, name.replace(" ", "\n", 1)) for name in names], fontsize=7)
    axis.xaxis.tick_top()
    axis.set_yticks(range(len(labels)), labels, fontsize=7.5)
    axis.get_yticklabels()[0].set_color(BLUE)
    axis.get_yticklabels()[0].set_fontweight("bold")
    axis.axhline(0.5, color="white", linewidth=3)  # G_n apart from the baselines
    axis.tick_params(length=0)
    for side in axis.spines.values():
        side.set_visible(False)
    figure.savefig(HERE / "era5_eventi_tipo.pdf")
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
    """Left: each method across its own clusters, G falls more slowly than most baselines. Right: on the clusters of G,
    the distance from normal has the same profile as G. Only the positions up to 7 cells, joined by a line."""
    figure, (own, same) = plt.subplots(1, 2, figsize=size(0.5), sharey=True, constrained_layout=True)
    for axis in (own, same):
        axis.axhline(50, color=MUTED, linewidth=0.8, linestyle="--")
        axis.axhline(95, color=MUTED, linewidth=0.8, linestyle=":")
        position_axis(axis)
        axis.set_xlim(-7.3, 7.3)
        axis.set_xticks([-6, -4, -2, 2, 4, 6], ["6", "4", "2", "2", "4", "6"])
        axis.set_xlabel("celle dal bordo: dentro | fuori", fontsize=TICKS)
    own.text(-7.1, 51, "valore tipico", fontsize=TICKS - 1, color=MUTED, va="bottom")
    own.text(-7.1, 92, "soglia", fontsize=TICKS - 1, color=MUTED, va="top")
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
    own.set_ylim(0, 104)
    own.set_yticks([0, 25, 50, 75, 100])
    own.set_ylabel("percentile mediano", fontsize=TICKS)
    end_labels(own, labels)

    table = shown(pd.read_csv(PROFILE))
    for column, colour, width in ((f"{NORMAL} percentile", ORANGE, 1.6), (f"{SCORE} percentile", BLUE, 2.6)):
        same.plot(table["position"], table[column], color=colour, linewidth=width)
    same.set_title(r"Sui cluster di $G_\mathrm{n}$", fontsize=9, loc="left", color=TEXT)
    last = table.iloc[-1]
    end_labels(same, ((last[f"{SCORE} percentile"], r"$G_\mathrm{n}$", BLUE),
                      (last[f"{NORMAL} percentile"], "distanza\ndal normale", ORANGE)))
    for axis in (own, same):
        axis.text(0.15, 3, "bordo", fontsize=TICKS - 1, color=TEXT)
    figure.savefig(HERE / "era5_eventi_profilo.pdf")
    plt.close(figure)


def profile_types_figure():
    """One cell out of the core some variables are back to normal, others are not: the variable of each type."""
    table = pd.read_csv(PROFILE)
    table = table[(table["type"] != "all") & (table["values"] >= MIN_VALUES)]
    chosen = {"total_precipitation": (BLUE, BLUE, 3.6, 0.42, -12), "vertical_velocity": (GREEN, GREEN_TEXT, -7.0, 1.62, 0),
              "total_cloud_cover": (ORANGE, ORANGE, -5.3, 3.3, 0),
              "2m_temperature": (YELLOW, "#a87300", 0.3, 2.4, 0)}
    figure, axis = plt.subplots(figsize=size(0.56), constrained_layout=True)
    for name, group in table.groupby("type"):
        group = group.sort_values("position")
        if name not in chosen:
            joined = group["position"].abs() < 8
            axis.plot(group["position"][joined], group["variable distance"][joined], color="#c8c6bf", linewidth=0.9, zorder=1)
    for name, (colour, label, x, y, shift) in chosen.items():
        group = table[table["type"] == name].sort_values("position")
        joined = group["position"].abs() < 8
        axis.plot(group["position"][joined], group["variable distance"][joined], color=colour, linewidth=2, zorder=2)
        axis.annotate(TYPES[name], (x, y), xytext=(0, shift), textcoords="offset points", fontsize=TICKS, color=label, zorder=3, path_effects=HALO)
    axis.annotate("altri tipi", (6.2, 1.62), fontsize=TICKS, color=MUTED)
    axis.text(0.15, 0.08, "bordo del cluster", fontsize=TICKS, color=TEXT)
    axis.set_ylim(0, 3.5)
    axis.set_ylabel("|scarto| della variabile (0 = normale)", fontsize=TICKS)
    position_axis(axis)
    axis.set_xlim(-7.3, 7.3)
    axis.set_xticks([-6, -4, -2, -1, 1, 2, 4, 6], ["6", "4", "2", "1", "1", "2", "4", "6"])
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
    figure.savefig(HERE / "era5_dropout.pdf")
    plt.close(figure)


def dropout_maps_figure(coastline_path):
    """The standard deviation of the draws cell by cell, mean over the test years, and its mean ratio to the error."""
    maps = dict(np.load(HERE / "dati" / "era5_dropout_mappe.npz"))
    grid = {key: maps[key] for key in ("rows", "cols", "latitudes", "longitudes")}
    coastline = coastline_of(coastline_path, grid)
    figure, axes = plt.subplots(1, 2, figsize=size(0.56), constrained_layout=True)
    for axis, key, title, label, scale in ((axes[0], "mean grezza", "Deviazione standard, media", r"deviazione standard dell'errore, $\times 10^{-4}$", 1e4),
                                           (axes[1], "mean ratio", "Rapporto, medio", "rapporto (%)", 100)):
        values = maps[key] * scale
        low, high = np.nanpercentile(values, (1, 99))
        mesh = draw_map(axis, grid, values, coastline, left=axis is axes[0], cmap="viridis", vmin=low, vmax=high)
        axis.set_title(title, fontsize=9, loc="left", color=TEXT)
        bar = figure.colorbar(mesh, ax=axis, orientation="horizontal", shrink=0.9, pad=0.02, aspect=30)
        bar.set_label(label, fontsize=TICKS)
        bar.ax.tick_params(labelsize=TICKS)
    figure.savefig(HERE / "era5_dropout_mappe.pdf", dpi=300)
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
    figure = plt.figure(figsize=size(0.9), constrained_layout=True)
    panels = figure.add_gridspec(2, 2, height_ratios=(1.55, 1.2))
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
        # The cell of the series below: a ring with a white halo, visible on every colour of the maps.
        axis.plot(record["cell longitude"], record["cell latitude"], marker="o", markersize=9, markerfacecolor="none",
                  markeredgecolor="white", markeredgewidth=2.6, zorder=5)
        axis.plot(record["cell longitude"], record["cell latitude"], marker="o", markersize=9, markerfacecolor="none",
                  markeredgecolor=TEXT, markeredgewidth=1.2, zorder=6)

    # The followed cell in two panels with the same time axis: the percentiles of the two scores, and the scarto of the
    # variable of the type, which the text uses to say when the variable leaves its normal.
    times = pd.DatetimeIndex(data["series_times"])
    rows = panels[1, 0].subgridspec(2, 1, height_ratios=(1.5, 1), hspace=0.08)
    cell = figure.add_subplot(rows[0])
    gap = figure.add_subplot(rows[1], sharex=cell)
    flagged = data["series_core"]
    bands = list(zip(np.nonzero(np.diff(np.r_[0, flagged.astype(int)]) == 1)[0],
                     np.nonzero(np.diff(np.r_[flagged.astype(int), 0]) == -1)[0]))
    half = pd.Timedelta(hours=1.5)  # a band of one timestamp would have no width
    for axis in (cell, gap):
        for begin, end in bands:
            axis.axvspan(times[begin] - half, times[end] + half, color=BAND, linewidth=0)
        axis.axvline(moment, color=TEXT, linewidth=0.8)
        axis.margins(x=0)
        axis.tick_params(labelsize=TICKS, length=2)
        clean(axis, "top", "right")
    cell.axhline(95, color=MUTED, linewidth=0.8, linestyle="--")
    level = percentile(data[f"series_{SCORE}"], data[f"quantiles_{SCORE}"])
    normal = percentile(data[f"series_{NORMAL}"], data[f"quantiles_{NORMAL}"])
    cell.plot(times, normal, color=ORANGE, linewidth=1.6, zorder=2)
    cell.plot(times, level, color=BLUE, linewidth=1.6, zorder=3)
    cell.set_ylim(0, 104)
    cell.set_yticks([0, 50, 95], ["0", "50", "soglia"])
    cell.set_ylabel("percentile", fontsize=TICKS)
    cell.tick_params(labelbottom=False)
    east = record["cell longitude"]
    where = f"{east:g}°E" if east >= 0 else f"{-east:g}°W"
    cell.set_title(f"La cella cerchiata ({record['cell latitude']:g}°N, {where})", fontsize=9, loc="left", color=TEXT)
    gap.axhline(0, color=MUTED, linewidth=0.8)
    gap.plot(times, data["series_input"], color=GREEN, linewidth=1.4)
    reach = max(3.0, float(np.ceil(np.nanmax(np.abs(data["series_input"])))))
    gap.set_ylim(-reach, 1.4 * reach)  # free room at the top for "nel cluster"
    gap.set_yticks([-reach, 0, reach], [f"{-reach:g}", "0", f"+{reach:g}"])
    gap.set_ylabel("scarto", fontsize=TICKS)
    gap.xaxis.set_major_locator(dates.DayLocator(interval=2))
    gap.xaxis.set_major_formatter(dates.DateFormatter("%d/%m"))
    widest = max(bands, key=lambda band: band[1] - band[0])
    middle = times[widest[0]] + (times[widest[1]] - times[widest[0]]) / 2
    # The names of the lines at their right end; "nel cluster" over the widest band, at the top of the lower panel.
    end_labels(cell, ((level[-1], r"$G_\mathrm{n}$", BLUE), (normal[-1], "distanza\ndal normale", ORANGE)))
    end_labels(gap, ((data["series_input"][-1], TYPES[record["type"]].lower().replace(" ", "\n", 1), GREEN_TEXT),))
    gap.annotate("nel cluster", (middle, 1), xycoords=("data", "axes fraction"), xytext=(0, -1), textcoords="offset points",
                 fontsize=6.5, color="#3d6fa8", ha="center", va="top", path_effects=HALO, zorder=5)

    chain = figure.add_subplot(panels[1, 1])
    event_times = pd.DatetimeIndex(data["event_times"])
    chain.fill_between(event_times, data["event_cells"], step="mid", color=BLUE, linewidth=0)
    chain.axvline(moment, color=TEXT, linewidth=0.8)
    chain.annotate("istante\ndelle mappe", (moment, 1), xycoords=("data", "axes fraction"), xytext=(3, -2),
                   textcoords="offset points", fontsize=6.5, color=TEXT, ha="left", va="top", path_effects=HALO)
    hours = record.get("hours", (event_times[-1] - event_times[0]) / pd.Timedelta(hours=1))
    chain.set_title(f"Il cluster dura {hours:.0f} ore", fontsize=9, loc="left", color=TEXT)
    chain.set_ylabel("celle del cluster", fontsize=TICKS)
    chain.set_ylim(0, 1.3 * float(np.max(data["event_cells"])))  # room for the name of the line over the bars
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

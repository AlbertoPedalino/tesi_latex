"""Generate the figures of Sections 6.7 and 6.8: the events of G with the normalisation, by size and type, its zones and
the specific cases. Made for a reader who is not an expert of the domain: one message for every figure.

Run from the thesis folder. First, once, gather the data from the folder downloaded from the server
(outputs/casi_good2_g, written by scripts/check_era5_event_zones.py and check_era5_event_cases.py of the code
repository):
    python figures/genera_figure_eventi.py --prepare <folder with the downloaded files>
which writes figures/dati/era5_eventi_zone.csv and era5_caso_NN.npz. Then:
    python figures/genera_figure_eventi.py --coastline <ne_50m_coastline.geojson>
which writes figures/era5_eventi_dimensione.pdf, era5_eventi_tipo.pdf, era5_eventi_zone.pdf,
era5_eventi_zone_tipi.pdf (from figures/dati/era5_eventi_zone_tipi.csv, the `_types.csv` of the zones), era5_caso_NN.pdf, and figures/dati/era5_casi_zone.json with the medians of every case.
The coastline, from Natural Earth (public domain), is not kept in the repository.

Run: good-sweep-2, test years 2005-2025, threshold from the validation years.
"""
import argparse
import json
import pathlib
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from genera_figure_g_d import BLUE, GRID, MUTED, ORANGE, TEXT, TICKS, coastline_of, draw_map, size

HERE = pathlib.Path(__file__).parent
ZONES = HERE / "dati" / "era5_eventi_zone.csv"
SIZES = ("1", "2-4", "5-19", "20-99", "100-999", "1000-9999", ">9999")
SIZE_LABELS = ("1", "2–4", "5–19", "20–99", "100–999", "1000–\n9999", "oltre\n9999")
TYPES = {"vertical_velocity": "Velocità verticale", "total_precipitation": "Precipitazione",
         "mean_sea_level_pressure": "Pressione al mare", "total_cloud_cover": "Copertura nuvolosa",
         "boundary_layer_height": "Strato limite", "specific_humidity": "Umidità specifica",
         "volumetric_soil_water_layer_1": "Umidità del suolo", "friction_velocity": "Velocità di attrito",
         "surface_solar_radiation_downwards": "Radiazione solare", "v_component_of_wind": "Vento nord-sud in quota",
         "u_component_of_wind": "Vento est-ovest in quota", "2m_temperature": "Temperatura al suolo",
         "temperature": "Temperatura in quota", "10m_v_component_of_wind": "Vento nord-sud al suolo",
         "10m_u_component_of_wind": "Vento est-ovest al suolo"}
RAMP = LinearSegmentedColormap.from_list("blu", ("#eef4fc", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"))
SCORE, NORMAL = "G+clim", "normale"
CORE, EDGE = "core", "edge"


def prepare(folder):
    folder = pathlib.Path(folder)
    shutil.copyfile(folder / "zone_5pc.csv", ZONES)
    for path in sorted(folder.glob("case_*.npz")):
        shutil.copyfile(path, HERE / "dati" / f"era5_caso_{path.stem[-2:]}.npz")
    print(f"written {ZONES}")


def read_zones():
    zones = pd.read_csv(ZONES, dtype={"size": str})
    far = zones[zones["type"].str.startswith("far")].iloc[0]
    return zones[~zones["type"].str.startswith("far")].set_index(["type", "size"]), far


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
    for axis, shares, title in ((axes[0], 100 * events / events.sum(), "Quota degli eventi"),
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
    axes[0].set_ylabel("valori dell'evento", fontsize=TICKS)
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
    axis.set_xlabel("valori dell'evento", fontsize=TICKS)
    axis.tick_params(length=0, labelsize=TICKS)
    for side in axis.spines.values():
        side.set_visible(False)


def types_figure(zones):
    """Which variables dominate, and at which size: the values flagged, by type and size."""
    order = type_order(zones)
    table = np.full((len(order), len(SIZES)), np.nan)
    for row, name in enumerate(order):
        for column, band in enumerate(SIZES):
            if (name, band) in zones.index:
                table[row, column] = zones.loc[(name, band), f"{SCORE} {CORE} values"]
    norm = LogNorm(vmin=5e3, vmax=1.5e7)
    figure, axis = plt.subplots(figsize=size(0.62), constrained_layout=True)
    axis.imshow(table, cmap=RAMP, norm=norm, aspect="auto")
    for (row, column), value in np.ndenumerate(table):
        if np.isfinite(value):
            axis.text(column, row, compact(value), ha="center", va="center", fontsize=7,
                      color="white" if norm(value) > 0.55 else TEXT)
    table_axes(axis, order)
    figure.savefig(HERE / "era5_eventi_tipo.pdf")
    plt.close(figure)


def zones_figure(zones, far):
    """Where G flags the variables are far from normal; moving away they go back to normal."""
    everything = zones.loc[("all", "all")]
    labels = ("Nucleo\n(celle segnalate da G)", "Contorno\n(le 8 celle attorno)", "Zona lontana\n(oltre 3 celle)")
    series = (("G con normalizzazione", SCORE, BLUE, (0.12, 5), "left"), ("Distanza dal normale", NORMAL, ORANGE, (0.12, -8), "left"))
    figure, axis = plt.subplots(figsize=size(0.5), constrained_layout=True)
    axis.axhline(50, color=MUTED, linewidth=0.8, linestyle="--")
    axis.text(-0.33, 50, "valore tipico", fontsize=TICKS, color=MUTED, va="bottom", ha="left")
    for name, key, colour, _, _ in series:
        points = [everything[f"{key} {CORE} percentile"], everything[f"{key} {EDGE} percentile"], far[f"{key} {CORE} percentile"]]
        axis.plot(range(3), points, color=colour, linewidth=2, marker="o", markersize=6, markeredgecolor="white", label=name)
        above = key == SCORE
        for x, y in enumerate(points):
            # At the far zone the second line is the higher one.
            high = above if x < 2 else not above
            axis.annotate(f"{y:.0f}", (x, y), xytext=(0, 7 if high else -13), textcoords="offset points", ha="center",
                          fontsize=TICKS, color=TEXT)
    axis.legend(frameon=False, fontsize=TICKS, loc="lower left")
    axis.set_xlim(-0.35, 2.45)
    axis.set_ylim(0, 108)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_xticks(range(3), labels)
    axis.set_ylabel("percentile mediano", fontsize=TICKS)
    axis.tick_params(labelsize=TICKS, length=0)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    clean(axis, "top", "right", "left")
    figure.savefig(HERE / "era5_eventi_zone.pdf")
    plt.close(figure)


def types_zones_figure():
    """For some types the variable is back near its normal one cell from the core, for others it is not."""
    table = pd.read_csv(HERE / "dati" / "era5_eventi_zone_tipi.csv").set_index("type")
    table = table.loc[(table["edge distance"] / table["core distance"]).sort_values().index]
    place = np.arange(len(table))[::-1]
    figure, axis = plt.subplots(figsize=size(0.62), constrained_layout=True)
    axis.hlines(place, table[["core distance", "edge distance", "far distance"]].min(axis=1),
                table[["core distance", "edge distance", "far distance"]].max(axis=1), color=GRID, linewidth=1.5, zorder=1)
    for column, label, colour in (("far distance", "Zona lontana", MUTED), ("edge distance", "Contorno", ORANGE),
                                  ("core distance", "Nucleo", BLUE)):
        axis.scatter(table[column], place, s=34, color=colour, edgecolor="white", linewidth=0.8, zorder=2, label=label)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles[::-1], labels[::-1], frameon=False, fontsize=TICKS, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    axis.set_yticks(place, [TYPES[name] for name in table.index])
    axis.set_xlim(0, 2.8)
    axis.set_xlabel("distanza della variabile dal normale (0 = normale)", fontsize=TICKS)
    axis.tick_params(labelsize=TICKS, length=0)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    clean(axis, "top", "right", "left")
    figure.savefig(HERE / "era5_eventi_zone_tipi.pdf")
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
    bar.set_label("distanza dal normale (0 = normale)", fontsize=TICKS)
    bar.ax.tick_params(labelsize=TICKS)
    left.set_title(TYPES[record["type"]], fontsize=9, loc="left", color=TEXT)
    mesh = draw_map(right, grid, percentile(data[f"score_{SCORE}"], data[f"quantiles_{SCORE}"]), coastline, left=False,
                    cmap="viridis", vmin=0, vmax=100)
    bar = figure.colorbar(mesh, ax=right, orientation="horizontal", shrink=0.9, pad=0.02, aspect=30, ticks=[0, 25, 50, 75, 95])
    bar.set_label("G, percentile (95 circa = soglia)", fontsize=TICKS)
    bar.ax.tick_params(labelsize=TICKS)
    right.set_title("Dove G segnala", fontsize=9, loc="left", color=TEXT)
    for axis in (left, right):
        axis.contour(longitudes, latitudes[::-1], picture[::-1], levels=[0.5], colors=TEXT, linewidths=0.6)

    times = pd.DatetimeIndex(data["series_times"])
    cell = figure.add_subplot(panels[1, 0])
    flagged = data["series_core"]
    for start, stop in zip(np.nonzero(np.diff(np.r_[0, flagged.astype(int)]) == 1)[0],
                           np.nonzero(np.diff(np.r_[flagged.astype(int), 0]) == -1)[0]):
        cell.axvspan(times[start], times[stop], color="#cde2fb", linewidth=0)
    cell.axvline(moment, color=TEXT, linewidth=0.8)
    cell.axhline(95, color=MUTED, linewidth=0.8, linestyle="--")
    cell.plot(times, percentile(data[f"series_{SCORE}"], data[f"quantiles_{SCORE}"]), color=BLUE, linewidth=1.6, label="G")
    cell.plot(times, percentile(data[f"series_{NORMAL}"], data[f"quantiles_{NORMAL}"]), color=ORANGE, linewidth=1.6,
              label="distanza dal normale")
    cell.legend(frameon=False, fontsize=TICKS, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, columnspacing=1.2)
    cell.set_ylim(0, 102)
    cell.set_yticks([0, 50, 95])
    cell.set_ylabel("percentile", fontsize=TICKS)
    cell.margins(x=0)
    cell.tick_params(labelsize=TICKS, length=2)
    cell.xaxis.set_major_locator(dates.DayLocator(interval=2))
    cell.xaxis.set_major_formatter(dates.DateFormatter("%d/%m"))
    cell.set_title(f"Una cella del nucleo ({record['cell latitude']:g}°N, {record['cell longitude']:g}°E)", fontsize=9, loc="left", color=TEXT)
    clean(cell, "top", "right")

    chain = figure.add_subplot(panels[1, 1])
    event_times = pd.DatetimeIndex(data["event_times"])
    chain.fill_between(event_times, data["event_cells"], step="mid", color=BLUE, linewidth=0)
    chain.axvline(moment, color=TEXT, linewidth=0.8)
    chain.set_title(f"La catena dura {len(event_times) * 3 / 24:.0f} giorni", fontsize=9, loc="left", color=TEXT)
    chain.set_ylabel("celle segnalate", fontsize=TICKS)
    chain.set_ylim(0, None)
    chain.margins(x=0)
    chain.tick_params(labelsize=TICKS, length=2)
    chain.set_xticks(event_times[[0, len(event_times) // 2, -1]])
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
    parser.add_argument("--coastline", type=pathlib.Path, help="Natural Earth coastline, GeoJSON")
    args = parser.parse_args()
    if args.prepare is not None:
        prepare(args.prepare)
    zones, far = read_zones()
    sizes_figure(zones)
    types_figure(zones)
    zones_figure(zones, far)
    types_zones_figure()
    medians = {}
    for path in sorted((HERE / "dati").glob("era5_caso_*.npz")):
        with np.load(path) as data:
            number = int(path.stem[-2:])
            arrays = {key: data[key] for key in data.files}
            case_figure(number, arrays, coastline_of(args.coastline, arrays))
            medians[json.loads(str(arrays["case"]))["type"]] = {"record": json.loads(str(arrays["case"])), **case_medians(arrays)}
    (HERE / "dati" / "era5_casi_zone.json").write_text(json.dumps(medians, indent=1), encoding="utf-8")
    print(json.dumps({name: {zone: value[zone] for zone in ("core", "edge", "far")} for name, value in medians.items()}, indent=1))


if __name__ == "__main__":
    main()

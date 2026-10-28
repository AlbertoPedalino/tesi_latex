"""Generate the figures of Sections 6.2-6.4: how the scores of the generator (G) and of the discriminator (D) go
together, and the two outputs of D that make its score.

Run from the thesis folder. First, once, gather the small data from the files downloaded from the server
(outputs/analisi_good2/ of the code repository: the folders correlazioni_g_d and correlazioni_g_dterms written by
scripts/check_era5_score_correlation.py, zone_g_dterms.csv by scripts/check_era5_event_zones.py, and
mappe_istanti.npz with G, D and D(real) at three timestamps of the scored test run):
    python figures/genera_figure_g_d.py --prepare <folder with the downloaded files>
which writes figures/dati/era5_g_d.npz. Then:
    python figures/genera_figure_g_d.py [--coastline <ne_50m_coastline.geojson>]
which writes figures/era5_g_d_correlazione_spaziale.pdf, era5_g_d_mappe.pdf, era5_g_d_correlazione_temporale.pdf,
era5_d_termini_correlazione_temporale.pdf, era5_d_termini_zone.pdf and era5_d_termini_mappa.pdf. The coastline,
from Natural Earth (public domain), is optional and not kept in the repository.

Run: good-sweep-2, test years 2005-2025. D's score is D(real) - D(generated), D's outputs on the observed and on the
reconstructed patch (1 = judged generated); D(generated) is read as D(real) minus D's score.
"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
DATA = HERE / "dati" / "era5_g_d.npz"
WIDTH_CM = 14.0
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e2e1dc"
COAST = "#3d3c39"
ROLLING_DAYS = 30
TIMES = ("2014-08-08 03:00:00", "2024-08-30 12:00:00")  # a typical timestamp; the highest mean G of the test
LIMIT = 0.5  # temporal correlations drawn on [-LIMIT, LIMIT]

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


def prepare(folder):
    """Gather the data of the figures from the downloaded files into figures/dati/era5_g_d.npz."""
    folder = pathlib.Path(folder)
    spatial = pd.read_csv(folder / "correlazioni_g_d" / "spatial_correlation.csv", index_col=0, parse_dates=True)
    daily = spatial[["G - D spearman", "G - D pearson"]].resample("1D").mean()
    with np.load(folder / "correlazioni_g_d" / "temporal_correlation.npz") as main, \
            np.load(folder / "correlazioni_g_dterms" / "temporal_correlation.npz") as terms, \
            np.load(folder / "mappe_istanti.npz") as maps:
        grid = json.loads(str(maps["grid"]))
        times = [str(value) for value in maps["times"]]
        chosen = [times.index(value) for value in TIMES]
        zones = pd.read_csv(folder / "zone_g_dterms.csv")
        everything = zones[(zones["type"] == "all") & (zones["size"] == "all")].iloc[0]
        far = zones[zones["type"].str.startswith("far")].iloc[0]
        names = ("G", "Dreale", "Dricostruito", "D")
        zone_table = np.array([[everything[f"{name} core percentile"], everything[f"{name} edge percentile"],
                                far[f"{name} core percentile"]] for name in names])
        np.savez_compressed(
            DATA, days=daily.index.values.astype("datetime64[D]"), spearman=daily["G - D spearman"].to_numpy(),
            pearson=daily["G - D pearson"].to_numpy(), temporal_g_d=main["G_-_D_spearman"],
            temporal_g_real=terms["G_-_Dreale_spearman"], temporal_g_generated=terms["G_-_Dricostruito_spearman"],
            times=np.array(TIMES), G=maps["G"][chosen], D=maps["D"][chosen], Dreale=maps["Dreale"][chosen],
            rows=np.asarray(grid["rows"]), cols=np.asarray(grid["cols"]), latitudes=np.asarray(grid["latitudes"]),
            longitudes=np.asarray(grid["longitudes"]), zones=zone_table, zone_scores=np.array(names),
            events=int(everything["events"]))
    print(f"written {DATA}")


def image(data, values):
    picture = np.full((len(data["latitudes"]), len(data["longitudes"])), np.nan)
    picture[data["rows"], data["cols"]] = values
    return picture


def draw_map(axis, data, values, coastline, **style):
    latitudes, longitudes = data["latitudes"], data["longitudes"]
    mesh = axis.pcolormesh(longitudes, latitudes, image(data, values), shading="nearest", rasterized=True, **style)
    for line in coastline:
        # A thin white halo keeps the coast visible on the dark and on the light colours of the maps.
        axis.plot(line[:, 0], line[:, 1], color=COAST, linewidth=0.35,
                  path_effects=[effects.withStroke(linewidth=1.1, foreground="white")])
    axis.set_xlim(longitudes.min() - 0.25, longitudes.max() + 0.25)
    axis.set_ylim(latitudes.min() - 0.25, latitudes.max() + 0.25)
    axis.set_aspect(1 / np.cos(np.radians(latitudes.mean())))
    axis.tick_params(labelsize=7)
    return mesh


def coastline_of(path, data):
    if path is None:
        return []
    lines = []
    west, east = data["longitudes"].min() - 2, data["longitudes"].max() + 2
    south, north = data["latitudes"].min() - 2, data["latitudes"].max() + 2
    for feature in json.load(open(path, encoding="utf-8"))["features"]:
        geometry = feature["geometry"]
        parts = geometry["coordinates"] if geometry["type"] == "MultiLineString" else [geometry["coordinates"]]
        for part in parts:
            points = np.asarray(part, dtype=float)
            if ((points[:, 0] > west) & (points[:, 0] < east) & (points[:, 1] > south) & (points[:, 1] < north)).any():
                lines.append(points)
    return lines


def size(height_ratio):
    width = WIDTH_CM / 2.54
    return width, width * height_ratio


def spatial_figure(data):
    figure, axis = plt.subplots(figsize=size(0.42), constrained_layout=True)
    days = pd.DatetimeIndex(data["days"])
    for key, colour, label in (("spearman", BLUE, "Spearman"), ("pearson", ORANGE, "Pearson")):
        series = pd.Series(data[key], index=days)
        axis.plot(days, series.to_numpy(), color=colour, linewidth=0.4, alpha=0.25)
        axis.plot(days, series.rolling(ROLLING_DAYS, min_periods=ROLLING_DAYS // 2).mean().to_numpy(), color=colour,
                  linewidth=1.4, label=label)
    axis.axhline(0, color=MUTED, linewidth=0.7)
    axis.set_ylim(-1, 0.5)
    axis.set_ylabel("correlazione spaziale G–D")
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", ncols=2, frameon=False, fontsize=8)
    figure.savefig(HERE / "era5_g_d_correlazione_spaziale.pdf")
    plt.close(figure)


def maps_figure(data, coastline):
    figure, axes = plt.subplots(2, 2, figsize=size(0.78), constrained_layout=True)
    for row, when in enumerate(data["times"]):
        for column, (key, label) in enumerate((("G", "score di G"), ("D", "score di D"))):
            values = data[key][row]
            low, high = np.nanpercentile(values, (1, 99))
            mesh = draw_map(axes[row, column], data, values, coastline, cmap="Blues", vmin=low, vmax=high)
            figure.colorbar(mesh, ax=axes[row, column], shrink=0.85, pad=0.02).ax.tick_params(labelsize=7)
            axes[row, column].set_title(f"{label}, {pd.Timestamp(str(when)):%d/%m/%Y %H} UTC", fontsize=8)
    figure.savefig(HERE / "era5_g_d_mappe.pdf", dpi=300)
    plt.close(figure)


def temporal_figure(data, coastline):
    figure, axis = plt.subplots(figsize=size(0.62), constrained_layout=True)
    mesh = draw_map(axis, data, data["temporal_g_d"], coastline, cmap="RdBu_r", vmin=-LIMIT, vmax=LIMIT)
    figure.colorbar(mesh, ax=axis, shrink=0.85, pad=0.02, label="correlazione temporale G–D (Spearman)")
    figure.savefig(HERE / "era5_g_d_correlazione_temporale.pdf", dpi=300)
    plt.close(figure)


def terms_temporal_figure(data, coastline):
    figure, axes = plt.subplots(1, 2, figsize=size(0.4), constrained_layout=True)
    for axis, key, title in zip(axes, ("temporal_g_real", "temporal_g_generated"),
                                ("G – D(reale)", "G – D(generato)")):
        mesh = draw_map(axis, data, data[key], coastline, cmap="RdBu_r", vmin=-LIMIT, vmax=LIMIT)
        axis.set_title(title, fontsize=8)
    figure.colorbar(mesh, ax=list(axes), shrink=0.85, pad=0.02, label="Spearman")
    figure.savefig(HERE / "era5_d_termini_correlazione_temporale.pdf", dpi=300)
    plt.close(figure)


def zones_figure(data):
    figure, axis = plt.subplots(figsize=size(0.42), constrained_layout=True)
    labels = {"G": "G", "Dreale": "D(reale)", "Dricostruito": "D(generato)", "D": "score di D"}
    zones = ("nucleo", "contorno", "lontano")
    shades = ("#184f95", "#5598e7", "#b7d3f6")  # one hue, darker at the core
    positions = np.arange(len(data["zone_scores"]))
    width = 0.26
    for index, (zone, shade) in enumerate(zip(zones, shades)):
        values = data["zones"][:, index]
        bars = axis.bar(positions + (index - 1) * width, values, width - 0.03, color=shade, label=zone)
        axis.bar_label(bars, labels=[f"{value:.0f}" for value in values], fontsize=7, color=TEXT, padding=1)
    axis.axhline(50, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
    axis.set_xticks(positions, [labels[str(name)] for name in data["zone_scores"]])
    axis.set_ylim(0, 108)
    axis.set_ylabel("percentile mediano dello score")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper right", ncols=3, frameon=False, fontsize=8)
    figure.savefig(HERE / "era5_d_termini_zone.pdf")
    plt.close(figure)


def terms_map_figure(data, coastline):
    row = 1  # the highest mean G of the test
    fields = (("G", "score di G", data["G"][row]), ("Dreale", "D(reale)", data["Dreale"][row]),
              ("Dricostruito", "D(generato)", data["Dreale"][row] - data["D"][row]),
              ("D", "score di D", data["D"][row]))
    figure, axes = plt.subplots(2, 2, figsize=size(0.78), constrained_layout=True)
    for axis, (_, label, values) in zip(axes.ravel(), fields):
        low, high = np.nanpercentile(values, (1, 99))
        mesh = draw_map(axis, data, values, coastline, cmap="Blues", vmin=low, vmax=high)
        figure.colorbar(mesh, ax=axis, shrink=0.85, pad=0.02).ax.tick_params(labelsize=7)
        axis.set_title(label, fontsize=8)
    figure.savefig(HERE / "era5_d_termini_mappa.pdf", dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prepare", type=pathlib.Path, help="Folder with the files downloaded from the server")
    parser.add_argument("--coastline", type=pathlib.Path, help="ne_50m_coastline.geojson of Natural Earth")
    args = parser.parse_args()
    if args.prepare is not None:
        prepare(args.prepare)
        return
    with np.load(DATA) as stored:
        data = {key: stored[key] for key in stored.files}
    coastline = coastline_of(args.coastline, data)
    spatial_figure(data)
    maps_figure(data, coastline)
    temporal_figure(data, coastline)
    terms_temporal_figure(data, coastline)
    zones_figure(data)
    terms_map_figure(data, coastline)
    print(f"figures written in {HERE} ({data['events']} events in the zones)")


if __name__ == "__main__":
    main()

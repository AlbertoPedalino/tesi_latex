"""Generate the figures of Sections 6.2-6.4: how the scores of the generator (G) and of the discriminator (D) go
together, and the two outputs of D that make its score.

Run from the thesis folder. First, once, gather the small data from the files downloaded from the server
(the output folder of scripts/run_era5_score_behaviour.sh in the code repository: the folders correlazioni_g_d
and correlazioni_g_dterms written by scripts/check_era5_score_correlation.py, zone_g_dterms_5pc.csv by
scripts/check_era5_event_zones.py, and mappe_istanti.npz with G, D and D(real) at two timestamps of the scored run,
first the typical one, then the one with the highest mean G):
    python figures/genera_figure_g_d.py --prepare <folder with the downloaded files>
which writes figures/dati/era5_g_d.npz. Then:
    python figures/genera_figure_g_d.py [--coastline <ne_50m_coastline.geojson>]
which writes figures/era5_g_d_correlazione_spaziale.pdf, era5_g_d_mappe.pdf, era5_g_d_correlazione_temporale.pdf,
era5_d_termini_correlazione_temporale.pdf, era5_d_termini_zone.pdf, era5_zone_schema.pdf (a sketch of the
zones) and era5_d_termini_mappa.pdf. The coastline,
from Natural Earth (public domain), is optional and not kept in the repository.

Run: good-sweep-2, validation years 2000-2004. D's score is D(real) - D(generated), D's outputs on the observed and on the
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
        chosen = [0, 1]  # the typical timestamp, then the one with the highest mean G
        times = np.array([str(value) for value in maps["times"]])[chosen]
        zones = pd.read_csv(folder / "zone_g_dterms_5pc.csv")
        everything = zones[(zones["type"] == "all") & (zones["size"] == "all")].iloc[0]
        far = zones[zones["type"].str.startswith("far")].iloc[0]
        names = ("G", "Dreale", "Dricostruito", "D")
        written = {"G": "G", "Dreale": "D_reale", "Dricostruito": "D_generato", "D": "D"}  # names in the files
        zone_table = np.array([[everything[f"{written[name]} core percentile"],
                                everything[f"{written[name]} edge percentile"],
                                far[f"{written[name]} core percentile"]] for name in names])
        np.savez_compressed(
            DATA, days=daily.index.values.astype("datetime64[D]"), spearman=daily["G - D spearman"].to_numpy(),
            pearson=daily["G - D pearson"].to_numpy(), temporal_g_d=main["G_-_D_spearman"],
            temporal_g_real=terms["G_-_D_reale_spearman"], temporal_g_generated=terms["G_-_D_generato_spearman"],
            times=times, G=maps["G"][chosen], D=maps["D"][chosen], Dreale=maps["Dreale"][chosen],
            rows=np.asarray(grid["rows"]), cols=np.asarray(grid["cols"]), latitudes=np.asarray(grid["latitudes"]),
            longitudes=np.asarray(grid["longitudes"]), zones=zone_table, zone_scores=np.array(names),
            events=int(everything["events"]))
    print(f"written {DATA}")


def image(data, values):
    picture = np.full((len(data["latitudes"]), len(data["longitudes"])), np.nan)
    picture[data["rows"], data["cols"]] = values
    return picture


REAL, GENERATED = r"$D_{\mathrm{reale}}$", r"$D_{\mathrm{generato}}$"
TICKS = 8  # tick labels, in points at the printed size
RING = 2   # the frame of the temporal correlation map, in cells from the border of the domain


def draw_map(axis, data, values, coastline, *, left=True, bottom=True, **style):
    latitudes, longitudes = data["latitudes"], data["longitudes"]
    mesh = axis.pcolormesh(longitudes, latitudes, image(data, values), shading="nearest", rasterized=True, **style)
    for line in coastline:
        # A thin white halo keeps the coast visible on the dark and on the light colours of the maps.
        axis.plot(line[:, 0], line[:, 1], color=COAST, linewidth=0.35,
                  path_effects=[effects.withStroke(linewidth=1.1, foreground="white")])
    axis.set_xlim(longitudes.min() - 0.25, longitudes.max() + 0.25)
    axis.set_ylim(latitudes.min() - 0.25, latitudes.max() + 0.25)
    axis.set_aspect(1 / np.cos(np.radians(latitudes.mean())))
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:g}°")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:g}°")
    axis.tick_params(labelsize=TICKS)
    if bottom:
        axis.set_xlabel("longitudine (E)", fontsize=TICKS)
    if left:
        axis.set_ylabel("latitudine (N)", fontsize=TICKS)
    return mesh


def colour_bar(figure, mesh, axes, label):
    bar = figure.colorbar(mesh, ax=axes, shrink=0.85, pad=0.02)
    bar.set_label(label, fontsize=TICKS)
    bar.ax.tick_params(labelsize=TICKS)
    return bar


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
    axis.set_xlim(days.min(), days.max())
    axis.set_ylim(-1, 0.5)
    axis.set_ylabel("correlazione spaziale\nfra G e D")
    axis.tick_params(labelsize=TICKS)
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", ncols=2, frameon=False, fontsize=8)
    figure.savefig(HERE / "era5_g_d_correlazione_spaziale.pdf")
    plt.close(figure)


def maps_figure(data, coastline):
    """G and D at the typical timestamp (the median spatial correlation); the other one is in era5_d_termini_mappa."""
    row = 0
    # Colour bars below the maps, so that the two maps take the whole width.
    figure, axes = plt.subplots(1, 2, figsize=size(0.56), constrained_layout=True)
    for column, (key, label) in enumerate((("G", "G (alto = più insolito)"), ("D", "D (alto = più insolito)"))):
        values = data[key][row]
        low, high = np.nanpercentile(values, (1, 99))
        mesh = draw_map(axes[column], data, values, coastline, left=column == 0, cmap="viridis", vmin=low, vmax=high)
        bar = figure.colorbar(mesh, ax=axes[column], orientation="horizontal", shrink=0.9, pad=0.02, aspect=30)
        bar.set_label(label, fontsize=TICKS)
        bar.ax.tick_params(labelsize=TICKS)
        bar.ax.locator_params(nbins=5)
    figure.savefig(HERE / "era5_g_d_mappe.pdf", dpi=300)
    plt.close(figure)


def temporal_figure(data, coastline):
    figure, axis = plt.subplots(figsize=size(0.62), constrained_layout=True)
    mesh = draw_map(axis, data, data["temporal_g_d"], coastline, cmap="viridis", vmin=-0.4, vmax=0.0)
    colour_bar(figure, mesh, axis, "correlazione temporale fra G e D (Spearman)")
    # The frame two cells from the border, where the correlation is weaker (an effect of the model).
    latitudes, longitudes = data["latitudes"], data["longitudes"]
    step = abs(float(latitudes[1] - latitudes[0]))
    south, north = latitudes.min() + RING * step, latitudes.max() - RING * step
    west, east = longitudes.min() + RING * step, longitudes.max() - RING * step
    axis.plot([west, east, east, west, west], [south, south, north, north, south], color="white", linewidth=0.8,
              linestyle=(0, (3, 2)))
    axis.annotate("cornice a due celle dal bordo", xy=(east, north - 4), xytext=(east - 4, north - 4), ha="right",
                  va="center", fontsize=7, color="white",
                  arrowprops={"arrowstyle": "->", "color": "white", "linewidth": 0.7})
    figure.savefig(HERE / "era5_g_d_correlazione_temporale.pdf", dpi=300)
    plt.close(figure)


def terms_temporal_figure(data, coastline):
    figure, axes = plt.subplots(1, 2, figsize=size(0.42), constrained_layout=True)
    for column, (axis, key, title) in enumerate(zip(axes, ("temporal_g_real", "temporal_g_generated"),
                                                    (f"Spearman fra G e {REAL}", f"Spearman fra G e {GENERATED}"))):
        mesh = draw_map(axis, data, data[key], coastline, left=column == 0, cmap="RdBu_r", vmin=-LIMIT, vmax=LIMIT)
        axis.set_title(title, fontsize=9)
    colour_bar(figure, mesh, list(axes), "correlazione temporale (Spearman)")
    figure.savefig(HERE / "era5_d_termini_correlazione_temporale.pdf", dpi=300)
    plt.close(figure)


def zones_figure(data):
    figure, axis = plt.subplots(figsize=size(0.42), constrained_layout=True)
    labels = {"G": "G", "Dreale": REAL, "Dricostruito": GENERATED, "D": "D"}
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
    axis.tick_params(labelsize=TICKS)
    axis.set_ylim(0, 108)
    axis.set_ylabel("percentile mediano")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper right", ncols=3, frameon=False, fontsize=8)
    figure.savefig(HERE / "era5_d_termini_zone.pdf")
    plt.close(figure)


def zones_scheme_figure():
    """A sketch of the three zones around an event on the grid: core, edge (the 8 cells around), far (> 3 cells)."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from scipy import ndimage

    side = 15
    core = np.zeros((side, side), dtype=bool)
    core[6:9, 5:8] = True
    core[7, 8:10] = True
    core[5, 6] = True
    square = np.ones((3, 3), dtype=bool)
    edge = ndimage.binary_dilation(core, square) & ~core
    near = ndimage.binary_dilation(core, square, iterations=3)
    zone = np.where(core, 0, np.where(edge, 1, np.where(near, 2, 3)))
    colours = ("#184f95", "#5598e7", "#ffffff", "#e7eef8")
    figure, axis = plt.subplots(figsize=size(0.36), constrained_layout=True)
    axis.imshow(zone, cmap=ListedColormap(colours), vmin=-0.5, vmax=3.5)
    axis.set_xticks(np.arange(-0.5, side), minor=True)
    axis.set_yticks(np.arange(-0.5, side), minor=True)
    axis.grid(which="minor", color=GRID, linewidth=0.6)
    axis.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for spine in axis.spines.values():
        spine.set_edgecolor(MUTED)
    legend = (("nucleo: le celle dell'evento", colours[0]), ("contorno: le 8 celle attorno al nucleo", colours[1]),
              ("celle intermedie, escluse", colours[2]), ("lontano: oltre 3 celle da ogni nucleo", colours[3]))
    axis.legend(handles=[Patch(facecolor=colour, edgecolor=MUTED, linewidth=0.5, label=label) for label, colour in legend],
                loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    figure.savefig(HERE / "era5_zone_schema.pdf")
    plt.close(figure)


def terms_map_figure(data, coastline):
    row = 1  # the highest mean G of the period
    fields = (("G", "G (alto = più insolito)", data["G"][row]),
              ("Dreale", f"{REAL} (1 = giudicato generato)", data["Dreale"][row]),
              ("Dricostruito", f"{GENERATED} (1 = giudicato generato)", data["Dreale"][row] - data["D"][row]),
              ("D", f"D = {REAL} $-$ {GENERATED}", data["D"][row]))
    figure, axes = plt.subplots(2, 2, figsize=size(0.80), constrained_layout=True)
    for index, (axis, (_, label, values)) in enumerate(zip(axes.ravel(), fields)):
        low, high = np.nanpercentile(values, (1, 99))
        mesh = draw_map(axis, data, values, coastline, left=index % 2 == 0, bottom=index >= 2, cmap="viridis",
                        vmin=low, vmax=high)
        colour_bar(figure, mesh, axis, label)
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
    zones_scheme_figure()
    terms_map_figure(data, coastline)
    print(f"figures written in {HERE} ({data['events']} events in the zones)")


if __name__ == "__main__":
    main()

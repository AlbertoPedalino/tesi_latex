"""Generate figures/localita_pvgis.pdf: map of the PVGIS locations in Piedmont with the ERA5 grid.

Descriptive figure of the dataset, not a result of the thesis. Run from the thesis folder:
    python figures/genera_mappa_pvgis.py <path to piedmont_pvgis_2019.nc>
or with a CSV that has the columns latitude and longitude, one row per location, for example
    python figures/genera_mappa_pvgis.py <physiq-pv>/.analysis/stgan_results_seed20/summary.csv
The locations are the same in every year: only their coordinates are used.

The ERA5 grid has a step of 0.5 degrees, with grid points at multiples of 0.5 degrees (Sez. 3.1); each ERA5 cell is the
square of side 0.5 degrees centred on its grid point. The script draws the grid points, the cell boundaries and one
cell in evidence, and prints how many ERA5 cells hold PVGIS locations and how many locations each holds: the numbers
quoted in Sez. 3.6.
"""
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

WIDTH_CM = 9.0
DOT_COLOUR = "#2a78d6"
ERA5_COLOUR = "#d9480f"
AXIS_COLOUR = "#52514e"
STEP = 0.5
EVIDENCE = (45.0, 8.0)  # grid point (latitude, longitude) of the cell shown in evidence
VIEW = (6.4, 9.45, 43.85, 46.7)  # west, east, south, north

source = pathlib.Path(sys.argv[1])
if source.suffix.lower() == ".csv":
    import pandas as pd

    table = pd.read_csv(source)
    latitude, longitude = table["latitude"].to_numpy(float), table["longitude"].to_numpy(float)
else:
    import xarray as xr

    dataset = xr.open_dataset(source)
    latitude, longitude = dataset["lat"].values, dataset["lon"].values

# ERA5 cell of each location: the nearest grid point
cell_lat = np.round(latitude / STEP) * STEP
cell_lon = np.round(longitude / STEP) * STEP
cells, per_cell = np.unique(np.stack([cell_lat, cell_lon], axis=1), axis=0, return_counts=True)
in_evidence = int(((cell_lat == EVIDENCE[0]) & (cell_lon == EVIDENCE[1])).sum())
print(f"{len(latitude)} locations in {len(cells)} ERA5 cells; locations per cell: median {np.median(per_cell):g}, "
      f"min {per_cell.min()}, max {per_cell.max()}; cell in evidence {EVIDENCE}: {in_evidence}")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "font.size": 9,
    "pdf.fonttype": 42,
})
west, east, south, north = VIEW
# One degree of longitude is shorter than one of latitude: keep distances undistorted.
aspect = 1.0 / np.cos(np.radians(latitude.mean()))
width = WIDTH_CM / 2.54
figure, axes = plt.subplots(figsize=(width, width * (north - south) * aspect / (east - west) + 0.8))

boundary = {"color": ERA5_COLOUR, "linewidth": 0.5, "alpha": 0.6, "linestyle": (0, (3, 2))}
for value in np.arange(np.floor(west) + STEP / 2, east, STEP):
    axes.axvline(value, **boundary)
for value in np.arange(np.floor(south) + STEP / 2, north, STEP):
    axes.axhline(value, **boundary)
axes.add_patch(Rectangle((EVIDENCE[1] - STEP / 2, EVIDENCE[0] - STEP / 2), STEP, STEP, facecolor=ERA5_COLOUR,
                         alpha=0.15, edgecolor="none", zorder=0))
axes.scatter(longitude, latitude, s=2.6, color=DOT_COLOUR, linewidths=0)
grid_lon, grid_lat = np.meshgrid(np.arange(np.ceil(west / STEP) * STEP, east, STEP),
                                 np.arange(np.ceil(south / STEP) * STEP, north, STEP))
axes.scatter(grid_lon.ravel(), grid_lat.ravel(), s=14, color=ERA5_COLOUR, linewidths=0, zorder=3)

handles = [
    Line2D([], [], marker="o", linestyle="none", markersize=2.5, color=DOT_COLOUR),
    Line2D([], [], marker="o", linestyle="none", markersize=5, color=ERA5_COLOUR),
    Line2D([], [], color=ERA5_COLOUR, linewidth=0.8, alpha=0.6, linestyle=(0, (3, 2))),
    Rectangle((0, 0), 1, 1, facecolor=ERA5_COLOUR, alpha=0.15, edgecolor="none"),
]
labels = [
    "località PVGIS (reticolo di 5 km)",
    "punti della griglia ERA5 (0.5°, circa 55 × 40 km)",
    "confini delle celle ERA5",
    f"una cella ERA5: contiene {in_evidence} località PVGIS",
]
axes.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=8)

axes.set_xlim(west, east)
axes.set_ylim(south, north)
axes.set_aspect(aspect)
axes.set_xlabel("longitudine (E)", color=AXIS_COLOUR)
axes.set_ylabel("latitudine (N)", color=AXIS_COLOUR)
axes.set_xticks(np.arange(6.5, 9.51, 0.5))
axes.set_yticks(np.arange(44.0, 46.51, 0.5))
axes.xaxis.set_major_formatter(lambda value, _: f"{value:.1f}°")
axes.yaxis.set_major_formatter(lambda value, _: f"{value:.1f}°")
axes.tick_params(colors=AXIS_COLOUR, width=0.5, length=3)
for side in ("top", "right"):
    axes.spines[side].set_visible(False)
for side in ("bottom", "left"):
    axes.spines[side].set_color(AXIS_COLOUR)
    axes.spines[side].set_linewidth(0.5)

output = pathlib.Path(__file__).with_name("localita_pvgis.pdf")
figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
if len(sys.argv) > 2:
    figure.savefig(sys.argv[2], bbox_inches="tight", pad_inches=0.02, dpi=200)
print(f"-> {output}")

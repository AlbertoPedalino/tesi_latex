"""Generate figures/localita_pvgis.pdf: map of the PVGIS locations in Piedmont.

Descriptive figure of the dataset, not a result of the thesis. Run from the thesis folder:
    python figures/genera_mappa_pvgis.py <path to piedmont_pvgis_2019.nc>
"""
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

WIDTH_CM = 9.0
DOT_COLOUR = "#2a78d6"
AXIS_COLOUR = "#52514e"

dataset = xr.open_dataset(sys.argv[1])
latitude, longitude = dataset["lat"].values, dataset["lon"].values

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "font.size": 9,
    "pdf.fonttype": 42,
})
# One degree of longitude is shorter than one of latitude: keep distances undistorted.
aspect = 1.0 / np.cos(np.radians(latitude.mean()))
span_x = longitude.max() - longitude.min()
span_y = latitude.max() - latitude.min()
width = WIDTH_CM / 2.54
figure, axes = plt.subplots(figsize=(width, width * span_y * aspect / span_x + 0.45))
axes.scatter(longitude, latitude, s=2.6, color=DOT_COLOUR, linewidths=0)
axes.set_aspect(aspect)
axes.set_xlabel("Longitudine est (gradi)", color=AXIS_COLOUR)
axes.set_ylabel("Latitudine nord (gradi)", color=AXIS_COLOUR)
axes.set_xticks(np.arange(7.0, 9.01, 0.5))
axes.set_yticks(np.arange(44.5, 46.01, 0.5))
axes.margins(0.04)
axes.tick_params(colors=AXIS_COLOUR, width=0.5, length=3)
for side in ("top", "right"):
    axes.spines[side].set_visible(False)
for side in ("bottom", "left"):
    axes.spines[side].set_color(AXIS_COLOUR)
    axes.spines[side].set_linewidth(0.5)

output = pathlib.Path(__file__).with_name("localita_pvgis.pdf")
figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
print(f"{len(latitude)} locations -> {output}")

"""Generate figures/dominio_era5.pdf: the ERA5 domain with the PVGIS area inside it.

Descriptive figure of the dataset, not a result of the thesis. Run from the thesis folder:
    python figures/genera_mappa_era5.py <path to ne_50m_coastline.geojson>

The coastline comes from Natural Earth (public domain), 1:50m, GeoJSON as published at
https://github.com/nvkelso/natural-earth-vector (geojson/ne_50m_coastline.geojson).
It is not kept in the repository.

Domain bounds read on 11 October 2026 from the metadata of an ERA5 run
(outputs/era5_gat_mc_paper_recent16_run1/metadata.json): 81 latitudes from 60 to 20
degrees north and 131 longitudes from -15 to 50 degrees east, step 0.5 degrees.
PVGIS bounds as in the data chapter.
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

WIDTH_CM = 12.0
LAT_SOUTH, LAT_NORTH = 20.0, 60.0
LON_WEST, LON_EAST = -15.0, 50.0
PVGIS_SOUTH, PVGIS_NORTH = 44.08, 46.47
PVGIS_WEST, PVGIS_EAST = 6.66, 9.20
COAST_COLOUR = "#7b7a75"
FRAME_COLOUR = "#52514e"
PVGIS_COLOUR = "#2a78d6"

coastline = json.load(open(sys.argv[1], encoding="utf-8"))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "font.size": 9,
    "pdf.fonttype": 42,
})
# One degree of longitude is shorter than one of latitude: keep distances
# undistorted at the central latitude of the domain.
aspect = 1.0 / np.cos(np.radians(0.5 * (LAT_SOUTH + LAT_NORTH)))
width = WIDTH_CM / 2.54
height = width * (LAT_NORTH - LAT_SOUTH) * aspect / (LON_EAST - LON_WEST)
figure, axes = plt.subplots(figsize=(width, height + 0.45))

segments = 0
for feature in coastline["features"]:
    geometry = feature["geometry"]
    lines = geometry["coordinates"] if geometry["type"] == "MultiLineString" else [geometry["coordinates"]]
    for line in lines:
        points = np.asarray(line, dtype=float)
        inside = ((points[:, 0] > LON_WEST - 2) & (points[:, 0] < LON_EAST + 2)
                  & (points[:, 1] > LAT_SOUTH - 2) & (points[:, 1] < LAT_NORTH + 2))
        if inside.any():
            axes.plot(points[:, 0], points[:, 1], color=COAST_COLOUR, linewidth=0.45)
            segments += 1

axes.add_patch(Rectangle((PVGIS_WEST, PVGIS_SOUTH), PVGIS_EAST - PVGIS_WEST, PVGIS_NORTH - PVGIS_SOUTH,
                         facecolor=PVGIS_COLOUR, edgecolor=PVGIS_COLOUR, alpha=0.9, linewidth=0.6))
axes.annotate("località PVGIS", xy=(PVGIS_EAST, PVGIS_NORTH), xytext=(12.0, 49.2),
              color=PVGIS_COLOUR, fontsize=8,
              arrowprops={"arrowstyle": "-", "color": PVGIS_COLOUR, "linewidth": 0.5})

axes.set_xlim(LON_WEST, LON_EAST)
axes.set_ylim(LAT_SOUTH, LAT_NORTH)
axes.set_aspect(aspect)
axes.set_xlabel("Longitudine est (gradi)", color=FRAME_COLOUR)
axes.set_ylabel("Latitudine nord (gradi)", color=FRAME_COLOUR)
axes.set_xticks(np.arange(-10, 50.1, 10))
axes.set_yticks(np.arange(20, 60.1, 10))
axes.tick_params(colors=FRAME_COLOUR, width=0.5, length=3)
for side in ("top", "right", "bottom", "left"):
    axes.spines[side].set_color(FRAME_COLOUR)
    axes.spines[side].set_linewidth(0.6)

output = pathlib.Path(__file__).with_name("dominio_era5.pdf")
figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
print(f"{segments} coastline segments -> {output}")

"""Generate the three ERA5 data figures of chapter 3 from figures/dati/era5_dati.json.

Descriptive figures of the training years, with no model. Run from the thesis folder:
    python figures/genera_figure_dati.py

The JSON is written on the server by scripts/summarize_era5_data_figures.py of the code repository
(--prepared-dir data/era5_prepared --train-end-year 1999). Writes figures/era5_cicli.pdf,
figures/era5_variazione_24h.pdf and figures/era5_correlazione_distanza.pdf.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pannelli import save_panels

HERE = pathlib.Path(__file__).parent
WIDTH_CM = 13.2
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e2e1dc"
CELL_COLORS = (BLUE, ORANGE, AQUA)
# The variables have colours of their own, never the blue, orange and green of the methods (G_n, distance from normal,
# change in three hours): the same in Figures 3.5, 6.13 and in the cases. Line colour, then a darker one for text.
VARIABLE_COLOURS = {"total_precipitation": ("#7b4fb3", "#6a3fa3"), "total_cloud_cover": ("#55534e", "#3f3d39"),
                    "vertical_velocity": ("#c2408f", "#a8307a"), "2m_temperature": ("#c08a00", "#946a00"),
                    "mean_sea_level_pressure": ("#13867f", "#0f6f69")}
NAMES = {
    "2m_temperature": "Temperatura a due metri",
    "10m_u_component_of_wind": "Vento zonale a dieci metri",
    "10m_v_component_of_wind": "Vento meridionale a dieci metri",
    "mean_sea_level_pressure": "Pressione al livello del mare",
    "total_precipitation": "Precipitazione",
    "total_cloud_cover": "Copertura nuvolosa",
    "surface_solar_radiation_downwards": "Radiazione solare",
    "volumetric_soil_water_layer_1": "Umidità del suolo",
    "friction_velocity": "Velocità d'attrito",
    "boundary_layer_height": "Strato limite",
    "u_component_of_wind": "Vento zonale a 850 hPa",
    "v_component_of_wind": "Vento meridionale a 850 hPa",
    "vertical_velocity": "Velocità verticale",
    "temperature": "Temperatura a 850 hPa",
    "specific_humidity": "Umidità specifica",
}
MONTHS = ["G", "F", "M", "A", "M", "G", "L", "A", "S", "O", "N", "D"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.5,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def save(figure, name):
    figure.savefig(HERE / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print("written", name)


def tidy(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)


def cycles_figure(data):
    """Monthly mean with one standard deviation, and mean by time of day in January and July."""
    part = data["cycles"]
    units = {"2m_temperature": ("Temperatura (°C)", lambda v: np.asarray(v) - 273.15, (-np.inf, np.inf)),
             "total_cloud_cover": ("Copertura\nnuvolosa (0–1)", np.asarray, (0.0, 1.0))}
    figure, axes = plt.subplots(2, 2, figsize=(14.0 / 2.54, 8.4 / 2.54), sharex="col",
                                gridspec_kw={"width_ratios": [1, 1]})
    for row, feature in enumerate(part["features"]):
        label, convert, (lowest, highest) = units[feature]
        for color, (cell, values) in zip(CELL_COLORS, part["cells"].items()):
            series = values[feature]
            mean = convert(series["monthly_mean"])
            std = np.asarray(series["monthly_std"])
            months = np.arange(1, 13)
            axes[row, 0].plot(months, mean, color=color, linewidth=1.5, label=cell)
            # One standard deviation of the 3-hourly values of the month as a light band: the dashed line is only
            # January, in the panel of the daily cycle.
            axes[row, 0].fill_between(months, np.maximum(mean - std, lowest), np.minimum(mean + std, highest),
                                      color=color, alpha=0.10, linewidth=0)
            for month, style in (("1", "--"), ("7", "-")):
                axes[row, 1].plot(part["hours"], convert(series["hourly_mean"][month]), style, color=color,
                                  linewidth=1.2)
        axes[row, 0].set_ylabel(label)
        axes[row, 1].set_ylabel(label)  # the right column stands alone as its own subfigure
        for axis in axes[row]:
            tidy(axis)
    axes[1, 0].set_xticks(np.arange(1, 13), MONTHS)
    axes[1, 0].set_xlabel("Mese")
    axes[1, 1].set_xticks([0, 6, 12, 18])
    axes[1, 1].set_xlabel("Ora UTC")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    cells = axes[1, 0].legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.32),
                              fontsize=7.5, columnspacing=1.0, handlelength=1.6)
    months = axes[1, 1].legend([plt.Line2D([], [], color=MUTED, linestyle="--"),
                                plt.Line2D([], [], color=MUTED, linestyle="-")], ["gennaio", "luglio"],
                               loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.32), fontsize=7.5,
                               columnspacing=1.0, handlelength=1.6)
    figure.tight_layout(rect=(0, 0.04, 1, 1), w_pad=2.0)
    figure.align_ylabels(axes[:, 0])
    save_panels(figure, [[axes[0, 0], axes[1, 0], cells], [axes[0, 1], axes[1, 1], months]], HERE, "era5_cicli",
                pad=0.02)
    plt.close(figure)
    print("written era5_cicli_a.pdf, era5_cicli_b.pdf")


def change_figure(data, weights):
    """Share of the mean squared change over 24 hours by variable: after the min-max of the article,
    and after the seasonal normalization and the min-max of the detector (from its weights)."""
    part = data["change_24h"]
    features = part["features"]
    article = 100 * np.asarray(part["share"])
    inverse = np.array([1.0 / weights[name] for name in features])
    detector = 100 * inverse / inverse.sum()
    order = np.argsort(article)
    rows = np.arange(len(order))
    figure, axis = plt.subplots(figsize=(9.0 / 2.54, 8.0 / 2.54))
    axis.barh(rows + 0.19, article[order], height=0.36, color=ORANGE, label="min–max di STGAN")
    axis.barh(rows - 0.19, detector[order], height=0.36, color=BLUE,
              label="normalizzazione stagionale e min–max del rilevatore")
    axis.set_yticks(rows, [NAMES.get(features[i], features[i]) for i in order])
    # A logarithmic axis: on a linear one the cloud cover (84%) hides every other bar.
    axis.set_xscale("log")
    axis.set_xlim(0.02, 150)
    axis.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    axis.axvline(100 / len(features), color=MUTED, linestyle="--", linewidth=0.8)
    axis.text(100 / len(features) * 1.08, -1.25, "percentuali uguali", color=MUTED, fontsize=8, va="center")
    axis.set_ylim(-1.6, len(features) - 0.2)
    top = order[-1]
    axis.text(article[top] * 1.08, top_row := len(order) - 1 + 0.19, f"{article[top]:.1f}%", color=TEXT, fontsize=8,
              va="center", ha="left")
    axis.text(detector[top] * 1.15, top_row - 0.36, f"{detector[top]:.1f}%", color=TEXT, fontsize=8, va="center",
              bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    axis.set_xlabel("Percentuale della variazione quadratica media su 24 ore (%)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.tick_params(axis="y", length=0)
    axis.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=1, borderaxespad=0.2)
    save(figure, "era5_variazione_24h.pdf")


def correlation_figure(data):
    """Median correlation of the anomalies of two cells against their distance, with the quartiles."""
    part = data["correlation"]
    edges = np.asarray(part["edges_km"], dtype=float)
    centres = (edges[:-1] + edges[1:]) / 2
    figure, axis = plt.subplots(figsize=(WIDTH_CM / 2.54, 6.0 / 2.54))
    # The legend lists the curves from the top down, as they are drawn: pressure, temperature, cloud cover, rain.
    order = ("mean_sea_level_pressure", "2m_temperature", "total_cloud_cover", "total_precipitation")
    assert sorted(order) == sorted(part["features"])
    for feature in order:
        color = VARIABLE_COLOURS[feature][0]
        rows = part["bins"][feature]
        keep = np.array([row[0] is not None for row in rows])
        median = np.array([row[0] for row in rows], dtype=float)
        low = np.array([row[1] for row in rows], dtype=float)
        high = np.array([row[2] for row in rows], dtype=float)
        axis.plot(centres[keep], median[keep], color=color, linewidth=1.5, label=NAMES.get(feature, feature))
        axis.fill_between(centres[keep], low[keep], high[keep], color=color, alpha=0.15, linewidth=0)
    axis.axhline(0, color=MUTED, linewidth=0.5)
    axis.set_xlim(0, edges[-1])
    axis.set_ylim(-0.1, 1.0)
    axis.set_xlabel("Distanza fra le celle (km)")
    axis.set_ylabel("Correlazione delle differenze\ndal normale per mese e ora")
    tidy(axis)
    axis.legend(fontsize=8, loc="upper right", labelcolor="linecolor")
    save(figure, "era5_correlazione_distanza.pdf")


def main():
    data = json.loads((HERE / "dati" / "era5_dati.json").read_text(encoding="utf-8"))
    weights = json.loads((HERE / "dati" / "era5_pesi.json").read_text(encoding="utf-8"))["weights"]
    cycles_figure(data)
    change_figure(data, weights)
    correlation_figure(data)


if __name__ == "__main__":
    main()

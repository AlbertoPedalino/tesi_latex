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

HERE = pathlib.Path(__file__).parent
WIDTH_CM = 13.2
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e2e1dc"
CELL_COLORS = (BLUE, ORANGE, AQUA)
NAMES = {
    "2m_temperature": "Temperatura a 2 m",
    "10m_u_component_of_wind": "Vento zonale a 10 m",
    "10m_v_component_of_wind": "Vento meridionale a 10 m",
    "mean_sea_level_pressure": "Pressione al livello del mare",
    "total_precipitation": "Precipitazione totale",
    "total_cloud_cover": "Copertura nuvolosa totale",
    "surface_solar_radiation_downwards": "Radiazione solare alla superficie",
    "volumetric_soil_water_layer_1": "Umidità del suolo",
    "friction_velocity": "Velocità d'attrito",
    "boundary_layer_height": "Altezza dello strato limite",
    "u_component_of_wind": "Vento zonale a 850 hPa",
    "v_component_of_wind": "Vento meridionale a 850 hPa",
    "vertical_velocity": "Velocità verticale a 850 hPa",
    "temperature": "Temperatura a 850 hPa",
    "specific_humidity": "Umidità specifica a 850 hPa",
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
             "total_cloud_cover": ("Nuvolosità (0–1)", np.asarray, (0.0, 1.0))}
    figure, axes = plt.subplots(2, 2, figsize=(11.8 / 2.54, 8.4 / 2.54), sharex="col",
                                gridspec_kw={"width_ratios": [1.5, 1]})
    for row, feature in enumerate(part["features"]):
        label, convert, (lowest, highest) = units[feature]
        for color, (cell, values) in zip(CELL_COLORS, part["cells"].items()):
            series = values[feature]
            mean = convert(series["monthly_mean"])
            std = np.asarray(series["monthly_std"])
            months = np.arange(1, 13)
            axes[row, 0].plot(months, mean, color=color, linewidth=1.5, label=cell)
            axes[row, 0].fill_between(months, np.maximum(mean - std, lowest), np.minimum(mean + std, highest),
                                      color=color, alpha=0.15, linewidth=0)
            for month, style in (("1", "--"), ("7", "-")):
                axes[row, 1].plot(part["hours"], convert(series["hourly_mean"][month]), style, color=color,
                                  linewidth=1.2)
        axes[row, 0].set_ylabel(label)
        for axis in axes[row]:
            tidy(axis)
    axes[1, 0].set_xticks(np.arange(1, 13), MONTHS)
    axes[1, 0].set_xlabel("Mese")
    axes[1, 1].set_xticks([0, 6, 12, 18])
    axes[1, 1].set_xlabel("Ora UTC")
    axes[0, 0].set_title("Ciclo stagionale", fontsize=9, color=TEXT)
    axes[0, 1].set_title("Ciclo giornaliero", fontsize=9, color=TEXT)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=MUTED, linestyle="--"), plt.Line2D([], [], color=MUTED, linestyle="-")]
    labels += ["gennaio", "luglio"]
    figure.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=7.5,
                  columnspacing=1.0, handlelength=1.6)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save(figure, "era5_cicli.pdf")


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
    figure, axis = plt.subplots(figsize=(9.6 / 2.54, 8.0 / 2.54))
    axis.barh(rows + 0.19, article[order], height=0.36, color=ORANGE, label="articolo: min–max")
    axis.barh(rows - 0.19, detector[order], height=0.36, color=BLUE,
              label="detector: normalizzazione stagionale e min–max")
    axis.set_yticks(rows, [NAMES.get(features[i], features[i]) for i in order])
    axis.axvline(100 / len(features), color=MUTED, linestyle="--", linewidth=0.8)
    axis.text(100 / len(features) + 1, -0.85, "quote uguali", color=MUTED, fontsize=8, va="center")
    axis.set_ylim(-1.2, len(features) - 0.2)
    top = order[-1]
    axis.text(article[top] + 1, top_row := len(order) - 1 + 0.19, f"{article[top]:.1f}%", color=TEXT, fontsize=8,
              va="center", ha="left")
    axis.text(detector[top] + 1, top_row - 0.38, f"{detector[top]:.1f}%", color=TEXT, fontsize=8, va="center")
    axis.set_xlabel("Quota della variazione quadratica media su 24 ore (%)")
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
    for color, feature in zip((ORANGE, BLUE, AQUA, PURPLE), part["features"]):
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
    axis.set_ylabel("Correlazione delle anomalie")
    tidy(axis)
    axis.legend(fontsize=8, loc="upper right")
    save(figure, "era5_correlazione_distanza.pdf")


def main():
    data = json.loads((HERE / "dati" / "era5_dati.json").read_text(encoding="utf-8"))
    weights = json.loads((HERE / "dati" / "era5_pesi_good2.json").read_text(encoding="utf-8"))["weights"]
    cycles_figure(data)
    change_figure(data, weights)
    correlation_figure(data)


if __name__ == "__main__":
    main()

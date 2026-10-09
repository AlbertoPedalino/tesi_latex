"""Generate figures/sde_traiettorie.tex: five Euler-Maruyama paths of a linear SDE for two diffusion values.

Illustrative figure, not a result of the thesis. Run from the thesis folder:
    python figures/genera_sde_traiettorie.py
"""
import pathlib

import numpy as np

THETA, MU, X0 = 2.5, 9.0, 0.0          # drift f(x) = THETA * (MU - x)
STEPS, PATHS, SEED = 80, 5, 7
PANELS = (("Diffusione bassa", 0.25), ("Diffusione alta", 4.0))
WIDTH, HEIGHT, GAP = 5.3, 3.3, 1.5     # cm
COLOURS = ("blue!70!black", "orange!85!black", "green!55!black", "red!75!black", "violet!80!black")


def simulate(diffusion, generator):
    step = 1.0 / STEPS
    paths = np.empty((PATHS, STEPS + 1))
    paths[:, 0] = X0
    for k in range(STEPS):
        noise = generator.standard_normal(PATHS)
        paths[:, k + 1] = paths[:, k] + THETA * (MU - paths[:, k]) * step + diffusion * np.sqrt(step) * noise
    return paths


generator = np.random.default_rng(SEED)
panels = [(title, simulate(diffusion, generator)) for title, diffusion in PANELS]
low = min(0.0, min(paths.min() for _, paths in panels))
high = max(paths.max() for _, paths in panels)
low, high = np.floor(low), np.ceil(high)
ticks = [value for value in range(int(low), int(high) + 1) if value % 3 == 0]

lines = ["% Generato da figures/genera_sde_traiettorie.py: non modificare a mano.",
         "\\begin{tikzpicture}[font=\\footnotesize]"]
for index, (title, paths) in enumerate(panels):
    lines.append(f"    \\begin{{scope}}[shift={{({index * (WIDTH + GAP):.2f},0)}}]")
    lines.append(f"        \\draw[gray!60] (0,0) rectangle ({WIDTH},{HEIGHT});")
    for tick in (0, 0.5, 1):
        lines.append(f"        \\draw[gray!60] ({tick * WIDTH:.2f},0) -- ++(0,-0.08) node[below] {{{tick:g}}};")
    for tick in ticks:
        y = (tick - low) / (high - low) * HEIGHT
        lines.append(f"        \\draw[gray!60] (0,{y:.2f}) -- ++(-0.08,0) node[left] {{{tick}}};")
    for colour, path in zip(COLOURS, paths):
        points = " ".join(f"({k / STEPS * WIDTH:.3f},{(value - low) / (high - low) * HEIGHT:.3f})"
                          for k, value in enumerate(path))
        lines.append(f"        \\draw[{colour},line width=0.6pt,line join=round] plot coordinates {{{points}}};")
    lines.append(f"        \\node[above] at ({WIDTH / 2:.2f},{HEIGHT}) {{\\small {title}}};")
    lines.append(f"        \\node[below] at ({WIDTH / 2:.2f},-0.45) {{$t$}};")
    if index == 0:
        lines.append(f"        \\node[rotate=90] at (-0.75,{HEIGHT / 2:.2f}) {{$x_t$}};")
    lines.append("    \\end{scope}")
lines.append("\\end{tikzpicture}")

target = pathlib.Path(__file__).with_name("sde_traiettorie.tex")
target.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(target.name, "range", low, high)

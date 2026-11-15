"""One order and one colour for every method, in every figure and table of Chapter 6 and Appendix A.

Order: G_n, the change in three hours, the distance from normal, then the six shallow models.
Colours (as in Figures 6.7c and A.1): G_n blue, distance from normal orange, change in three hours green, the six
shallow models grey. The three hues pass the colour-vision checks of the dataviz validator (light surface: lightness
band, chroma floor, CVD separation ΔE 9.2 deutan, normal-vision ΔE 24); the green is below 3:1 contrast on white, so
every method is also named next to its mark. The grey of the shallow models is a de-emphasis, not an identity: the
six are named one by one where they must be told apart.
"""
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
GREEN_TEXT = "#12875d"  # green for text on white
SHALLOW_GREY = "#b9b7b0"
SHALLOW_TEXT = "#6f6d67"  # grey for text on white
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e2e1dc"

ORDER = ("Gn", "change_3h", "normal", "pca", "cblof", "mcd", "hbos", "copod", "iforest")
SHALLOW = ("pca", "cblof", "mcd", "hbos", "copod", "iforest")
LABELS = {"Gn": r"$G_\mathrm{n}$", "change_3h": "Variazione in tre ore", "normal": "Distanza dal normale",
          "pca": "PCA", "cblof": "CBLOF", "mcd": "MCD", "hbos": "HBOS", "copod": "COPOD", "iforest": "Isolation Forest"}
COLOURS = {"Gn": BLUE, "change_3h": GREEN, "normal": ORANGE, **{key: SHALLOW_GREY for key in SHALLOW}}
TEXT_COLOURS = {"Gn": BLUE, "change_3h": GREEN_TEXT, "normal": ORANGE, **{key: SHALLOW_TEXT for key in SHALLOW}}
# Names used in the CSV files of the server runs for the same methods.
ALIASES = {"G+clim": "Gn", "G_n": "Gn"}


def key_of(name):
    """The key of ORDER for a method name read from a file."""
    return ALIASES.get(name, name)

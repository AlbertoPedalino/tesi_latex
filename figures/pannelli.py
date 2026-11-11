"""Save the panels of one figure as separate PDFs, stem_a.pdf, stem_b.pdf, ..., for subfigures with their own
captions. Every panel is cut from the same drawing, so fonts and sizes stay those of the whole figure, and in LaTeX
each part is included at its natural size."""
import pathlib

from matplotlib.transforms import Bbox


def save_panels(figure, groups, folder, stem, pad=0.04, **options):
    """groups: one list of artists per panel (its axes, colour bar, legend...), in the order of the letters."""
    figure.canvas.draw()
    # Freeze the layout: the constrained layout depends a little on the dpi, and savefig draws again at another dpi,
    # so the boxes measured here would not match the saved drawing.
    figure.set_layout_engine("none")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    inches = figure.dpi_scale_trans.inverted()
    names = []
    for letter, group in zip("abcdefgh", groups):
        boxes = [artist.get_tightbbox(renderer) for artist in group]
        for artist in group:  # the axis labels whole: the tight box of an axes may trim a label wider than the axes
            if hasattr(artist, "xaxis"):
                for text in (artist.xaxis.label, artist.yaxis.label, artist.title, artist._left_title):
                    if text.get_visible() and text.get_text():
                        boxes.append(text.get_window_extent(renderer))
        box = Bbox.union([b for b in boxes if b is not None]).transformed(inches).padded(pad)
        name = f"{stem}_{letter}.pdf"
        figure.savefig(pathlib.Path(folder) / name, bbox_inches=box, **options)
        names.append(name)
    return names

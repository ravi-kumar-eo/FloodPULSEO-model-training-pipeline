"""Shared figure style for every thesis chapter.

The palette and typography follow the figures of the FloodPULSEO data paper
(Kumar et al., 2026), measured from the published images: one base blue,
used at 100, 75 and 50 % opacity, a family of lighter blues, neutral greys,
and Lato.

Usage from a chapter script (chapters/chapterN/plot/scripts/figN_M_name.py):

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import plot_style as ps
    ps.apply()
    ...
    ps.save(fig, __file__)      # writes ../png/<script name>.png at 600 dpi
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb

# ─── Page geometry ───────────────────────────────────────────────────────────
# sapthesis, A4 layout: \textwidth = 396 pt = 13.97 cm.
TEXT_WIDTH_CM = 13.97
CM = 1 / 2.54
TEXT_WIDTH_IN = TEXT_WIDTH_CM * CM
DPI = 600

# ─── Colours ─────────────────────────────────────────────────────────────────
BLUE = "#0257CE"          # base colour of the data paper
BLUE_75 = "#4181DA"       # BLUE at 75 % opacity on white
BLUE_50 = "#80ABE6"       # BLUE at 50 % opacity on white
LIGHT_BLUE = "#94BDFC"    # permanent water in the data paper
PALE_BLUE = "#C9DCF7"

# Continent identity, used by every figure that colours by continent.
# Okabe–Ito palette (chosen by Ravi, Fig 3.2), assigned to continents with the
# dataviz validator so that continents whose dots touch on a map
# (Europe–Asia–Africa, Asia–Oceania, North–South America) stay apart for
# colour-blind readers. Drawn with transparency, the outline needs >= 0.9
# opacity to keep that (protan/deutan ΔE >= 8, normal vision ΔE >= 15).
CONTINENT = {
    "Europe": "#0072B2",
    "Australia/Oceania": "#56B4E9",
    "Asia": "#D55E00",
    "Africa": "#009E73",
    "South America": "#E69F00",
    "North America": "#CC79A7",
}
CONTINENT_EDGE_ALPHA = 0.9
CONTINENT_ORDER = list(CONTINENT)          # by number of mapping products

GREY_BG = "#D3D3D3"       # background / non-flood class
GREY_OTHER = "#B0B0B0"    # "Other" category
GREY_LAND = "#EBECEC"     # land on maps

# World and regional maps (Ravi, 16 Oct 2026: grey continents look dull): white land on a
# light-blue ocean, shared by Figures 3.2, 3.7, 3.10 and 4.3.
MAP_BG = dict(ocean="#DCE9F5", land="#FFFFFF", coast="#9FB0C0", border="#CFCFCF")
GREY_GRID = "#DDDDDD"
TEXT = "#222222"
TEXT_SOFT = "#555555"

ORANGE = "#E0942A"        # accents used on the data-paper map
TEAL = "#73C0B4"
PURPLE = "#A457BF"
YELLOW = "#FDC838"

CLIMATE = {               # ColorBrewer Greens, as in the data paper
    "Tropical": "#A1D99B",
    "Arid": "#74C476",
    "Temperate": "#005A32",
    "Cold": "#238B45",
    "Polar": "#E5F5E0",
}

FLOOD = BLUE
PERMANENT_WATER = LIGHT_BLUE
NON_FLOOD = GREY_BG
PRECIPITATION = BLUE
SOIL_MOISTURE = ORANGE


def alpha_on_white(color, alpha):
    """The opaque colour that `color` at `alpha` opacity shows on white paper."""
    r, g, b = to_rgb(color)
    return tuple(alpha * c + (1 - alpha) for c in (r, g, b))


def soften(cmap, amount=0.15, name=None):
    """Blend a colour map with white, the same lightening the data paper applies
    through transparency, so colour bars match the image exactly."""
    base = plt.get_cmap(cmap)
    colors = base(np.linspace(0, 1, base.N))      # keeps discrete maps discrete
    colors[:, :3] = colors[:, :3] * (1 - amount) + amount
    return ListedColormap(colors, name=name or f"{base.name}_soft")


def two_class_cmap(zero_color, one_color):
    """Colour map for a binary layer (0, 1)."""
    return ListedColormap([zero_color, one_color])


def ramp(colors, name="ramp"):
    return LinearSegmentedColormap.from_list(name, colors)


# ─── Dataset layers: one colour map and range per variable ───────────────────
# Shared by every map of a FloodPULSEO layer so a variable looks the same in
# every figure.
def layer_styles():
    from matplotlib.colors import LogNorm, Normalize
    viridis = ListedColormap(plt.get_cmap("viridis")(np.linspace(0.35, 1.0, 256)))
    return {
        "label": dict(cmap=two_class_cmap(GREY_LAND, FLOOD), norm=Normalize(0, 1)),
        "water": dict(cmap=two_class_cmap(GREY_LAND, PERMANENT_WATER), norm=Normalize(0, 1)),
        "ndvi": dict(cmap=ramp(["#8C6D31", "#F3ECD8", "#A6D96A", "#1A9641"], "ndvi"),
                     norm=Normalize(-0.4, 0.9)),
        # NDBI: earthy sand-tan, pale (negative) to tan (built-up); never green or grey
        "ndbi": dict(cmap=ramp(["#FBF8F1", "#E6D5B3", "#C9A96E", "#9A7442"], "ndbi"),
                     norm=Normalize(-0.5, 0.3)),
        # elevation always in greyscale (Ravi): low dark, high light
        "elev": dict(cmap=ListedColormap(plt.get_cmap("Greys_r")(np.linspace(0.45, 0.97, 256))),
                     norm=None),
        "uda": dict(cmap=soften("Blues", 0.10), norm=LogNorm(1e-2, 1e4)),
        "hand": dict(cmap=soften(viridis, 0.10), norm=Normalize(0, 100)),
        # clay: warm grey (clay is greyish), kept apart from the neutral grey of elevation
        "clay": dict(cmap=ramp(["#F6F4F1", "#DCD6CE", "#B9B0A4", "#8F857A"], "clay"),
                     norm=Normalize(0, 80)),
        # sand: light tan; clay and sand share one 0–80 % range so they compare directly
        "sand": dict(cmap=ramp(["#FEFBF3", "#F6EAC9", "#EBD49A", "#D6B76C"], "sand"),
                     norm=Normalize(0, 80)),
    }


# ─── Typography and axes ─────────────────────────────────────────────────────
FONT_SIZE = 7             # at print size; panel titles one step larger
TITLE_SIZE = 8


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Lato", "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.titlepad": 4,
        "axes.labelsize": FONT_SIZE,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": TEXT,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": False,
        "grid.color": GREY_GRID,
        "grid.linewidth": 0.5,
        "xtick.labelsize": FONT_SIZE - 0.5,
        "ytick.labelsize": FONT_SIZE - 0.5,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "legend.fontsize": FONT_SIZE,
        "legend.handlelength": 1.2,
        "legend.handleheight": 0.8,
        "text.color": TEXT,
        "mathtext.default": "regular",
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "savefig.dpi": DPI,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "image.interpolation": "nearest",
    })


# ─── Soft chart style (Figure 3.8, chosen by Ravi) ───────────────────────────
# Charts: grey axes and ticks, dark-grey text one to two points below the map
# figures, and every series drawn as a soft transparent fill with a stronger
# outline of the same colour.
SOFT_AXES = "#9A9A9A"
FILL_ALPHA, EDGE_ALPHA = 0.18, 0.6


def apply_soft():
    """apply() plus the chart style of Figure 3.8."""
    apply()
    mpl.rcParams.update({
        "axes.labelsize": FONT_SIZE - 1,
        "xtick.labelsize": FONT_SIZE - 1.5,
        "ytick.labelsize": FONT_SIZE - 1.5,
        "axes.titlesize": FONT_SIZE,
        "axes.titleweight": "normal",
        "legend.fontsize": FONT_SIZE - 1,
        "axes.edgecolor": SOFT_AXES,
        "xtick.color": SOFT_AXES,
        "ytick.color": SOFT_AXES,
        "xtick.labelcolor": TEXT_SOFT,
        "ytick.labelcolor": TEXT_SOFT,
        "axes.labelcolor": TEXT_SOFT,
        "text.color": TEXT_SOFT,
    })


def soft(color, fill=None, edge=None):
    """RGBA (face, edge) pair: a soft transparent fill and a stronger outline.
    Defaults are read at call time, so apply_clear() raises them for every later call."""
    fill = FILL_ALPHA if fill is None else fill
    edge = EDGE_ALPHA if edge is None else edge
    r, g, b = to_rgb(color)
    return (r, g, b, fill), (r, g, b, edge)


# ─── Clear style: the Chapter 5 rules, standard for every chapter ───────────
# Ravi, 12–13 Oct 2026 (Chapter 5) and 16 Oct 2026 (all chapters): grey text and
# axes and 5.5 pt ticks are hard to read in print and on slides. Black text,
# axis lines and ticks; text 2 pt larger than apply_soft() (at print size:
# smallest 6.5 pt, ticks 7.5 pt, labels and legends 8 pt, panel titles 9 pt);
# fills at 0.6 strength with an outline in the full colour.
CLEAR_FONT_SIZE = 9
CLEAR_FILL, CLEAR_EDGE = 0.6, 1.0
DOT_ALPHA = 0.8           # scatter and map dots


def apply_clear(base=CLEAR_FONT_SIZE):
    """apply_soft() with the Chapter 5 rules; call before drawing (it sets FONT_SIZE,
    which the scripts read at draw time). Charts use the default base (9); diagrams,
    whose boxes are sized for their text, pass a smaller base (8)."""
    global FONT_SIZE, TITLE_SIZE, FILL_ALPHA, EDGE_ALPHA
    FONT_SIZE, TITLE_SIZE = base, base + 1
    FILL_ALPHA, EDGE_ALPHA = CLEAR_FILL, CLEAR_EDGE
    apply_soft()
    mpl.rcParams.update({
        "text.color": "black", "axes.labelcolor": "black", "axes.titlecolor": "black",
        "xtick.labelcolor": "black", "ytick.labelcolor": "black", "legend.labelcolor": "black",
        "axes.edgecolor": "black", "xtick.color": "black", "ytick.color": "black",
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "lines.linewidth": 1.3,
    })


def ygrid(ax):
    """Light grid at the labelled y values and an unlabelled minor tick halfway between them (Chapter 5)."""
    from matplotlib.ticker import AutoMinorLocator
    ax.yaxis.grid(True, color=GREY_GRID, lw=0.5)
    ax.set_axisbelow(True)
    if ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def save(fig, script_file, suffix="", subdir=""):
    """Write the figure as ../png/<subdir>/<script name><suffix>.png at 600 dpi (subdir empty: ../png/)."""
    script = Path(script_file).resolve()
    out = script.parent.parent / "png" / subdir / f"{script.stem}{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")
    return out

"""Common dimensions and colors for manuscript figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, RED, GREEN, PURPLE, GRAY = "#176B91", "#BD5935", "#12866E", "#7961A0", "#59616B"
COLORS = dict(
    sapp=GREEN,
    mode=BLUE,
    mpurge=RED,
    chamfer="#333333",
    pnn="#A77A1E",
    cnn=PURPLE,
    residual="#777777",
    sst="#009E73",
    cao="#CC79A7",
)
NAMES = dict(
    sapp="SAPP",
    mode="SAPP: mode",
    mpurge="MPUrge-MAP",
    chamfer="Chamfer weighted kNN",
    pnn="P-NN",
    cnn="CNN regressor",
    residual="Residual map",
    sst="L-SwiGLU",
    cao="PDP Transformer",
)
ROOM_NAMES = {"L": "L room", "R2_concave_T": "T room", "R3_oblique_hexagon": "Oblique + diffuse"}


def configure():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig, name, output):
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        folder / f"{name}.pdf",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={"Author": "", "Creator": "SAPP", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(folder / f"{name}.png", dpi=170, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

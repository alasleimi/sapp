"""Method diagram, room geometry and held-out delay predictions."""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D
from sapp import load_anchor_map
from sapp.fields import predict_intensity
from sapp.kernels import range_surfaces
from .style import BLUE, RED, GREEN, GRAY, ROOM_NAMES, save

ORANGE = "#C64F17"


def method(results, output):
    fig, ax = plt.subplots(figsize=(7.05, 1.67))
    ax.set(xlim=(0, 10), ylim=(0, 2.2))
    ax.axis("off")
    boxes = [
        (
            0.05,
            0.40,
            1.75,
            1.30,
            "SURVEY",
            r"Coordinates $\mathbf{x}_i$" + "\n" + r"Delay sets $\mathcal{Y}_i$",
            BLUE,
        ),
        (2.10, 1.16, 2.15, 0.91, "EFFECTIVE SOURCES", "Range surfaces\nVisibility fields", BLUE),
        (
            2.10,
            0.05,
            2.15,
            0.91,
            "RESIDUAL DELAYS",
            "Unexplained peaks\nSpatial interpolation",
            GREEN,
        ),
        (
            4.75,
            0.40,
            2.25,
            1.30,
            "DELAY-INTENSITY MAP",
            r"Intensity $f=a+b$" + "\n" + r"Expected count $A$",
            BLUE,
        ),
        (7.72, 0.40, 2.22, 1.30, "POSITION ESTIMATE", "Query likelihood\nMode or median", RED),
    ]
    for x, y, w, h, title, body, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=.035,rounding_size=.07",
                linewidth=0.85,
                edgecolor=color,
                facecolor=color + "0D",
            )
        )
        ax.text(
            x + w / 2,
            y + h - 0.19,
            title,
            ha="center",
            va="center",
            fontsize=7.6,
            weight="bold",
            color=color,
        )
        ax.text(
            x + w / 2,
            y + h / 2 - 0.08,
            body,
            ha="center",
            va="center",
            fontsize=7.7,
            linespacing=1.7,
        )
    for start, end in [
        ((1.83, 1.19), (2.06, 1.55)),
        ((1.83, 0.94), (2.06, 0.52)),
        ((4.3, 1.6), (4.7, 1.25)),
        ((4.3, 0.50), (4.7, 0.87)),
        ((7.05, 1.04), (7.67, 1.04)),
    ]:
        ax.add_patch(
            FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, color=GRAY, lw=0.8)
        )
    ax.text(8.83, 2.02, r"Query delays $\mathcal{Y}$", ha="center", fontsize=8)
    ax.add_patch(
        FancyArrowPatch(
            (8.83, 1.89), (8.83, 1.73), arrowstyle="-|>", mutation_scale=8, color=GRAY, lw=0.8
        )
    )
    save(fig, "method", output)


def rooms(results, output):
    records = results.rooms
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.10), layout="constrained")
    for i, (ax, r) in enumerate(zip(axes, records)):
        poly = np.asarray(r["polygon_m"])
        ref = np.asarray(r["reference_xy_m"])
        ax.add_patch(Polygon(poly, facecolor="#F7F8FA", edgecolor="#333C46", lw=1.25))
        ax.scatter(*ref.T, s=4, color=BLUE, alpha=0.55, zorder=3, label="Survey")
        for ap_id, ap in r["access_points_m"].items():
            ax.scatter(*ap[:2], marker="*", s=95, color=RED, edgecolor="white", lw=0.5, zorder=6)
            ax.annotate(
                ap_id[-1],
                ap[:2],
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color=RED,
                weight="bold",
            )
        for obj in r["layouts"][-1]["objects"]:
            v = np.asarray(obj["vertices_m"])
            ax.plot(v[[0, 2], 0], v[[0, 2], 1], color="#292F37", lw=2, zorder=5)
        ax.set_aspect("equal")
        ax.set(xlabel="x (m)", ylabel="y (m)", xlim=(-0.8, 7.6), ylim=(-0.4, 7.4))
        ax.set_title(
            f"({chr(97 + i)}) {ROOM_NAMES[r['room_id']]}\n{len(ref)} survey positions per AP",
            fontsize=8.5,
        )
        ax.set_xticks([0, 2, 4, 6])
        ax.set_yticks([0, 2, 4, 6])
        ax.grid(False)
    save(fig, "rooms", output)


def surface(results, output):
    room = results.rooms[-1]
    rid = room["room_id"]
    ap = next(iter(room["access_points_m"]))
    xy = np.asarray(room["reference_xy_m"])
    poly = np.asarray(room["polygon_m"])
    record = json.loads(
        (results.source / "nist/diagnostics" / f"surfaces_{rid}_{ap}.json").read_text()
    )["folds"][1]
    train = np.asarray(record["train_indices"])
    held = np.asarray(record["heldout_indices"])
    model = load_anchor_map(results.source / "nist/diagnostics" / f"holdout_{rid}_{ap}_1.npz")
    np.testing.assert_array_equal(model.reference_xy, xy[train])
    with np.load(results.root / "data/power" / f"{rid}_{ap}.npz") as z:
        sets = z["train_peaks"][:, 0]
    yy = np.unique(xy[:, 1])
    level = yy[len(yy) // 2]
    indices = np.flatnonzero(xy[:, 1] == level)
    xx = np.linspace(xy[indices, 0].min(), xy[indices, 0].max(), 400)
    line = np.column_stack((xx, np.full(len(xx), level)))
    ranges = range_surfaces(line, model.anchors)
    weights = predict_intensity(model, line)
    chosen = np.argsort(-weights.mean(0))[:6]
    lower = np.floor((ranges[:, chosen].min() - 0.35) * 2) / 2
    upper = np.ceil((ranges[:, chosen].max() + 0.6) * 2) / 2
    bounds = [xy[held, 0].min() - 0.25, xy[held, 0].max() + 0.25]
    fig, axes = plt.subplots(
        1, 2, figsize=(7.05, 2.75), gridspec_kw={"width_ratios": [1, 2.05]}, layout="constrained"
    )
    a, b = axes
    outline = np.vstack((poly, poly[0]))
    a.plot(*outline.T, color="#59616B", lw=1)
    a.axvspan(*bounds, color=ORANGE, alpha=0.09, zorder=0)
    a.scatter(*xy[train].T, s=9, c=GRAY, zorder=2)
    a.scatter(*xy[held].T, s=16, facecolors="white", edgecolors=ORANGE, lw=0.8, zorder=3)
    a.plot([xx.min(), xx.max()], [level, level], "k--", lw=1, zorder=4)
    a.annotate(
        "Slice in (b)",
        xy=(xx.max() - 0.35, level),
        xytext=(xy[:, 0].max() - 2, 1.1),
        fontsize=7,
        arrowprops={"arrowstyle": "->", "lw": 0.8},
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
    a.set(
        xlabel="Receiver x (m)",
        ylabel="Receiver y (m)",
        title="(a) Survey positions",
        aspect="equal",
    )
    a.set_xticks([0, 2, 4, 6])
    a.set_yticks([0, 2, 4, 6])
    b.axvspan(*bounds, color=ORANGE, alpha=0.09, zorder=0)
    for x in bounds:
        b.axvline(x, color=ORANGE, alpha=0.6, lw=0.8, ls=":")
    inside = (xx >= bounds[0]) & (xx <= bounds[1])
    for j in chosen:
        b.plot(xx, np.where(inside, np.nan, ranges[:, j]), color=BLUE, lw=1.25, zorder=2)
        b.plot(xx, np.where(inside, ranges[:, j], np.nan), color=BLUE, lw=1.55, ls="--", zorder=2)
    for i in indices:
        values = sets[i][np.isfinite(sets[i])]
        if i in held:
            b.scatter(
                np.full(len(values), xy[i, 0]),
                values,
                s=25,
                facecolors="white",
                edgecolors=ORANGE,
                lw=1,
                zorder=4,
            )
        else:
            b.scatter(np.full(len(values), xy[i, 0]), values, s=12, color=GRAY, zorder=3)
    b.text(
        np.mean(bounds),
        upper - 0.12,
        "Withheld strip",
        ha="center",
        va="top",
        fontsize=8,
        color=ORANGE,
    )
    b.set(
        xlabel="Receiver x along slice (m)",
        ylabel=r"Path length $c\tau$ (m)",
        title=f"(b) Predicted and observed paths at y = {level:.2f} m",
        xlim=(xx.min() - 0.15, xx.max() + 0.15),
        ylim=(lower, upper),
    )
    b.grid(alpha=0.15, lw=0.5)
    legend = [
        Line2D([], [], marker="o", ls="", color=GRAY, ms=3, label="Survey used for fitting"),
        Line2D(
            [], [], marker="o", ls="", mfc="white", mec=ORANGE, ms=4, label="Withheld observations"
        ),
        Line2D([], [], color=BLUE, lw=1.3, label="Fitted path predictions"),
    ]
    b.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=3,
        frameon=False,
        fontsize=6.8,
        columnspacing=1.0,
    )
    save(fig, "surface_validation", output)

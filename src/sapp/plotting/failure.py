"""Likelihood map and peak intensities for the largest L-room mode error."""

from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from sapp import load_anchor_map, localize
from sapp.localization import candidate_grid
from sapp.kernels import range_surfaces
from sapp.fields import predict_intensity, _prepare_background, _background_density
from sapp.likelihood import _semiparametric_components
from sapp.experiments.common import write_json
from sapp.experiments.predict import scores_at
from .style import save


def failure(results, output):
    li, ai, index = np.unravel_index(results.cube("mode")[0].argmax(), (9, 2, 100))
    room = results.rooms[0]
    ap = list(room["access_points_m"])[ai]
    layout = room["layouts"][li]["layout_id"]
    truth = np.asarray(room["query_xy_m"])[index]
    model = load_anchor_map(results.source / "nist/models" / f"L_{ap}_standard_9_full_map.npz")
    with np.load(results.root / "data/nist/receiver" / f"L_{ap}_{layout}_standard_peaks.npz") as z:
        row = z["ranges"][index, :9]
    query = np.sort(row[np.isfinite(row)])
    mode = localize([query], model)[0]
    median = localize(
        [query], replace(model, parameters=replace(model.parameters, decode="posterior_median"))
    )[0]
    grid, _ = candidate_grid(model.reference_xy, 0.1)
    scores = scores_at(model, grid, query)
    true_score, mode_score = scores_at(model, np.vstack((truth, mode)), query)
    ranges = np.linspace(0, 25, 1001)
    points = np.vstack((truth, mode))
    bg, mass = _background_density(model, _prepare_background(model, points), ranges)
    intensity, _ = _semiparametric_components(
        ranges,
        range_surfaces(points, model.anchors),
        predict_intensity(model, points),
        scale_m=0.16,
        range_limit_m=25,
        background_density=bg,
        background_integral=mass,
    )
    intensity = 0.75 * intensity + 2 / 25
    fig, axes = plt.subplots(
        1, 2, figsize=(7.05, 2.3), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25]}
    )
    im = axes[0].scatter(
        *grid.T,
        c=np.maximum(scores - mode_score, -16),
        s=3,
        cmap="viridis",
        vmin=-16,
        vmax=0,
        rasterized=True,
    )
    axes[0].add_patch(Polygon(np.asarray(room["polygon_m"]), fill=False, ec="#303841", lw=0.8))
    axes[0].scatter(
        *truth, marker="x", color="#E26936", s=65, lw=1.8, label="True coordinate", zorder=5
    )
    axes[0].scatter(
        *mode,
        marker="o",
        facecolor="white",
        edgecolor="#222222",
        s=35,
        lw=1.2,
        label="Posterior mode",
        zorder=5,
    )
    axes[0].scatter(
        *median,
        marker="D",
        facecolor="#E6B34A",
        edgecolor="#222222",
        s=28,
        lw=0.7,
        label="Geometric median",
        zorder=5,
    )
    axes[0].set(
        xlabel="x (m)",
        ylabel="y (m)",
        title="(a) Largest L-room error",
        xlim=(-0.5, 7),
        ylim=(-0.5, 7.5),
        aspect="equal",
    )
    axes[0].legend(loc="upper right", framealpha=0.9, fontsize=6.4)
    bar = fig.colorbar(im, ax=axes[0], shrink=0.8, pad=0.02)
    bar.ax.set_title(r"$\Delta\log L$", fontsize=7, pad=5)
    for curve, color, label in zip(
        intensity, ["#E26936", "#176B91"], ["At true coordinate", "At posterior mode"]
    ):
        axes[1].plot(ranges, curve, color=color, label=label, lw=1.3)
    for q in query:
        axes[1].axvline(q, color="#303841", alpha=0.42, lw=0.8, ls=":")
    axes[1].plot([], [], color="#303841", ls=":", label="Observed query peaks")
    axes[1].set(
        xlim=(max(0, query.min() - 1.5), min(25, query.max() + 1.5)),
        xlabel="Path range (m)",
        ylabel="Predicted peak intensity (1/m)",
        title=f"(b) Competing delay explanations; {len(query)} peaks",
    )
    axes[1].legend(frameon=False, loc="upper right", fontsize=6.4)
    axes[1].grid(alpha=0.18, lw=0.5)
    save(fig, "failure_diagnostic", output)
    write_json(
        output / "failure_diagnostic.json",
        dict(
            ap=ap,
            layout=layout,
            query_index=int(index),
            truth=truth,
            mode=mode,
            geometric_median=median,
            mode_error_m=float(np.linalg.norm(mode - truth)),
            geometric_median_error_m=float(np.linalg.norm(median - truth)),
            true_log_likelihood=float(true_score),
            mode_log_likelihood=float(mode_score),
        ),
    )

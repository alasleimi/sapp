"""Error distributions, ablations and operating-condition figures."""

import numpy as np
import matplotlib.pyplot as plt
from sapp.experiments.common import read_json
from sapp.experiments.metrics import crossed_bootstrap, block_bootstrap
from sapp.experiments.reporting import errors
from .style import COLORS, NAMES, BLUE, RED, save


def curve(ax, values, key, **kwargs):
    e = np.sort(np.asarray(values).ravel())
    ax.plot(
        e, np.arange(1, len(e) + 1) / len(e), color=COLORS[key], label=NAMES[key], lw=1.35, **kwargs
    )


def cdf(results, output):
    data = {k: results.cube(k) for k in ("sapp", "mode", "mpurge", "chamfer")}
    data.update(cnn=results.neural("cnn"), pnn=results.neural("pnn"))
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.05), sharey=True, layout="constrained")
    styles = {
        "sapp": "-",
        "mode": "-",
        "mpurge": "--",
        "chamfer": ":",
        "cnn": "-.",
        "pnn": (0, (3, 1, 1, 1)),
    }
    for ri, ax in enumerate(axes):
        for key, value in data.items():
            curve(ax, value[:, ri] if value.ndim == 5 else value[ri], key, ls=styles[key])
        ax.set(
            xlabel="Position error (m)",
            xlim=(0, 6),
            ylim=(0, 1.015),
            title=["(a) L room", "(b) T room", "(c) Oblique + diffuse"][ri],
        )
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Fraction of queries")
    axes[-1].legend(loc="lower right", frameon=False, fontsize=6.1)
    save(fig, "cdf", output)


def ablations(results, output):
    names = ["Full SAPP", "Source intensity only", "Constant visibility", "Residual intensity only"]
    fig, ax = plt.subplots(figsize=(3.4, 2.35), layout="constrained")
    for shift, color, marker, methods in [
        (
            -0.10,
            BLUE,
            "o",
            ["mode", "source_only_mode", "constant_visibility_mode", "residual_mode"],
        ),
        (0.10, RED, "s", ["sapp", "source_only", "constant_visibility", "residual"]),
    ]:
        for i, key in enumerate(methods):
            value = results.cube(key)
            mean, interval = value.mean(), crossed_bootstrap(value)
            ax.errorbar(
                mean,
                i + shift,
                xerr=np.array([[mean - interval[0]], [interval[1] - mean]]),
                fmt=marker,
                color=color,
                capsize=2,
                lw=1,
                ms=3.5,
            )
    ax.plot([], [], "o", color=BLUE, label="Mode")
    ax.plot([], [], "s", color=RED, label="Geometric median")
    ax.set_yticks(range(4), names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean position error (m)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    save(fig, "ablations", output)


def sensitivity(results, output):
    fig, axes = plt.subplots(
        1, 2, figsize=(7.05, 2.30), layout="constrained", gridspec_kw={"width_ratios": [1, 1.2]}
    )
    for key, label, color in [("sapp", "SAPP: mode", BLUE), ("mpurge", "MPUrge-MAP", RED)]:
        full = results.cube("mode" if key == "sapp" else key)[0]
        values = [results.sensitivity(key, f"density{n}").mean() for n in (20, 32, 64, 100)] + [
            full.mean()
        ]
        axes[0].plot([20, 32, 64, 100, 164], values, "o-", label=label, color=color, lw=1.2, ms=4)
        axes[0].plot(
            64, results.sensitivity(key, "irregular64").mean(), "D", mfc="white", mec=color, ms=6
        )
        arrays = [full] + [
            results.sensitivity(key, c)
            for c in ("snr25", "bandwidth1", "bandwidth1_snr25", "clock")
        ]
        means = np.array([a.mean() for a in arrays])
        ci = np.array([crossed_bootstrap(a) for a in arrays])
        axes[1].barh(
            np.arange(5) + (-0.16 if key == "sapp" else 0.16),
            means,
            height=0.29,
            color=color,
            xerr=np.vstack((means - ci[:, 0], ci[:, 1] - means)),
            error_kw={"lw": 0.65, "capsize": 1.5},
        )
    axes[0].set(
        xlabel="Survey positions per AP",
        ylabel="Mean position error (m)",
        title="(a) Survey density and central gap",
        xticks=[20, 64, 100, 164],
    )
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].text(
        0.03,
        0.03,
        "Open diamonds: irregular 64-point survey",
        transform=axes[0].transAxes,
        fontsize=6.4,
    )
    axes[1].set_yticks(
        range(5),
        ["2 GHz / 40 dB", "2 GHz / 25 dB", "1 GHz / 40 dB", "1 GHz / 25 dB", "Clock ±0.30 m"],
    )
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Mean position error (m)", title="(b) Receiver and timing sensitivity")
    save(fig, "sensitivity", output)


def uwb_cdf(results, output):
    fig, ax = plt.subplots(figsize=(3.45, 2.4), layout="constrained")
    for key, style in [("sapp", "-"), ("residual", "--"), ("chamfer", ":"), ("mpurge", "-.")]:
        curve(ax, results.measured(key), key, ls=style)
    ax.set(
        xlabel="Position error (m)",
        ylabel="Fraction of test bursts",
        xlim=(0, 2.5),
        ylim=(0, 1.015),
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "uwb_cdf", output)


def recent_cdf(results, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.6), layout="constrained", sharey=True)
    for scope, ax in zip(("nist", "uwb"), axes):
        curve(ax, results.cube("sapp") if scope == "nist" else results.measured("sapp"), "sapp")
        for key, style in [("pnn", "--"), ("sst", "-."), ("cao", ":")]:
            curve(ax, results.neural(key, scope), key, ls=style)
        ax.set(
            xlabel="Position error (m)",
            xlim=(0, 6),
            ylim=(0, 1.015),
            title="NIST receiver peaks" if scope == "nist" else "Measured UWB: six anchors",
        )
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, loc="lower right")
    axes[0].set_ylabel("Fraction of queries")
    save(fig, "recent_cdf", output)


def cnn_validation(results, output):
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.25), layout="constrained", sharey=True)
    for room, ax in zip(results.rooms, axes):
        rid = room["room_id"]
        path = results.source / "cnn/validation" / f"{rid}.json"
        records = read_json(path)["history"]
        value = np.convolve([r["validation_mean"] for r in records], np.ones(10) / 10, mode="valid")
        ax.plot(np.arange(10, len(records) + 1), value, color=BLUE, lw=1.25)
        ax.axvline(250, color=RED, ls="--", lw=0.8)
        ax.set(
            xlabel="Training epoch",
            title={"L": "L room", "R2_concave_T": "T room", "R3_oblique_hexagon": "Oblique room"}[
                rid
            ],
        )
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Validation mean error (m)")
    save(fig, "cnn_validation", output)


def uwb_survey(results, output):
    with np.load(results.root / "data/uwb/train.npz") as z:
        train = z["xy"]
    with np.load(results.root / "data/uwb/test.npz") as z:
        test = z["xy"]
    with np.load(results.root / "data/uwb/survey.npz") as z:
        ids = z["landmarks"]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.6), layout="constrained")
    axes[0].plot(*train.T, color=".75", lw=0.5, label="Training trajectory")
    axes[0].scatter(*train[ids].T, s=4, color=BLUE, label="515 survey positions")
    axes[0].plot(*test.T, color=RED, lw=0.6, label="Test trajectory")
    axes[0].set(
        xlabel="x (m)", ylabel="y (m)", title="Measured survey and test coverage", aspect="equal"
    )
    axes[0].legend(frameon=False, fontsize=6)
    for key in ("mpurge", "chamfer", "residual", "sapp"):
        axes[1].plot(
            np.arange(1, 7),
            results.measured(key, "single").mean(0),
            "o-",
            label=NAMES[key],
            color=COLORS[key],
            ms=3,
        )
    axes[1].set(
        xlabel="Anchor",
        ylabel="Mean position error (m)",
        title="Individual-anchor results",
        xticks=np.arange(1, 7),
    )
    axes[1].legend(frameon=False, fontsize=6)
    axes[1].grid(alpha=0.18)
    save(fig, "uwb_survey", output)


def uwb_operating(results, output):
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.5), sharey=True, layout="constrained")
    for ax, n in zip(axes, (128, 256, 515)):
        gains, intervals = [], []
        for k in (1, 2, 3, 6):
            difference = errors(
                results.source / "uwb/operating" / f"n{n}_residual_k{k}.npz"
            ) - errors(results.source / "uwb/operating" / f"n{n}_sapp_k{k}.npz")
            gains.append(difference.mean())
            intervals.append(block_bootstrap(difference))
        gain, ci = np.asarray(gains), np.asarray(intervals)
        ax.axhline(0, color=".5", lw=0.7)
        ax.errorbar(
            [1, 2, 3, 6],
            gain,
            yerr=np.maximum(np.stack([gain - ci[:, 0], ci[:, 1] - gain]), 0),
            fmt="o-",
            color=BLUE,
            capsize=3,
            lw=1.4,
            ms=4,
        )
        ax.set(xticks=[1, 2, 3, 6], xlabel="Observed anchors", title=f"{n} survey positions")
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Residual error minus SAPP error (m)")
    save(fig, "uwb_operating", output)


FIGURES = {
    f.__name__: f
    for f in (
        cdf,
        ablations,
        sensitivity,
        uwb_cdf,
        recent_cdf,
        cnn_validation,
        uwb_survey,
        uwb_operating,
    )
}

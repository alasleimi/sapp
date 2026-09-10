"""Additional numerical tables for decoder seeds, peak counts and baseline checks."""

import numpy as np
from .common import Context, read_json
from .metrics import metrics, crossed_bootstrap
from .reporting import errors, independent_metrics


def generate(results, output, table, interval):
    rows = []
    for key, label in [
        ("mpurge", "MPUrge-MAP"),
        ("residual", "Residual only, median"),
        ("mode", "SAPP, mode"),
        ("sapp", "SAPP, median"),
    ]:
        rows.append(
            [label, *[f"{v:.3f}" for v in results.cube(key, "native").mean(axis=(1, 2, 3))]]
        )
    table(
        output,
        "supp_raw.tex",
        ["Method", "L", "T", "Oblique"],
        rows,
        "Mean position error from exact simulator delays, in metres. Each room contributes 1,800 cases.",
        "tab:rawrooms",
    )
    rows = []
    for method, duration in (("cnn100", 100), ("cnn", 250)):
        value = results.neural(method)
        for ri, label in [(0, "L"), (1, "T"), (2, "Oblique"), (None, "Pooled")]:
            selected = value if ri is None else value[:, ri]
            stat = independent_metrics(selected)
            means = selected.reshape(3, -1).mean(1)
            rows.append(
                [
                    label,
                    duration,
                    f"{stat['mean']:.3f}",
                    f"{stat['rmse']:.3f}",
                    f"{means.min():.3f}--{means.max():.3f}",
                ]
            )
    table(
        output,
        "supp_cnn.tex",
        ["Room", "Epochs", "Mean", "RMSE", "Mean across fits"],
        rows,
        "Common-survey CNN query errors, in metres, averaging three independent fits.",
        "tab:cnn",
        align="llrrl",
    )
    rows = []
    for condition, label, selected in (
        ("layout_rich", "Two APs", 300),
        ("paper_diversity", "60 APs", 150),
    ):
        path = results.source / "cnn/budget" / condition / "selection.json"
        if path.exists():
            selected = read_json(path)["epochs"]
        for duration in (100, selected):
            value = results.cnn_budget(condition, duration)
            stat = independent_metrics(value)
            means = value.reshape(3, -1).mean(1)
            rows.append(
                [
                    label,
                    duration,
                    *[f"{v:.3f}" for v in value.mean(axis=(0, 2, 3, 4))],
                    f"{stat['mean']:.3f}",
                    f"{stat['median']:.3f}",
                    f"{stat['p90']:.3f}",
                    f"{means.min():.3f}--{means.max():.3f}",
                ]
            )
    table(
        output,
        "supp_cnn_budget.tex",
        ["Regime", "Epochs", "L", "T", "Oblique", "Mean", "Median", "P90", "Mean range"],
        rows,
        "Additional CNN survey regimes, with seven training layouts. Two APs use all survey coordinates; 60 APs use 20 coordinates per room. Errors average three independent fits, in metres.",
        "tab:cnnbudget",
        align="llrrrrrrl",
    )
    fingerprints = 2 * sum((len(room["reference_xy_m"]) for room in results.rooms))
    metadata = (
        "\\begin{tabular}{lrrrrr}\\toprule"
        + "\n"
        + "Training regime & APs & Labels & Layouts & Fingerprints & Inputs\\\\\\midrule"
        + "\n"
        + f"Common survey & 2 & All & 1 & {fingerprints:,} & {fingerprints * 7:,}\\\\"
        + "\n"
        + f"Two APs, seven layouts & 2 & All & 7 & {fingerprints * 7:,} & {fingerprints * 7:,}\\\\"
        + "\n"
        + "60 APs, seven layouts & 60 & 20 & 7 & 25,200 & 25,200\\\\\\bottomrule\\end{tabular}\\par\\medskip"
        + "\n"
    )
    path = output / "generated/supp_cnn_budget.tex"
    content = path.read_text(encoding="utf-8")
    content = content.replace("\\begin{tabular}", metadata + "\\begin{tabular}", 1)
    path.write_text(content, encoding="utf-8")
    rows = []
    for si, seed in enumerate((None, 3101, 3102, 3103)):
        if seed is None:
            mode, median = (results.cube("mode")[..., ::5], results.cube("sapp")[..., ::5])
        else:
            values = []
            for key in ("sapp", "posterior_median"):
                values.append(
                    np.concatenate(
                        [
                            errors(
                                results.source
                                / "nist/diagnostics"
                                / f"seed_{room['room_id']}_{ap}_{seed}.npz",
                                key,
                            )
                            for room in results.rooms
                            for ap in room["access_points_m"]
                        ]
                    )
                )
            mode, median = values
        a, b = (metrics(mode), metrics(median))
        rows.append(
            [si + 1, f"{a['mean']:.3f}", f"{a['rmse']:.3f}", f"{b['mean']:.3f}", f"{b['rmse']:.3f}"]
        )
    table(
        output,
        "supp_seeds.tex",
        ["Fit", "Mode mean", "Mode RMSE", "Median mean", "Median RMSE"],
        rows,
        "Independent discovery fits on every fifth query coordinate, pooling all layouts and APs: 1,080 cases per fit. Errors are in metres.",
        "tab:seeds",
    )
    rows = []
    ctx = Context(results.root, results.source)
    for cap in (9, 24):
        for ri, room in enumerate(results.rooms):
            rid = room["room_id"]
            count = np.concatenate(
                [
                    np.isfinite(ctx.observations(rid, ap, layout["layout_id"], cap=cap)).sum(1)
                    for ap in room["access_points_m"]
                    for layout in room["layouts"]
                ]
            )
            if cap == 9:
                diff = results.cube("conditional")[ri] - results.cube("mode")[ri]
            else:
                diff = np.empty((9, 2, 100))
                for ai, ap in enumerate(room["access_points_m"]):
                    path = results.source / "nist/counts" / f"{rid}_{ap}.npz"
                    diff[:, ai] = (errors(path, "conditional") - errors(path, "sapp")).reshape(
                        9, 100
                    )
            rows.append(
                [
                    rid,
                    cap,
                    f"{count.mean():.2f}",
                    f"{100 * np.mean(count == cap):.1f}",
                    f"{diff.mean():.3f}",
                    interval(crossed_bootstrap(diff)),
                ]
            )
    table(
        output,
        "supp_counts.tex",
        ["Room", "Cap", "Mean peaks", "At cap (\\%)", "Count benefit", "95\\% interval"],
        rows,
        "Observed peak counts and paired error reduction from the cardinality term, in metres. Both scores use the same map and mode decoder.",
        "tab:counts",
        align="lrrrrl",
    )
    rows = []
    for room in results.rooms:
        rid = room["room_id"]
        for setting in ("published", "calibrated"):
            records = []
            for ap in room["access_points_m"]:
                path = results.source / "nist/vt_diagnostics" / f"{rid}_{ap}_{setting}.json"
                records.append(read_json(path))

            def total(key):
                return sum((np.asarray(r[key]).sum() for r in records))

            number = sum((len(r["predicted_counts"]) for r in records))
            rows.append(
                [
                    rid,
                    setting.title(),
                    f"{total('predicted_counts') / number:.2f}",
                    f"{total('observed_counts') / number:.2f}",
                    f"{100 * total('hit_weight') / total('total_weight'):.1f}",
                    f"{100 * total('covered_peaks') / total('observed_peaks'):.1f}",
                    f"{100 * (1 - total('fitted_counts') / total('track_counts')):.1f}",
                ]
            )
    table(
        output,
        "supp_vt.tex",
        [
            "Room",
            "Setting",
            "Pred. peaks",
            "Obs. peaks",
            "Agree. (\\%)",
            "Cover. (\\%)",
            "Mean fit (\\%)",
        ],
        rows,
        "VT predictions at alternating survey positions withheld from fitting. Agreement and coverage use a 0.18 m tolerance; mean fit denotes tracks with insufficient support for trilateration.",
        "tab:vtcheck",
        align="llrrrrr",
    )
    path = results.source / "uwb/selection.json"
    selected = read_json(path if path.exists() else results.root / "configs/uwb_operating.json")[
        "selected"
    ]
    rows = []
    for n in (128, 256, 515):
        for k in (1, 2, 3, 6):
            values = []
            for method in ("residual", "sapp"):
                r = next((r for r in selected if (r["n"], r["k"], r["method"]) == (n, k, method)))
                values.extend((f"{r[key]:g}" for key in ("h", "sigma", "temp")))
            rows.append([n, k, *values])
    table(
        output,
        "supp_uwb_selected.tex",
        ["Survey", "Anchors", "$b_R$", "$\\sigma_R$", "$T_R$", "$b_S$", "$\\sigma_S$", "$T_S$"],
        rows,
        "Training-selected UWB spatial factors, delay scales (metres), and temperatures. Subscripts R and S denote residual and SAPP models.",
        "tab:uwbselected",
    )

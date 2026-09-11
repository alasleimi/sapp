"""LaTeX tables computed from predictions, fitted maps and held-out observations."""

import numpy as np
from sapp import load_anchor_map
from sapp.fields import _prepare_background
from sapp.localization import candidate_grid
from .common import read_json, write_json
from .metrics import metrics, crossed_bootstrap, block_bootstrap
from .reporting import Results, errors, independent_metrics


def table(output, name, headers, rows, caption, label, wide=False, align=None):
    folder = output / "generated"
    folder.mkdir(parents=True, exist_ok=True)
    kind = "table*" if wide else "table"
    labels = {"R2_concave_T": "T", "R3_oblique_hexagon": "Oblique"}

    def cells(row):
        if isinstance(row, str):
            return row
        return " & ".join(labels.get(str(v), str(v)) for v in row) + r"\\"

    text = "\n".join(
        [
            rf"\begin{{{kind}}}[{'H' if name.startswith('supp_') else 't'}]",
            r"\centering\small",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{align or ('l' + 'r' * (len(headers) - 1))}}}",
            r"\toprule",
            cells(headers),
            r"\midrule",
            *[cells(row) for row in rows],
            r"\bottomrule",
            r"\end{tabular}",
            rf"\end{{{kind}}}",
            "",
        ]
    )
    (folder / name).write_text(text, encoding="utf-8")


def interval(value):
    return f"[{value[0]:.3f}, {value[1]:.3f}]"


def main_tables(results, output):
    keys = [
        ("mca", "MCA"),
        (
            "vt_calibrated",
            r"\shortstack[l]{VT interpolation\\(Zayets \& Steinbach, 2018)}",
        ),
        ("chamfer", "Chamfer weighted kNN"),
        ("mpurge", "MPUrge-MAP"),
        ("cnn", "CNN regressor"),
        ("mode", "SAPP (mode)"),
        ("sapp", "SAPP (geometric median)"),
        ("pnn", r"P-NN \cite{oh2024pnn}"),
    ]
    records = []
    summary = {}
    for key, label in keys:
        if key in ("cnn", "pnn"):
            data = results.neural(key)
            stat = independent_metrics(data)
            room_means = data.mean(axis=(0, 2, 3, 4))
        else:
            data = results.cube(key)
            stat = metrics(data)
            room_means = data.mean(axis=(1, 2, 3))
        records.append(
            [*room_means, stat["mean"], stat["median"], stat["p90"], 100 * stat["over2"]]
        )
        summary[key] = stat
    best = np.min(records, axis=0)
    rows = []
    for (_, label), values in zip(keys, records):
        row = []
        for j, v in enumerate(values):
            text = f"{v:.1f}" if j == 6 else f"{v:.3f}"
            row.append(r"\textbf{" + text + "}" if v == best[j] else text)
        rows.append([label, *row])
    rows.insert(0, r"\multicolumn{8}{l}{\emph{Detected-delay input}}\\")
    rows.insert(-1, r"\midrule\multicolumn{8}{l}{\emph{Power-profile input}}\\")
    table(
        output,
        "main_table.tex",
        [
            "Method",
            "L mean",
            "T mean",
            "Oblique mean",
            "Overall mean",
            "Median",
            "90th pct.",
            r"$>2$ m (\%)",
        ],
        rows,
        "Localization error after receiver simulation, using 5,400 single-AP queries. Each room contributes 1,800 queries; the last four columns pool all rooms. Errors are in metres. Delay-input methods use up to nine detected peaks; P-NN uses 24 power bins. Neural metrics average three independent fits. Best values are bold.",
        "tab:main",
        wide=True,
    )
    rows = []
    for key, label in [
        ("mpurge", "MPUrge-MAP"),
        ("chamfer", "Chamfer weighted kNN"),
        ("residual", "SAPP without sources"),
        ("sapp", "SAPP"),
    ]:
        v = metrics(results.cube(key, "native"))
        rows.append([label, f"{v['mean']:.3f}", f"{v['p90']:.3f}"])
    table(
        output,
        "raw_table.tex",
        ["Method", "Mean (m)", "90th pct. (m)"],
        rows,
        "Localization error from exact simulator path delays on the same 5,400 query positions and layouts. Both SAPP variants use geometric-median decoding.",
        "tab:raw",
    )
    rows = []
    for key, label in [
        ("mpurge", "MPUrge-MAP"),
        ("chamfer", "Chamfer weighted kNN"),
        ("residual", "Residual map"),
        ("pnn", "P-NN"),
        ("sst", "L-SwiGLU"),
        ("cao", "PDP Transformer"),
        ("sapp", "SAPP"),
    ]:
        v = (
            independent_metrics(results.neural(key, "uwb"))
            if key in ("pnn", "sst", "cao")
            else metrics(results.measured(key))
        )
        rows.append([label, *[f"{v[k]:.3f}" for k in ("mean", "median", "p90")]])
    table(
        output,
        "uwb_table.tex",
        ["Method", "Mean", "Median", "90th pct."],
        rows,
        "Joint six-anchor localization on measured UWB channels: 515 survey positions and 3,382 test bursts. Errors are in metres. Neural metrics average three independent fits.",
        "tab:uwb",
    )
    write_json(output / "metrics.json", summary)


def supplementary(results, output):
    base = results.cube("sapp")
    mode = results.cube("mode")
    mpurge = results.cube("mpurge")
    ap_rows, object_rows, map_rows = [], [], []
    for ri, room in enumerate(results.rooms):
        rid = room["room_id"]
        for ai, ap in enumerate(room["access_points_m"]):
            ap_rows.append([rid, ap, *[f"{v[ri, :, ai].mean():.3f}" for v in (mode, base, mpurge)]])
            model = load_anchor_map(
                results.source / "nist/models" / f"{rid}_{ap}_standard_9_full_map.npz"
            )
            grid, _ = candidate_grid(model.reference_xy, 0.1)
            mass = _prepare_background(model, model.reference_xy)[2].mean()
            map_rows.append(
                [rid, ap, len(model.reference_xy), len(model.anchors), len(grid), f"{mass:.2f}"]
            )
        for g, n in enumerate((1, 3, 5)):
            object_rows.append(
                [rid, n, *[f"{v[ri, g * 3 : g * 3 + 3].mean():.3f}" for v in (mode, base, mpurge)]]
            )
    table(
        output,
        "supp_ap.tex",
        ["Room", "AP", "SAPP mode", "SAPP median", "MPUrge-MAP"],
        ap_rows,
        "Mean position error by AP (metres), with 900 cases per row.",
        "tab:ap",
        align="llrrr",
    )
    table(
        output,
        "supp_objects.tex",
        ["Room", "Plates", "SAPP mode", "SAPP median", "MPUrge-MAP"],
        object_rows,
        "Mean error by obstruction count, with 600 cases per row.",
        "tab:objects",
    )
    table(
        output,
        "supp_maps.tex",
        ["Room", "AP", "Survey", "Sources", "Candidates", "Residual mass"],
        map_rows,
        "Nine-peak map structure and mean residual count at survey coordinates.",
        "tab:maps",
        align="llrrrr",
    )
    rows = []
    for key, label in [
        ("sapp", "SAPP"),
        ("residual", "Matched residual"),
        ("optimized_residual", "Independently tuned residual"),
        ("swap", "Cluster-fitted sources"),
    ]:
        value = results.cube(key)
        diff = value - base
        rows.append(
            [
                label,
                *[f"{v:.3f}" for v in value.mean(axis=(1, 2, 3))],
                f"{value.mean():.3f}",
                f"{diff.mean():.3f} {interval(crossed_bootstrap(diff))}",
            ]
        )
    table(
        output,
        "supp_optimized.tex",
        ["Model", "L", "T", "Oblique", "Pooled", "Difference from SAPP"],
        rows,
        "Source fitting and independent residual tuning. Means and paired differences are in metres.",
        "tab:optimized",
        align="lrrrrl",
    )
    rows = []
    for decoder, full, keys in [
        ("Mode", mode, ["mode", "residual_mode", "source_only_mode", "constant_visibility_mode"]),
        ("Median", base, ["sapp", "residual", "source_only", "constant_visibility"]),
    ]:
        for key, label in zip(
            keys, ["SAPP", "Residual only", "Sources only", "Constant visibility"]
        ):
            value = results.cube(key)
            for ri, room in enumerate(results.rooms):
                difference = value[ri] - full[ri]
                rows.append(
                    [
                        room["room_id"],
                        decoder,
                        label,
                        f"{value[ri].mean():.3f}",
                        f"{difference.mean():.3f}",
                        interval(crossed_bootstrap(difference)),
                    ]
                )
    table(
        output,
        "supp_components.tex",
        ["Room", "Decoder", "Model", "Mean", "Increase", r"95\% interval"],
        rows,
        "Map components under matched decoding. The increase compares each control with full SAPP.",
        "tab:components",
        align="lllrrl",
    )
    surface_rows, room_rows = [], []
    for room in results.rooms:
        gains = []
        totals = np.zeros(4)
        for ap in room["access_points_m"]:
            record = read_json(
                results.source / "nist/diagnostics" / f"surfaces_{room['room_id']}_{ap}.json"
            )
            for r in record["folds"]:
                gain = np.asarray(r["log_density_gain"])
                gains.extend(gain)
                totals += [
                    r[k] for k in ("hit_weight", "total_weight", "covered_peaks", "observed_peaks")
                ]
                surface_rows.append(
                    [
                        room["room_id"],
                        ap,
                        r["fold"],
                        len(gain),
                        r["sources"],
                        f"{gain.mean():.2f}",
                        f"{100 * r['hit_weight'] / r['total_weight']:.1f}",
                        f"{100 * r['covered_peaks'] / r['observed_peaks']:.1f}",
                    ]
                )
        room_rows.append(
            [
                room["room_id"],
                f"{np.mean(gains):.2f}",
                f"{100 * totals[0] / totals[1]:.1f}",
                f"{100 * totals[2] / totals[3]:.1f}",
            ]
        )
    table(
        output,
        "supp_surfaces.tex",
        ["Room", "AP", "Fold", "Held out", "Sources", "Gain", "Agree. (\%)", "Cover. (\%)"],
        surface_rows,
        "Contiguous-strip prediction: gain in nats per fingerprint; proximity tolerance 0.18 m.",
        "tab:suppsurfaces",
        align="lllrrrrr",
    )
    table(
        output,
        "journal_surface_table.tex",
        ["Room", "Gain (nats)", "Agreement (\%)", "Coverage (\%)"],
        room_rows,
        "Prediction of held-out survey delays, pooling both APs. Proximity tolerance is 0.18 m.",
        "tab:surfaces",
    )
    rows = []
    for scope in ("nist", "uwb"):
        target = base if scope == "nist" else results.measured("sapp")
        for key, label in [("pnn", "P-NN"), ("sst", "L-SwiGLU"), ("cao", "PDP Transformer")]:
            value = results.neural(key, scope)
            diff = value.mean(0) - target
            ci = crossed_bootstrap(diff) if scope == "nist" else block_bootstrap(diff)
            means = value.reshape(3, -1).mean(1)
            rows.append(
                [
                    scope.upper(),
                    label,
                    f"{value.mean():.3f}",
                    f"{diff.mean():.3f}",
                    interval(ci),
                    f"{means.min():.3f}--{means.max():.3f}",
                ]
            )
    table(
        output,
        "supp_recent.tex",
        ["Data", "Comparator", "Mean", "Reduction", "95\% interval", "Fit range"],
        rows,
        "Recent power-profile networks and paired reductions relative to SAPP, in metres. Neural models are fitted three times independently.",
        "tab:recentpaired",
        align="llrrll",
    )
    rows, blocks = [], []
    for n in (128, 256, 515):
        for k in (1, 2, 3, 6):
            sapp = errors(results.source / "uwb/operating" / f"n{n}_sapp_k{k}.npz")
            residual = errors(results.source / "uwb/operating" / f"n{n}_residual_k{k}.npz")
            diff = residual - sapp
            rows.append(
                [
                    n,
                    k,
                    sapp.shape[1],
                    f"{residual.mean():.3f}",
                    f"{sapp.mean():.3f}",
                    f"{diff.mean():.3f}",
                    interval(block_bootstrap(diff)),
                ]
            )
            blocks.append(
                [n, k, interval(block_bootstrap(diff)), interval(block_bootstrap(diff, 200))]
            )
    table(
        output,
        "supp_uwb_operating.tex",
        ["Positions", "Anchors", "Subsets", "Residual", "SAPP", "Gain", "95\% interval"],
        rows,
        "Measured UWB performance across survey densities and all anchor subsets. Errors and paired gains are in metres.",
        "tab:uwboperating",
        align="rrrrrrl",
    )
    table(
        output,
        "supp_uwb_blocks.tex",
        ["Positions", "Anchors", "100 bursts", "200 bursts"],
        blocks,
        "Paired residual-minus-SAPP intervals under two trajectory-block lengths, in metres.",
        "tab:uwbblocks",
        align="rrll",
    )
    rows = []
    values = {k: results.measured(k, "single") for k in ("mpurge", "chamfer", "residual", "sapp")}
    for ai in range(6):
        rows.append([ai + 1, *[f"{v[:, ai].mean():.3f}" for v in values.values()]])
    table(
        output,
        "supp_uwb_anchors.tex",
        ["Anchor", "MPUrge-MAP", "Chamfer kNN", "Residual", "SAPP"],
        rows,
        "Individual-anchor mean error with the 515-position survey, in metres.",
        "tab:uwbanchors",
    )


def generate(root, source, output):
    results = Results(root, source)
    main_tables(results, output)
    supplementary(results, output)
    from .supplementary_tables import generate as additional_tables

    additional_tables(results, output, table, interval)
    import re

    for generated in (output / "generated").glob("*.tex"):
        original = root / "paper/generated" / generated.name
        if original.exists():
            label = re.search(r"\\label\{([^}]+)\}", original.read_text(encoding="utf-8"))
            if label:
                text = generated.read_text(encoding="utf-8")
                text = re.sub(
                    r"\\label\{[^}]+\}", lambda _: r"\label{" + label.group(1) + "}", text, count=1
                )
                generated.write_text(text, encoding="utf-8")
    print("Regenerated primary tables and paired supplementary comparisons.", flush=True)

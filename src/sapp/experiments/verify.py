"""Check input integrity and compare recomputed error distributions with the paper."""

import numpy as np
from .common import read_json, write_json, sha
from .reporting import Results, errors
from .metrics import metrics


def verify_inputs(root):
    manifest = read_json(root / "data/manifest.json")
    missing = []
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or sha(path) != expected["sha256"]:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Input integrity check failed for {missing[:8]}")
    return len(manifest["files"])


def verify(ctx, data_only=False, profile="paper"):
    count = verify_inputs(ctx.root)
    if data_only:
        print(f"Verified {count} local input files.")
        return
    expected = Results(ctx.root, ctx.root / "reference")
    actual = Results(ctx.root, ctx.output)
    report = {"inputs": count, "profile": profile, "comparisons": {}, "missing": []}

    def compare(name, reader, neural=False):
        try:
            a, b = (reader(actual), reader(expected))
        except FileNotFoundError as error:
            report["missing"].append(dict(comparison=name, path=str(error.filename)))
            return
        if a.shape != b.shape:
            report["comparisons"][name] = dict(status="FAIL", shapes=[a.shape, b.shape])
            return
        current, original = (metrics(a), metrics(b))
        delta = float(np.max(np.abs(a - b)))
        status = "PASS" if delta < 5e-05 else "RETRAINED" if neural else "FAIL"
        report["comparisons"][name] = dict(
            status=status,
            maximum_error_discrepancy_m=delta,
            metrics=current,
            reference_metrics=original,
            metric_differences={k: current[k] - original[k] for k in current if k != "n"},
        )

    for scope in ("nist", "native"):
        if profile not in ("paper", "primary" if scope == "nist" else "native"):
            continue
        for key in ("sapp", "mode", "residual", "mpurge", "chamfer"):
            compare(f"{scope}/{key}", lambda r, k=key, s=scope: r.cube(k, s))
        if scope == "nist":
            compare("nist/mca", lambda r: r.cube("mca"))
    if profile in ("paper", "uwb"):
        for scope in ("single", "joint"):
            for key in ("sapp", "residual", "mpurge", "chamfer"):
                compare(f"uwb/{scope}/{key}", lambda r, k=key, s=scope: r.measured(k, s))
    if profile == "paper":
        for key in (
            "optimized_residual",
            "swap",
            "vt_published",
            "vt_shared_tolerance",
            "vt_calibrated",
            "direct_mca",
            "source_only",
            "constant_visibility",
            "residual_mode",
            "source_only_mode",
            "constant_visibility_mode",
            "integrated",
            "conditional",
        ):
            compare(f"nist/{key}", lambda r, k=key: r.cube(k))
        for condition in (
            "density20",
            "density32",
            "density64",
            "density100",
            "irregular64",
            "snr25",
            "bandwidth1",
            "bandwidth1_snr25",
            "clock",
        ):
            for key in ("sapp", "mpurge", "mca"):
                compare(
                    f"sensitivity/{condition}/{key}",
                    lambda r, k=key, c=condition: r.sensitivity(k, c),
                )
        for n in (128, 256, 515):
            for k in (1, 2, 3, 6):
                for method in ("sapp", "residual"):
                    name = f"n{n}_{method}_k{k}.npz"
                    compare(
                        f"uwb/operating/{name}",
                        lambda r, f=name: errors(r.source / "uwb/operating" / f),
                    )
        for scope in ("nist", "uwb"):
            methods = (
                ("cnn", "cnn100", "pnn", "sst", "cao") if scope == "nist" else ("pnn", "sst", "cao")
            )
            for method in methods:
                compare(
                    f"neural/{scope}/{method}",
                    lambda r, m=method, s=scope: r.neural(m, s),
                    neural=True,
                )
        for condition, selected in (("layout_rich", 300), ("paper_diversity", 150)):
            for epoch in (100, selected):
                compare(
                    f"cnn_budget/{condition}/{epoch}",
                    lambda r, c=condition, e=epoch: r.cnn_budget(c, e),
                    neural=True,
                )
        for room in actual.rooms:
            rid = room["room_id"]
            for ap in room["access_points_m"]:
                stem = f"{rid}_{ap}"
                for seed in (3101, 3102, 3103):
                    for key in ("sapp", "posterior_median"):
                        filename = f"seed_{stem}_{seed}.npz"
                        compare(
                            f"seeds/{stem}/{seed}/{key}",
                            lambda r, f=filename, k=key: errors(
                                r.source / "nist/diagnostics" / f, k
                            ),
                        )
                for key in ("sapp", "conditional", "mpurge"):

                    def count_errors(r, k=key, s=stem):
                        return errors(r.source / "nist/counts" / f"{s}.npz", k)

                    compare(f"counts24/{stem}/{key}", count_errors)
                if rid == actual.rooms[2]["room_id"]:
                    for key in ("sapp", "posterior_median", "mpurge", "mca"):

                        def diffuse_errors(r, k=key, s=stem):
                            folder = r.source / "nist/diffuse_stress"
                            return errors(folder / f"{s}.npz", k)

                        compare(f"diffuse/{stem}/{key}", diffuse_errors)
    states = {r["status"] for r in report["comparisons"].values()}
    report["status"] = (
        "FAIL"
        if "FAIL" in states
        else "INCOMPLETE"
        if report["missing"]
        else "PASS_WITH_RETRAINING_DIFFERENCES"
        if "RETRAINED" in states
        else "PASS"
    )
    write_json(ctx.output / "verification.json", report)
    print(f"Verification: {report['status']}. Report: {ctx.output / 'verification.json'}")
    if report["status"] in ("FAIL", "INCOMPLETE"):
        raise RuntimeError("See verification.json for missing stages or numerical mismatches")

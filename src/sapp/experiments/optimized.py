"""Independent residual-map selection and alternative source fitting."""

from itertools import product
import numpy as np
from sapp.types import AnchorMap
from sapp.fitting import calibrate_background, soft_association_counts
from sapp.baselines.source_fitting import cluster_sources
from .common import unpack, read_json, write_json, save_npz, residual_model
from .predict import predict


def source_model(base, peaks, radius):
    sources, support, _ = cluster_sources(base.reference_xy, peaks, radius)
    rows = unpack(peaks)
    counts, glob, clutter = soft_association_counts(
        base.reference_xy, rows, sources, base.parameters
    )
    model = AnchorMap(
        sources,
        base.reference_xy,
        counts,
        glob,
        clutter,
        base.parameters,
        support,
        np.zeros(len(sources)),
    )
    calibrate_background(model, rows)
    return model


def run(ctx, tune=False):
    selected = read_json(ctx.root / "configs/residual.json")["selected"]
    if tune:
        configs = [
            (h, s, t)
            for h, s, t in product([0.5, 1.0, 2.0, 4.0], [0.08, 0.16, 0.32, 0.64], [1.0, 2.0, 4.0])
        ]
        records = []

        def evaluate(configurations, swap=True):
            rows = []
            for room in ctx.rooms:
                xy = np.asarray(room["reference_xy_m"])
                rid = room["room_id"]
                ij = np.rint((xy - xy.min(0)) / 0.5).astype(int)
                folds = ij[:, 0] % 2 + 2 * (ij[:, 1] % 2)
                for ap, fold in product(room["access_points_m"], range(4)):
                    peaks = ctx.observations(rid, ap, "empty_survey")
                    tr, va = folds != fold, folds == fold
                    base = AnchorMap(
                        np.empty((0, 3)),
                        xy[tr],
                        np.empty((sum(tr), 0)),
                        np.empty(0),
                        1.0,
                        ctx.parameters,
                        np.empty(0),
                        np.empty(0),
                    )
                    calibrate_background(base, unpack(peaks[tr]))
                    for h, sigma, temp in configurations:
                        m = residual_model(base, h, sigma)
                        path = (
                            ctx.output
                            / "optimized/validation"
                            / f"{rid}_{ap}_{fold}_{h}_{sigma}_{temp}.json"
                        )
                        if path.exists():
                            row = read_json(path)
                        else:
                            pred = predict(
                                m, peaks[va], ctx.output / "checks" / path.name, temperature=temp
                            )
                            row = dict(
                                method="residual",
                                h=h,
                                sigma=sigma,
                                temp=temp,
                                total=float(np.linalg.norm(pred - xy[va], axis=1).sum()),
                                count=sum(va),
                            )
                            write_json(path, row)
                        rows.append(row)
                    if swap:
                        for radius in (1.0, 2.0, 3.0):
                            path = (
                                ctx.output
                                / "optimized/validation"
                                / f"{rid}_{ap}_{fold}_sources{radius}.json"
                            )
                            if path.exists():
                                row = read_json(path)
                            else:
                                model = source_model(base, peaks[tr], radius)
                                pred = predict(model, peaks[va], ctx.output / "checks" / path.name)
                                row = dict(
                                    method="swap",
                                    radius=radius,
                                    total=float(np.linalg.norm(pred - xy[va], axis=1).sum()),
                                    count=sum(va),
                                )
                                write_json(path, row)
                            rows.append(row)
            return rows

        records = evaluate(configs)

        def choose(rows, method):
            keys = ("h", "sigma", "temp") if method == "residual" else ("radius",)
            groups = {}
            for row in rows:
                if row["method"] == method:
                    groups.setdefault(tuple(row[k] for k in keys), []).append(row)
            candidates = [
                dict(
                    method=method,
                    **dict(zip(keys, k)),
                    mean_m=sum(r["total"] for r in v) / sum(r["count"] for r in v),
                )
                for k, v in groups.items()
            ]
            return min(candidates, key=lambda r: r["mean_m"])

        best = choose(records, "residual")
        additions = []
        for key, values in [
            ("h", [0.5, 1.0, 2.0, 4.0]),
            ("sigma", [0.08, 0.16, 0.32, 0.64]),
            ("temp", [1.0, 2.0, 4.0]),
        ]:
            for factor in (0.5, 2.0):
                value = best[key] * factor
                if min(values) <= value <= max(values):
                    continue
                candidate = dict(best, **{key: value})
                additions.append(tuple(candidate[k] for k in ("h", "sigma", "temp")))
        if additions:
            records.extend(evaluate(additions, swap=False))
        selected = {m: choose(records, m) for m in ("residual", "swap")}
        write_json(
            ctx.output / "optimized/selection.json", dict(selected=selected, validation=records)
        )
    for room in ctx.rooms:
        rid = room["room_id"]
        xy = np.asarray(room["reference_xy_m"])
        for ap in room["access_points_m"]:
            stem = f"{rid}_{ap}"
            dest = ctx.output / "nist/optimized" / f"{stem}.npz"
            if dest.exists():
                continue
            survey = ctx.observations(rid, ap, "empty_survey")
            base = ctx.fitted(
                ctx.output / "nist/models" / f"{stem}_standard_9_full_map.npz", xy, survey
            )
            query = np.vstack(
                [ctx.observations(rid, ap, layout["layout_id"]) for layout in room["layouts"]]
            )
            config = selected["residual"]
            residual = residual_model(base, config["h"], config["sigma"])
            sources = source_model(base, survey, selected["swap"]["radius"])
            save_npz(
                dest,
                truth=np.tile(room["query_xy_m"], (9, 1)),
                residual=predict(
                    residual,
                    query,
                    ctx.output / "checks" / f"{stem}_optimized.json",
                    temperature=config["temp"],
                ),
                swap=predict(
                    sources, query, ctx.output / "checks" / f"{stem}_cluster_sources.json"
                ),
            )
            print(f"Optimized comparisons/{stem}: complete", flush=True)

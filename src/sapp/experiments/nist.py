"""Receiver and native-delay comparisons on the three NIST rooms."""

from itertools import product
import numpy as np

from sapp.baselines import mca, virtual_transmitters as vt, chamfer
from sapp.baselines._mpurge_map_numba import predict as mpurge_predict
from .common import unpack, read_json, save_npz, write_json, residual_model
from .predict import predict


def matching_predictions(query, survey, xy):
    return dict(
        mpurge=mpurge_predict(
            query, np.isfinite(query).sum(1), survey, np.isfinite(survey).sum(1), xy
        ),
        mca=np.asarray([mca.localize(q, unpack(survey), xy, epsilon_m=0.5) for q in unpack(query)]),
    )


def select_chamfer(ctx, native=False):
    scope = "native" if native else "receiver"
    target = ctx.output / f"chamfer/{scope}_selection.json"
    if target.exists():
        return read_json(target)["selected"]
    configs = [dict(neighbors=k, power=p) for k, p in product([1, 3, 5, 9], [1, 2])]
    totals = np.zeros(len(configs))
    count = 0
    for room in ctx.rooms:
        xy = np.asarray(room["reference_xy_m"])
        ij = np.rint((xy - xy.min(0)) / 0.5).astype(int)
        fold = ij[:, 0] % 2 + 2 * (ij[:, 1] % 2)
        for ap in room["access_points_m"]:
            survey = ctx.observations(room["room_id"], ap, "empty_survey", native=native)
            for f in range(4):
                dist = chamfer.distance(survey[fold == f], survey[fold != f])
                for i, config in enumerate(configs):
                    pred = chamfer.predict(dist, xy[fold != f], **config)
                    totals[i] += np.linalg.norm(pred - xy[fold == f], axis=1).sum()
                count += sum(fold == f)
    winner = configs[int(totals.argmin())]
    write_json(
        target,
        dict(
            selected=winner,
            candidates=[dict(**c, mean_m=e / count) for c, e in zip(configs, totals)],
        ),
    )
    return winner


def run(ctx, native=False, room_filter=None, limit=None, backend="fast"):
    scope = "native" if native else "nist"
    config = select_chamfer(ctx, native)
    for room in ctx.rooms:
        rid = room["room_id"]
        if room_filter and rid != room_filter:
            continue
        xy = np.asarray(room["reference_xy_m"])
        for ap in room["access_points_m"]:
            stem = f"{rid}_{ap}"
            survey = ctx.observations(rid, ap, "empty_survey", native=native)
            filename = f"{stem}.npz" if native else f"{stem}_standard_9_full_map.npz"
            model = ctx.fitted(ctx.output / scope / "models" / filename, xy, survey)
            queries = np.vstack(
                [
                    ctx.observations(rid, ap, layout["layout_id"], native=native)
                    for layout in room["layouts"]
                ]
            )
            truth = np.tile(room["query_xy_m"], (len(room["layouts"]), 1))
            if limit:
                queries, truth = queries[:limit], truth[:limit]
            dest = ctx.output / scope / "predictions" / f"{stem}.npz"
            if dest.exists():
                continue
            check = ctx.output / "checks" / f"{scope}_{stem}"
            full = predict(model, queries, check.with_suffix(".json"), backend=backend)
            mode = predict(
                model,
                queries,
                check.with_name(check.name + "_mode.json"),
                mode=True,
                backend=backend,
            )
            residual = predict(
                residual_model(model),
                queries,
                check.with_name(check.name + "_residual.json"),
                backend=backend,
            )
            result = dict(
                truth=truth,
                sapp=full,
                mode=mode,
                residual=residual,
                **matching_predictions(queries, survey, xy),
            )
            result["chamfer"] = chamfer.predict(chamfer.distance(queries, survey), xy, **config)
            save_npz(dest, **result)
            print(
                f"{scope}/{stem}: {len(truth)} queries, SAPP mean {np.linalg.norm(full - truth, axis=1).mean():.6f} m",
                flush=True,
            )


def virtual_transmitters(ctx):
    config = read_json(ctx.root / "configs/vt.json")
    for room in ctx.rooms:
        rid = room["room_id"]
        xy = np.asarray(room["reference_xy_m"])
        for ap in room["access_points_m"]:
            stem = f"{rid}_{ap}"
            dest = ctx.output / "nist/vt" / f"{stem}.npz"
            if dest.exists():
                continue
            survey = unpack(ctx.observations(rid, ap, "empty_survey"))
            query = unpack(
                np.vstack(
                    [ctx.observations(rid, ap, layout["layout_id"]) for layout in room["layouts"]]
                )
            )
            published_xy, published_sets = vt.build_map(xy, survey)
            calibrated_xy, calibrated_sets = vt.build_map(
                xy,
                survey,
                cluster_gap=config["cluster_gap"],
                group_threshold=config["group_threshold"],
            )
            save_npz(
                dest,
                truth=np.tile(room["query_xy_m"], (9, 1)),
                published=vt.localize(query, published_xy, published_sets),
                shared_tolerance=vt.localize(query, published_xy, published_sets, tolerance=0.5),
                calibrated=vt.localize(
                    query, calibrated_xy, calibrated_sets, tolerance=config["tolerance"]
                ),
                direct_mca=vt.localize(query, xy, survey, tolerance=config["tolerance"]),
            )
            print(f"VT/{stem}: complete", flush=True)

"""Component ablations, spatial prediction, survey density and seed sensitivity."""

from copy import deepcopy
from dataclasses import replace
import numpy as np

from sapp import localize
from sapp.fields import predict_intensity, _prepare_background
from sapp.kernels import range_surfaces
from sapp.likelihood import (
    semiparametric_marginal_log_density,
    semiparametric_conditional_log_density,
)
from .common import write_json, save_npz, unpack, residual_model, farthest_indices
from .cnn import random_generator
from .nist import matching_predictions
from .predict import predict, scores_at


def peak_agreement(predicted, observed, weights=None):
    hit = coverage = total = observed_total = 0.0
    for i, (pred, obs) in enumerate(zip(predicted, observed)):
        w = np.ones(len(pred)) if weights is None else weights[i]
        total += w.sum()
        observed_total += len(obs)
        if len(pred) and len(obs):
            distance = np.abs(np.asarray(pred)[:, None] - obs[None])
            hit += np.sum(w * (distance.min(1) <= 0.18))
            coverage += np.sum(distance.min(0) <= 0.18)
    return dict(
        hit_weight=float(hit),
        total_weight=float(total),
        covered_peaks=int(coverage),
        observed_peaks=int(observed_total),
    )


def surfaces(ctx):
    for room in ctx.rooms:
        rid = room["room_id"]
        xy = np.asarray(room["reference_xy_m"])
        for ap in room["access_points_m"]:
            dest = ctx.output / "nist/diagnostics" / f"surfaces_{rid}_{ap}.json"
            if dest.exists():
                continue
            peaks = ctx.observations(rid, ap, "empty_survey")
            sets = unpack(peaks)
            records = []
            for fold, levels in enumerate(np.array_split(np.unique(xy[:, 0]), 4)):
                test = np.flatnonzero(np.isin(xy[:, 0], levels))
                train = np.flatnonzero(~np.isin(xy[:, 0], levels))
                model = ctx.fitted(
                    ctx.output / "nist/diagnostics" / f"holdout_{rid}_{ap}_{fold}.npz",
                    xy[train],
                    peaks[train],
                )
                residual = residual_model(model)
                heldout = [sets[i] for i in test]
                gains = [
                    float(
                        scores_at(model, xy[i : i + 1], sets[i])[0]
                        - scores_at(residual, xy[i : i + 1], sets[i])[0]
                    )
                    for i in test
                ]
                predicted = range_surfaces(xy[test], model.anchors)
                weights = predict_intensity(model, xy[test])
                records.append(
                    dict(
                        fold=fold,
                        train_indices=train,
                        heldout_indices=test,
                        sources=len(model.anchors),
                        log_density_gain=gains,
                        source_mass=weights.sum(1),
                        residual_mass=_prepare_background(model, xy[test])[2],
                        **peak_agreement(predicted, heldout, weights),
                    )
                )
            write_json(dest, dict(room=rid, ap=ap, folds=records))
            print(f"Spatial holdouts/{rid}/{ap}: complete", flush=True)


def components(ctx):
    for room in ctx.rooms:
        rid = room["room_id"]
        xy = np.asarray(room["reference_xy_m"])
        for ap in room["access_points_m"]:
            stem = f"{rid}_{ap}"
            dest = ctx.output / "nist/components" / f"{stem}.npz"
            if dest.exists():
                continue
            peaks = ctx.observations(rid, ap, "empty_survey")
            model = ctx.fitted(
                ctx.output / "nist/models" / f"{stem}_standard_9_full_map.npz", xy, peaks
            )
            query = np.vstack(
                [ctx.observations(rid, ap, layout["layout_id"]) for layout in room["layouts"]]
            )
            result = dict(truth=np.tile(room["query_xy_m"], (9, 1)))
            for kind in ("source_only", "constant_visibility", "residual"):
                m = residual_model(model) if kind == "residual" else deepcopy(model)
                if kind == "source_only":
                    m.background_responsibility[:] = 0
                elif kind == "constant_visibility":
                    m.soft_counts[:] = m.global_counts
                for mode in (True,) if kind == "residual" else (False, True):
                    key = kind + ("_mode" if mode else "")
                    result[key] = predict(
                        m, query, ctx.output / "checks" / f"{stem}_{key}.json", mode=mode
                    )
            for key, likelihood in [
                ("integrated", semiparametric_marginal_log_density),
                ("conditional", semiparametric_conditional_log_density),
            ]:
                result[key] = localize(unpack(query), model, likelihood=likelihood)
            save_npz(dest, **result)
            print(f"Component ablations/{stem}: complete", flush=True)


def sensitivity(ctx):
    room = ctx.rooms[0]
    rid = room["room_id"]
    xy = np.asarray(room["reference_xy_m"])
    conditions = [
        "density20",
        "density32",
        "density64",
        "density100",
        "irregular64",
        "snr25",
        "bandwidth1",
        "bandwidth1_snr25",
        "clock",
    ]
    for ap in room["access_points_m"]:
        for condition in conditions:
            dest = ctx.output / "nist/sensitivity" / f"{rid}_{ap}_{condition}.npz"
            if dest.exists():
                continue
            ids = np.arange(len(xy))
            if condition.startswith("density"):
                ids = farthest_indices(xy, int(condition[7:]))
            if condition == "irregular64":
                eligible = np.flatnonzero(
                    np.linalg.norm(xy - np.median(xy, axis=0), axis=1) >= 0.85
                )
                ids = np.sort(
                    random_generator(rid, "irregular").choice(eligible, 64, replace=False)
                )
            obs_condition = (
                condition
                if condition in ("snr25", "bandwidth1", "bandwidth1_snr25")
                else "standard"
            )
            survey = ctx.observations(rid, ap, "empty_survey", condition=obs_condition)[ids]
            model = ctx.fitted(
                ctx.output / "nist/models" / f"{rid}_{ap}_{condition}.npz", xy[ids], survey
            )
            rows = []
            for layout in room["layouts"]:
                q = ctx.observations(rid, ap, layout["layout_id"], condition=obs_condition)
                if condition == "clock":
                    shift = random_generator(rid, "clock").uniform(-0.3, 0.3, len(q))
                    q = np.clip(q + shift[:, None], 1e-6, 25)
                rows.append(q)
            query = np.vstack(rows)
            result = matching_predictions(query, survey, xy[ids])
            result["sapp"] = predict(
                model, query, ctx.output / "checks" / f"{rid}_{ap}_{condition}.json", mode=True
            )
            save_npz(dest, truth=np.tile(room["query_xy_m"], (9, 1)), **result)
            print(f"Sensitivity/{rid}/{ap}/{condition}: complete", flush=True)

    room = ctx.rooms[2]
    rid = room["room_id"]
    xy = np.asarray(room["reference_xy_m"])
    for ap in room["access_points_m"]:
        dest = ctx.output / "nist/diffuse_stress" / f"{rid}_{ap}.npz"
        if dest.exists():
            continue
        survey = ctx.observations(rid, ap, "empty_survey", condition="snr25")
        model = ctx.fitted(ctx.output / "nist/models" / f"{rid}_{ap}_snr25.npz", xy, survey)
        query = np.vstack(
            [
                ctx.observations(rid, ap, layout["layout_id"], condition="snr25")
                for layout in room["layouts"]
            ]
        )
        result = matching_predictions(query, survey, xy)
        for mode, key in ((True, "sapp"), (False, "posterior_median")):
            result[key] = predict(
                model, query, ctx.output / "checks" / f"{rid}_{ap}_snr25_{key}.json", mode=mode
            )
        save_npz(dest, truth=np.tile(room["query_xy_m"], (9, 1)), **result)
        print(f"Diffuse-room sensitivity/{ap}: complete", flush=True)


def seeds(ctx):
    for room in ctx.rooms:
        rid = room["room_id"]
        xy = np.asarray(room["reference_xy_m"])
        for ap in room["access_points_m"]:
            peaks = ctx.observations(rid, ap, "empty_survey")
            query = np.vstack(
                [ctx.observations(rid, ap, layout["layout_id"])[::5] for layout in room["layouts"]]
            )
            for seed in (3101, 3102, 3103):
                dest = ctx.output / "nist/diagnostics" / f"seed_{rid}_{ap}_{seed}.npz"
                if dest.exists():
                    continue
                model = ctx.fitted(
                    ctx.output / "nist/models" / f"{rid}_{ap}_seed{seed}.npz",
                    xy,
                    peaks,
                    replace(ctx.parameters, seed=seed),
                )
                pred = predict(model, query, ctx.output / "checks" / f"seed_{rid}_{ap}_{seed}.json")
                mode = predict(
                    model,
                    query,
                    ctx.output / "checks" / f"seed_{rid}_{ap}_{seed}_mode.json",
                    mode=True,
                )
                save_npz(
                    dest,
                    truth=np.tile(np.asarray(room["query_xy_m"])[::5], (9, 1)),
                    sapp=mode,
                    posterior_median=pred,
                    query_indices=np.arange(0, 100, 5),
                    sources=len(model.anchors),
                )
                print(f"Discovery seed/{rid}/{ap}/{seed}: complete", flush=True)


def counts(ctx):
    for room in ctx.rooms:
        rid = room["room_id"]
        xy = np.asarray(room["reference_xy_m"])
        for ap in room["access_points_m"]:
            dest = ctx.output / "nist/counts" / f"{rid}_{ap}.npz"
            if dest.exists():
                continue
            peaks = ctx.observations(rid, ap, "empty_survey", cap=24)
            model = ctx.fitted(
                ctx.output / "nist/models" / f"{rid}_{ap}_standard_24_full_map.npz", xy, peaks
            )
            query = np.vstack(
                [
                    ctx.observations(rid, ap, layout["layout_id"], cap=24)
                    for layout in room["layouts"]
                ]
            )
            pred = predict(
                model, query, ctx.output / "checks" / f"{rid}_{ap}_cap24.json", mode=True
            )
            conditional = localize(
                unpack(query), model, likelihood=semiparametric_conditional_log_density
            )
            save_npz(
                dest,
                truth=np.tile(room["query_xy_m"], (9, 1)),
                sapp=pred,
                conditional=conditional,
                **matching_predictions(query, peaks, xy),
            )

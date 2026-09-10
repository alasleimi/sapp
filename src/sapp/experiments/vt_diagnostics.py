"""Held-out VT delay prediction and a smooth-source positive control."""

import numpy as np
from scipy.spatial import cKDTree
from sapp.baselines import virtual_transmitters as vt
from sapp.kernels import range_surfaces
from .common import read_json, write_json, unpack
from .diagnostics import peak_agreement


def predict_delays(reference, survey, target, cluster_gap=0.1, group_threshold=0.5):
    tree = cKDTree(reference)
    predictions, tracks_n, fitted_n, residuals = [], [], [], []
    for point in target:
        ids = sorted(tree.query_ball_point(point, 1.0 + 1e-9))
        tracks = vt.matched_tracks(
            reference[ids],
            [survey[i] for i in ids],
            cluster_gap=cluster_gap,
            group_threshold=group_threshold,
        )
        values, fitted = [], 0
        for xy, ranges in tracks:
            source, mean = vt.source_fit(xy, ranges)
            if source is None:
                value = mean
            else:
                fitted += 1
                value = range_surfaces(point[None], source[None])[0, 0]
                residuals.append(
                    float(np.sqrt(np.mean((range_surfaces(xy, source[None])[:, 0] - ranges) ** 2)))
                )
            if 0 <= value <= 25:
                values.append(value)
        predictions.append(np.sort(values))
        tracks_n.append(len(tracks))
        fitted_n.append(fitted)
    return predictions, dict(
        track_counts=tracks_n, fitted_counts=fitted_n, fitted_range_rmse=residuals
    )


def run(ctx):
    selected = read_json(ctx.root / "configs/vt.json")
    for room in ctx.rooms:
        rid = room["room_id"]
        xy = np.asarray(room["reference_xy_m"])
        grid = np.rint((xy - xy.min(0)) / 0.5).astype(int)
        test = np.flatnonzero(grid.sum(1) % 2 == 0)
        train = np.flatnonzero(grid.sum(1) % 2 == 1)
        for ap in room["access_points_m"]:
            survey = unpack(ctx.observations(rid, ap, "empty_survey"))
            for setting, gap, group in (
                ("published", 0.1, 0.5),
                ("calibrated", selected["cluster_gap"], selected["group_threshold"]),
            ):
                path = ctx.output / "nist/vt_diagnostics" / f"{rid}_{ap}_{setting}.json"
                if path.exists():
                    continue
                predicted, diagnostic = predict_delays(
                    xy[train], [survey[i] for i in train], xy[test], gap, group
                )
                observed = [survey[i] for i in test]
                write_json(
                    path,
                    dict(
                        room=rid,
                        ap=ap,
                        heldout_indices=test,
                        train_indices=train,
                        predicted_counts=list(map(len, predicted)),
                        observed_counts=list(map(len, observed)),
                        **diagnostic,
                        **peak_agreement(predicted, observed),
                    ),
                )
    x = np.arange(0, 2.01, 0.5)
    xy = np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)
    sources = np.array([[-4.0, -3.0, 1.0], [12.0, -5.0, 2.0], [-12.0, 10.0, 1.5]])
    survey = list(range_surfaces(xy, sources))
    target = np.array([[0.3, 0.4], [0.7, 1.2], [1.3, 0.8], [1.6, 1.7]])
    expected = range_surfaces(target, sources)
    predicted, diagnostic = predict_delays(xy, survey, target)
    np.testing.assert_allclose(predicted, np.sort(expected, axis=1), rtol=0, atol=1e-5)
    dense_xy, dense_sets = vt.build_map(xy, survey)
    estimated = vt.localize(list(expected), dense_xy, dense_sets)
    write_json(
        ctx.output / "nist/vt_diagnostics/positive_control.json",
        dict(
            maximum_range_error_m=float(np.max(np.abs(predicted - np.sort(expected, axis=1)))),
            position_errors_m=np.linalg.norm(estimated - target, axis=1),
            **diagnostic,
        ),
    )

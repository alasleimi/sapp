"""Measured UWB evaluation with shared anchor subsets and spatial validation."""

from itertools import combinations, product
import numpy as np
from numba import njit, prange

from sapp import AnchorParameters
from sapp.acceleration import score_model, direct_scores, decode
from sapp.baselines import chamfer
from sapp.baselines._mpurge_map_numba import pair_score
from sapp.localization import candidate_grid
from .common import read_json, write_json, save_npz, model_at

SUBSETS = {k: list(combinations(range(6), k)) for k in (1, 2, 3, 6)}
LIMIT = 299792458.0 * 200e-9


def data(ctx, n, fold=None, limit=None):
    with np.load(ctx.data / "uwb/train.npz") as z:
        train = {k: z[k] for k in ("xy", "peaks", "burst")}
    with np.load(ctx.data / "uwb/survey.npz") as z:
        landmarks, folds = z["landmarks"], z["fold"]
    ids = landmarks[:n]
    if fold is None:
        with np.load(ctx.data / "uwb/test.npz") as z:
            query = {k: z[k][:limit] for k in ("xy", "peaks", "burst")}
    else:
        ids = ids[folds[ids] != fold]
        val = landmarks[np.arange(len(landmarks)) % 5 == fold]
        query = {k: train[k][val] for k in train}
    xy = train["xy"][ids]
    grid, radius = candidate_grid(xy, 0.1)
    return train["peaks"][ids], xy, query, grid, radius


def anchor_scores(ctx, n, method, h, sigma, fold=None, limit=None):
    peaks, xy, query, grid, radius = data(ctx, n, fold, limit)
    stage = "final" if fold is None else f"fold_{fold}"
    folder = ctx.output / "uwb"
    folder.mkdir(parents=True, exist_ok=True)
    scratch = folder / "scores.npy"
    scores = np.lib.format.open_memmap(
        scratch, mode="w+", dtype=np.float32, shape=(6, len(query["xy"]), len(grid))
    )
    for ai in range(6):
        path = folder / "models" / stage / f"n{n}_a{ai}.npz"
        base = ctx.fitted(
            path, xy, peaks[:, ai], AnchorParameters(range_limit_m=LIMIT, association_scale_m=0.32)
        )
        model = model_at(base, peaks[:, ai], method, h, sigma)
        check = ctx.output / "checks" / f"uwb_{stage}_{n}_{method}_{h}_{sigma}_{ai}.json"
        scores[ai] = (score_model if fold is None else direct_scores)(
            model, grid, query["peaks"][:, ai], check
        )
    scores.flush()
    return scores, xy, query, grid, radius


def release_scores(scores):
    path = scores.filename
    scores._mmap.close()
    from pathlib import Path

    Path(path).unlink()


def subset_predictions(scores, subsets, temperature, grid, radius, xy):
    result = []
    for subset in subsets:
        total = np.zeros(scores.shape[1:], dtype=np.float64)
        for ai in subset:
            total += scores[ai]
        result.append(decode(total, grid, temperature, radius, xy))
    return np.stack(result, axis=1)


def validate(ctx, sizes=(128, 256, 515)):
    """Select h, delay scale and temperature on two training-coordinate folds."""
    hs, scales, temps = [0.5, 1.0, 2.0], [0.16, 0.32, 0.64], [1.0, 2.0, 4.0]
    all_selected, all_rows = [], []
    for n in sizes:
        rows = []

        def evaluate(method, h, sigma, targets):
            records = []
            for fold in (0, 2):
                path = ctx.output / "uwb/validation" / f"{n}_{method}_{h}_{sigma}_{fold}.json"
                cached = read_json(path) if path.exists() else []
                missing = [
                    (k, t)
                    for k, t in targets
                    if not any(r["k"] == k and r["temp"] == t for r in cached)
                ]
                if missing:
                    scores, xy, q, grid, radius = anchor_scores(ctx, n, method, h, sigma, fold)
                    for k, t in missing:
                        pred = subset_predictions(scores, SUBSETS[k], t, grid, radius, xy)
                        error = np.linalg.norm(pred - q["xy"][:, None], axis=-1)
                        cached.append(
                            dict(
                                n=n,
                                method=method,
                                h=h,
                                sigma=sigma,
                                temp=t,
                                k=k,
                                fold=fold,
                                total=float(error.sum()),
                                count=error.size,
                            )
                        )
                    release_scores(scores)
                    write_json(path, cached)
                records.extend(r for r in cached if (r["k"], r["temp"]) in targets)
            return records

        for method, h, sigma in product(["residual", "sapp"], hs, scales):
            rows.extend(evaluate(method, h, sigma, list(product(SUBSETS, temps))))

        def candidates(method, k):
            grouped = {}
            for r in rows:
                if r["method"] == method and r["k"] == k:
                    grouped.setdefault((r["h"], r["sigma"], r["temp"]), []).append(r)
            return [
                dict(
                    n=n,
                    method=method,
                    k=k,
                    h=h,
                    sigma=s,
                    temp=t,
                    mean_m=sum(r["total"] for r in rs) / sum(r["count"] for r in rs),
                )
                for (h, s, t), rs in grouped.items()
            ]

        jobs = {}
        for method, k in product(["residual", "sapp"], SUBSETS):
            best = min(candidates(method, k), key=lambda r: r["mean_m"])
            for dimension, values in [("h", hs), ("sigma", scales), ("temp", temps)]:
                for factor in (0.5, 2.0):
                    value = best[dimension] * factor
                    if min(values) <= value <= max(values):
                        continue
                    config = dict(best, **{dimension: value})
                    jobs.setdefault((method, config["h"], config["sigma"]), set()).add(
                        (k, config["temp"])
                    )
        for (method, h, sigma), targets in sorted(jobs.items()):
            rows.extend(evaluate(method, h, sigma, sorted(targets)))
        all_selected.extend(
            min(candidates(m, k), key=lambda r: r["mean_m"])
            for m, k in product(["residual", "sapp"], SUBSETS)
        )
        all_rows.extend(rows)
        print(f"UWB validation: {n} survey coordinates complete", flush=True)
    write_json(ctx.output / "uwb/selection.json", dict(selected=all_selected, validation=all_rows))


def run(ctx, operating=False, limit=None):
    selected_path = ctx.output / "uwb/selection.json"
    selected = read_json(
        selected_path if selected_path.exists() else ctx.root / "configs/uwb_operating.json"
    )["selected"]
    if not operating:
        selected = [r for r in selected if r["n"] == 515 and r["k"] in (1, 6)]
    configs = sorted(set((r["n"], r["method"], r["h"], r["sigma"]) for r in selected))
    for n, method, h, sigma in configs:
        chosen = [
            r
            for r in selected
            if (r["n"], r["method"], r["h"], r["sigma"]) == (n, method, h, sigma)
        ]
        if all(
            (ctx.output / "uwb/operating" / f"n{n}_{method}_k{r['k']}.npz").exists() for r in chosen
        ):
            continue
        scores, xy, q, grid, radius = anchor_scores(ctx, n, method, h, sigma, limit=limit)
        for r in chosen:
            pred = subset_predictions(scores, SUBSETS[r["k"]], r["temp"], grid, radius, xy)
            save_npz(
                ctx.output / "uwb/operating" / f"n{n}_{method}_k{r['k']}.npz",
                truth=q["xy"],
                prediction=pred,
                burst=q["burst"],
                subsets=SUBSETS[r["k"]],
            )
            print(
                f"UWB/{n}/{method}/{r['k']}: mean {np.linalg.norm(pred - q['xy'][:, None], axis=-1).mean():.6f} m",
                flush=True,
            )
        release_scores(scores)


@njit(cache=True, parallel=True)
def mpurge_distances(query, survey, p, alpha):
    output = np.empty((len(query), len(survey)))
    for flat in prange(output.size):
        qi, si = flat // len(survey), flat % len(survey)
        q = query[qi][np.isfinite(query[qi])]
        s = survey[si][np.isfinite(survey[si])]
        output[qi, si] = pair_score(q, s, p, alpha)
    return output


def weighted_neighbors(distance, xy, neighbors, power=1):
    order = np.argsort(distance, axis=1, kind="stable")[:, :neighbors]
    values = np.take_along_axis(distance, order, axis=1)
    weights = np.where(
        np.isfinite(values), 1.0 / np.maximum(values, np.finfo(float).eps) ** power, 0.0
    )
    norm = weights.sum(axis=1)
    good = norm > 0
    weights[good] /= norm[good, None]
    pred = np.sum(weights[:, :, None] * xy[order], axis=1)
    pred[~good] = xy.mean(axis=0)
    return pred


def matching_baselines(ctx, limit=None):
    dest = ctx.output / "uwb/matching.npz"
    if dest.exists():
        return
    survey, xy, q, _, _ = data(ctx, 515, limit=limit)
    settings = read_json(ctx.root / "configs/uwb.json")["matching"]
    result = dict(truth=q["xy"], burst=q["burst"])
    for method, scope in product(["mpurge", "chamfer"], ["single", "joint"]):
        config = settings[method][scope]
        distances = []
        for ai in range(6):
            distance = (
                mpurge_distances(q["peaks"][:, ai], survey[:, ai], config["p"], config["alpha"])
                if method == "mpurge"
                else chamfer.distance(q["peaks"][:, ai], survey[:, ai])
            )
            distances.append(distance)
        power = config.get("power", 1)
        if scope == "joint":
            pred = weighted_neighbors(np.sum(distances, axis=0) / 6, xy, config["neighbors"], power)
        else:
            pred = np.stack(
                [weighted_neighbors(d, xy, config["neighbors"], power) for d in distances], axis=1
            )
        result[f"{scope}_{method}"] = pred
    save_npz(dest, **result)

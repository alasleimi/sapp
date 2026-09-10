"""Cluster-matched source fitting for the shared intensity-map control."""

import numpy as np
from scipy.spatial import cKDTree
from . import virtual_transmitters as vt
from sapp.kernels import range_surfaces


def unpack(peaks):
    return [r[np.isfinite(r)] for r in peaks]


def cluster_sources(xy, peaks, radius):
    rows = unpack(peaks)
    tree = cKDTree(xy)
    neighborhoods = sorted(set(tuple(sorted(i)) for i in tree.query_ball_point(xy, radius + 1e-9)))
    candidates = []
    tracks_n = 0
    fitted_n = 0
    for ids in neighborhoods:
        tracks = vt.matched_tracks(
            xy[list(ids)], [rows[i] for i in ids], cluster_gap=0.6, group_threshold=0.5
        )
        tracks_n += len(tracks)
        for points, ranges in tracks:
            if len(points) < 3 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
                continue
            source, mean = vt.source_fit(points, ranges)
            if source is not None and np.all(np.isfinite(source)):
                fitted_n += 1
                candidates.append(source)
    if not candidates:
        return np.empty((0, 3)), np.empty(0), dict(tracks=tracks_n, fitted=fitted_n, retained=0)
    candidates = np.asarray(candidates)
    curves = range_surfaces(xy, candidates).T
    available = np.where(np.isfinite(peaks), peaks, np.inf)
    chosen = []
    support = []
    active = np.ones(len(candidates), dtype=bool)
    required = max(5, int(np.ceil(0.08 * len(xy))))
    for _ in range(20):
        nearest = np.nanmin(np.abs(curves[:, :, None] - available[None, :, :]), axis=2)
        hits = nearest <= 0.18
        count = hits.sum(axis=1)
        smooth = np.exp(-0.5 * (nearest / 0.18) ** 2).sum(axis=1)
        score = count + smooth / (len(xy) + 1)
        score[~active] = -np.inf
        best = int(np.argmax(score))
        if not active[best] or count[best] < required:
            break
        chosen.append(best)
        support.append(int(count[best]))
        distance = np.sqrt(np.mean((curves - curves[best]) ** 2, axis=1))
        active[distance < 0.12] = False
        residuals = np.abs(available - curves[best, :, None])
        slot = np.nanargmin(residuals, axis=1)
        i = np.flatnonzero(hits[best])
        available[i, slot[i]] = np.inf
    return (
        candidates[chosen],
        np.asarray(support),
        dict(
            tracks=tracks_n, fitted=fitted_n, retained=len(chosen), neighborhoods=len(neighborhoods)
        ),
    )

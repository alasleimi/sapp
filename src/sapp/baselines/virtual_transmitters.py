"""Zayets & Steinbach (ICC 2018), Algorithms 1--4 and Table I.

Interpolation settings: radius 1 m, cluster gap .1 m, group threshold .5,
alpha .5, MCA tolerance 1 m. The coplanar survey parameterizes the unobservable
vertical sign by a nonnegative offset. Cluster grouping uses increasing-cost
pair edges with at most one cluster from each reference in a group. Smaller
clusters receive minimum-cost distinct slots in the mean full-sized cluster.
"""

from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial import cKDTree


def clusters(row, gap=0.1):
    row = np.sort(np.asarray(row, dtype=float))
    return [] if not len(row) else np.split(row, np.flatnonzero(np.diff(row) >= gap) + 1)


def matched_tracks(xy, sets, *, cluster_gap=0.1, group_threshold=0.5):
    nodes = [(i, c) for i, row in enumerate(sets) for c in clusters(row, cluster_gap)]
    n = len(nodes)
    if not n:
        return []
    owners = np.array([i for i, c in nodes])
    features = np.array([[c[0], len(c), c[-1] - c[0]] for i, c in nodes])
    cost = 0.5 * np.abs(features[:, None, :] - features[None, :, :]).sum(axis=2)
    a, b = np.triu_indices(n, 1)
    valid = (owners[a] != owners[b]) & (cost[a, b] <= group_threshold)
    a, b = a[valid], b[valid]
    order = np.argsort(cost[a, b], kind="stable")
    parent = np.arange(n)
    members = [{int(i)} for i in owners]

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for edge in order:
        ra, rb = root(a[edge]), root(b[edge])
        if ra != rb and members[ra].isdisjoint(members[rb]):
            parent[rb] = ra
            members[ra] |= members[rb]
    groups = {}
    for i, node in enumerate(nodes):
        groups.setdefault(root(i), []).append(node)
    tracks = []
    for group in groups.values():
        length = max(len(c) for i, c in group)
        target = np.mean([c for i, c in group if len(c) == length], axis=0)
        slots = [[] for _ in range(length)]
        for i, c in group:
            if len(c) == length:
                cols = np.arange(length)
            else:
                _, cols = linear_sum_assignment(np.abs(c[:, None] - target[None, :]))
            for value, col in zip(c, cols):
                slots[col].append((i, value))
        tracks.extend(
            (xy[[i for i, v in slot]], np.array([v for i, v in slot])) for slot in slots if slot
        )
    return tracks


def source_fit(xy, ranges):
    """Nonlinear range trilateration; sparse/rank-deficient tracks use the mean."""
    origin = xy.mean(axis=0)
    local = xy - origin
    if len(xy) < 3 or np.linalg.matrix_rank(local) < 2:
        return None, float(np.mean(ranges))
    design = np.column_stack((-2 * local, np.ones(len(local))))
    coef = np.linalg.lstsq(design, ranges * ranges - np.sum(local * local, axis=1), rcond=None)[0]
    initial = np.r_[coef[:2], np.sqrt(max(coef[2] - coef[:2] @ coef[:2], 1e-8))]

    def residual(p):
        return np.sqrt(np.sum((local - p[:2]) ** 2, axis=1) + p[2] ** 2) - ranges

    def jacobian(p):
        diff = p[:2] - local
        d = np.maximum(np.sqrt(np.sum(diff * diff, axis=1) + p[2] ** 2), 1e-12)
        return np.column_stack((diff, np.full(len(local), p[2]))) / d[:, None]

    fit = least_squares(
        residual,
        initial,
        jac=jacobian,
        bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]),
        max_nfev=100,
    )
    return np.r_[fit.x[:2] + origin, fit.x[2]], None


def build_map(reference, sets, step=0.1, *, cluster_gap=0.1, group_threshold=0.5):
    reference = np.asarray(reference)
    tree = cKDTree(reference)
    spacing = float(np.median(tree.query(reference, k=2)[0][:, 1]))
    lower = reference.min(axis=0) - spacing / 2
    upper = reference.max(axis=0) + spacing / 2
    x = np.arange(np.ceil(lower[0] / step) * step, upper[0] + 1e-8, step)
    y = np.arange(np.ceil(lower[1] / step) * step, upper[1] + 1e-8, step)
    grid = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
    distance = tree.query(grid)[0]
    grid = grid[(distance <= 0.82 * spacing) & (distance > 1e-7)]

    @lru_cache(maxsize=None)
    def neighborhood(indices):
        xy = reference[list(indices)]
        rows = [sets[i] for i in indices]
        return [
            source_fit(points, ranges)
            for points, ranges in matched_tracks(
                xy, rows, cluster_gap=cluster_gap, group_threshold=group_threshold
            )
        ]

    output = []
    for point in grid:
        indices = tuple(sorted(tree.query_ball_point(point, 1.0 + 1e-9)))
        values = []
        for source, mean in neighborhood(indices):
            value = (
                mean
                if source is None
                else np.sqrt(np.sum((source[:2] - point) ** 2) + source[2] ** 2)
            )
            if 0 <= value <= 25:
                values.append(value)
        output.append(np.sort(values))
    return np.vstack((reference, grid)), list(sets) + output


def localize(queries, xy, sets, tolerance=1.0):
    """Published MCA sum of squared coincidence scores, evaluated in batches.

    Identical maximum scores use the original map order, as the shared MCA
    implementation does; measured references precede interpolated references.
    """
    packed = np.full((len(sets), max(map(len, sets))), np.inf)
    for i, row in enumerate(sets):
        packed[i, : len(row)] = row
    predictions = []
    for query in queries:
        if not len(query):
            predictions.append(xy[0])
            continue
        nearest = np.min(np.abs(np.asarray(query)[None, :, None] - packed[:, None, :]), axis=2)
        scores = np.maximum(tolerance - nearest, 0.0) ** 2
        predictions.append(xy[int(np.argmax(scores.sum(axis=1)))])
    return np.asarray(predictions)

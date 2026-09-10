"""Symmetric Chamfer distance with inverse-distance weighted neighbors."""

import numpy as np


def distance(query, survey):
    out = np.empty((len(query), len(survey)))
    smask = np.isfinite(survey)
    for lo in range(0, len(query), 32):
        q = query[lo : lo + 32]
        qmask = np.isfinite(q)
        delta = np.abs(q[:, None, :, None] - survey[None, :, None, :])
        delta = np.where(np.isfinite(delta), delta, np.inf)
        left = np.where(qmask[:, None, :], delta.min(3), 0.0).sum(2) / np.maximum(
            qmask.sum(1)[:, None], 1
        )
        right = np.where(smask[None, :, :], delta.min(2), 0.0).sum(2) / np.maximum(
            smask.sum(1)[None, :], 1
        )
        val = 0.5 * (left + right)
        val[(qmask.sum(1)[:, None] == 0) | (smask.sum(1)[None, :] == 0)] = 150.0
        out[lo : lo + len(q)] = val
    return out


def predict(dist, xy, neighbors, power):
    order = np.argsort(dist, axis=1, kind="stable")[:, :neighbors]
    selected = np.take_along_axis(dist, order, axis=1)
    weights = 1.0 / np.maximum(selected, 1e-9) ** power
    zero = selected <= 1e-9
    rows = zero.any(1)
    weights[rows] = zero[rows]
    weights /= weights.sum(1, keepdims=True)
    return np.sum(weights[:, :, None] * xy[order], axis=1)

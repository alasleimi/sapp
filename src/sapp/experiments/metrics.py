"""Position errors and paired uncertainty estimates."""

import numpy as np


def metrics(errors):
    e = np.asarray(errors).ravel()
    if not len(e) or not np.isfinite(e).all():
        raise ValueError("Metrics require a nonempty finite error array")
    return dict(
        mean=float(e.mean()),
        median=float(np.median(e)),
        p90=float(np.quantile(e, 0.9)),
        p95=float(np.quantile(e, 0.95)),
        rmse=float(np.sqrt(np.mean(e * e))),
        maximum=float(e.max()),
        over2=float(np.mean(e > 2)),
        n=len(e),
    )


def crossed_bootstrap(values, draws=20000):
    a = np.asarray(values)
    if a.ndim == 3:
        a = a[None]
    if a.ndim != 4 or a.shape[1:3] != (9, 2):
        raise ValueError("Expected [room, nine layouts, two APs, query] errors")
    a = a.mean(axis=2)
    nr, _, nq = a.shape
    random = np.random.default_rng(2026090541)
    means = []
    for offset in range(0, draws, 250):
        batch = min(250, draws - offset)
        rooms = []
        for ri in range(nr):
            q = random.integers(0, nq, (batch, nq))
            groups = []
            for group in (np.arange(3), np.arange(3, 6), np.arange(6, 9)):
                layouts = random.choice(group, (batch, 3))
                block = a[ri][layouts]
                paired = np.take_along_axis(
                    block, np.broadcast_to(q[:, None, :], block.shape), axis=2
                )
                groups.append(paired.mean(axis=(1, 2)))
            rooms.append(np.stack(groups).mean(axis=0))
        means.extend(np.stack(rooms).mean(axis=0))
    return np.quantile(means, [0.025, 0.975])


def block_bootstrap(values, block=100):
    v = np.asarray(values)
    if v.ndim == 2:
        v = v.mean(axis=1)
    ids = np.arange(len(v)) // block
    sums = np.bincount(ids, weights=v)
    counts = np.bincount(ids)
    index = np.random.default_rng(5907).integers(0, len(sums), (4000, len(sums)))
    return np.quantile(sums[index].sum(1) / counts[index].sum(1), [0.025, 0.975])

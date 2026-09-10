from itertools import combinations
import numpy as np
from sapp.baselines.matching import size_unify, mpurge_scores
from sapp.baselines.chamfer import distance, predict
from sapp.baselines.virtual_transmitters import source_fit
from sapp.kernels import range_surfaces


def test_size_unification_matches_exhaustive_subsets():
    rng = np.random.default_rng(45)
    for _ in range(20):
        longer = np.sort(rng.uniform(0, 20, 7))
        short = np.sort(rng.uniform(0, 20, 4))
        a, b, _, _ = size_unify(longer, short)
        expected = min(np.abs(longer[list(ids)] - short).sum() for ids in combinations(range(7), 4))
        np.testing.assert_allclose(np.abs(a - b).sum(), expected, atol=1e-12)


def test_compiled_mpurge_agrees_with_equations():
    from sapp.baselines._matching_numba import pair_score

    rng = np.random.default_rng(37)
    for p in (1, 2, 3):
        for _ in range(15):
            q = np.sort(rng.uniform(0.2, 20, rng.integers(2, 25)))
            s = np.sort(rng.uniform(0.2, 20, rng.integers(2, 25)))
            expected, _ = mpurge_scores(q, [s], half_window_p=p, alpha=0.7)
            np.testing.assert_allclose(
                pair_score(q, s, p, 0.7), expected[0], atol=1e-11, rtol=1e-12
            )


def test_chamfer_is_symmetric_and_recovers_exact_reference():
    a = np.array([[1.0, 2.0, 5.0], [2.0, 4.0, np.nan]])
    b = np.array([[0.5, 3.0, 7.0], [1.0, 2.0, 5.0]])
    np.testing.assert_array_equal(distance(a, b), distance(b, a).T)
    xy = np.array([[0.0, 0.0], [5.0, 3.0]])
    np.testing.assert_array_equal(predict(distance(a[:1], b), xy, 2, 1), xy[1:])


def test_virtual_transmitter_fits_coplanar_range_surface():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.7]])
    source = np.array([[3.0, -2.0, 1.5]])
    ranges = range_surfaces(xy, source)[:, 0]
    fitted, fallback = source_fit(xy, ranges)
    assert fallback is None
    np.testing.assert_allclose(range_surfaces(xy, fitted[None])[:, 0], ranges, atol=1e-8)

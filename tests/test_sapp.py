from dataclasses import replace
import numpy as np
import pytest
from sapp import fit, localize, load_anchor_map, save_anchor_map
from sapp.kernels import _student_density
from sapp.experiments.common import read_json, unpack


@pytest.fixture(scope="module")
def maps():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    room = read_json(root / "data/nist/design.json")["rooms"][0]
    output = {}
    for ap in ("LA", "LB"):
        with np.load(root / "data/nist/receiver" / f"L_{ap}_empty_survey_standard_peaks.npz") as z:
            survey = unpack(z["ranges"][:, :9])
        output[ap] = fit(np.asarray(room["reference_xy_m"]), survey)
    return output


def test_fitting_matches_published_maps(root, maps):
    for ap, model in maps.items():
        expected = load_anchor_map(
            root / "reference/nist/models" / f"L_{ap}_standard_9_full_map.npz"
        )
        for key in ("anchors", "soft_counts", "global_counts", "background_responsibility"):
            np.testing.assert_array_equal(getattr(model, key), getattr(expected, key))


@pytest.mark.parametrize("ap,layout,index", [("LA", "L05", 81), ("LB", "L02", 44)])
def test_extreme_mode_and_multimodal_case(root, maps, ap, layout, index):
    with np.load(root / "data/nist/receiver" / f"L_{ap}_{layout}_standard_peaks.npz") as z:
        query = unpack(z["ranges"][index : index + 1, :9])
    row = 100 * (int(layout[1:]) - 1) + index
    with np.load(root / "reference/nist/predictions" / f"L_{ap}.npz") as z:
        expected_mode = z["mode"][row]
        expected_median = z["sapp"][row]
    model = maps[ap]
    np.testing.assert_allclose(localize(query, model)[0], expected_mode, atol=1e-10, rtol=0)
    median = replace(model, parameters=replace(model.parameters, decode="posterior_median"))
    np.testing.assert_allclose(localize(query, median)[0], expected_median, atol=1e-10, rtol=0)


def test_map_serialization_and_query_permutation(maps, tmp_path):
    model = maps["LA"]
    path = tmp_path / "map.npz"
    save_anchor_map(path, model)
    loaded = load_anchor_map(path)
    query = np.array([2.0, 5.0, 7.1, 9.2])
    np.testing.assert_array_equal(localize([query], model), localize([query[::-1]], loaded))
    np.testing.assert_array_equal(localize([np.empty(0)], model), model.reference_xy.mean(0)[None])


def test_delay_kernels_integrate_to_one_at_boundaries():
    axis = np.linspace(0, 25, 50001)
    density = _student_density(
        axis[:, None], np.array([0.0, 0.1, 12.0, 24.9, 25.0])[None], 0.16, 25.0
    )
    np.testing.assert_allclose(np.trapezoid(density, axis, axis=0), 1.0, atol=4e-7, rtol=0)

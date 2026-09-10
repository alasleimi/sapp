"""Small, self-contained example from the L-room benchmark."""

from importlib.resources import files
import numpy as np
from . import fit, localize, AnchorParameters


def run():
    with files("sapp").joinpath("sample.npz").open("rb") as handle, np.load(handle) as z:
        xy, survey, queries, truth = (z[k] for k in ("survey_xy", "survey", "query", "truth"))

    def rows(a):
        return [np.sort(r[np.isfinite(r)]) for r in a]

    model = fit(xy, rows(survey), AnchorParameters(decode="posterior_median"))
    prediction = localize(rows(queries), model)
    error = np.linalg.norm(prediction - truth, axis=1)
    print(f"Fitted {len(model.anchors)} sources from {len(xy)} survey positions.")
    print(
        f"Localized {len(error)} queries: mean {error.mean():.3f} m; P90 {np.quantile(error, 0.9):.3f} m."
    )

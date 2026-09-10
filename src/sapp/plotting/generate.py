"""Generate every manuscript figure from numerical inputs."""

from sapp.experiments.reporting import Results
from sapp.experiments.common import write_json
from .style import configure
from .geometry import method, rooms, surface
from .failure import failure
from .results import FIGURES

GENERATORS = {
    "method": method,
    "rooms": rooms,
    "surface_validation": surface,
    "failure_diagnostic": failure,
    **FIGURES,
}


def generate(root, source, output, only=None):
    configure()
    results = Results(root, source)
    chosen = list(GENERATORS) if only is None else only
    unknown = set(chosen) - set(GENERATORS)
    if unknown:
        raise ValueError(f"Unknown figures: {sorted(unknown)}")
    for name in chosen:
        GENERATORS[name](results, output)
        print(f"Generated figures/{name}.pdf", flush=True)
    write_json(
        output / "figure_manifest.json",
        dict(
            figures=chosen,
            source="reference" if source == root / "reference" else "computed predictions",
        ),
    )

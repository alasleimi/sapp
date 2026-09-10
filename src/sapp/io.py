"""Portable NumPy serialization for fitted maps."""

import numpy as np
from pathlib import Path
from .types import AnchorMap, AnchorParameters


def save_anchor_map(path, model: AnchorMap) -> None:
    """Persist a fitted survey map without simulator-side information."""

    from dataclasses import asdict
    import json

    payload = dict(
        anchors=model.anchors,
        reference_xy=model.reference_xy,
        soft_counts=model.soft_counts,
        global_counts=model.global_counts,
        clutter_rate=np.asarray(model.clutter_rate),
        parameters=np.asarray(json.dumps(asdict(model.parameters))),
        discovery_support=model.discovery_support,
        discovery_residual_m=model.discovery_residual_m,
    )
    if model.background_delay is not None:
        payload.update(
            background_delay=model.background_delay,
            background_mask=model.background_mask,
            background_responsibility=model.background_responsibility,
        )
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_name(path.name + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def load_anchor_map(path) -> AnchorMap:
    """Load a map written by :func:`save_anchor_map`."""

    import json

    with np.load(path, allow_pickle=False) as payload:
        parameters = AnchorParameters(**json.loads(str(payload["parameters"])))
        names = set(payload.files)
        return AnchorMap(
            anchors=np.asarray(payload["anchors"], dtype=np.float64),
            reference_xy=np.asarray(payload["reference_xy"], dtype=np.float64),
            soft_counts=np.asarray(payload["soft_counts"], dtype=np.float64),
            global_counts=np.asarray(payload["global_counts"], dtype=np.float64),
            clutter_rate=float(payload["clutter_rate"]),
            parameters=parameters,
            discovery_support=np.asarray(payload["discovery_support"], dtype=np.int64),
            discovery_residual_m=np.asarray(
                payload["discovery_residual_m"],
                dtype=np.float64,
            ),
            background_delay=(
                np.asarray(payload["background_delay"], dtype=np.float64)
                if "background_delay" in names
                else None
            ),
            background_mask=(
                np.asarray(payload["background_mask"], dtype=bool)
                if "background_mask" in names
                else None
            ),
            background_responsibility=(
                np.asarray(payload["background_responsibility"], dtype=np.float64)
                if "background_responsibility" in names
                else None
            ),
        )

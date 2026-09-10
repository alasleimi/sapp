"""Data locations, atomic outputs and shared benchmark conventions."""

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from sapp import AnchorParameters, fit, load_anchor_map, save_anchor_map
from sapp.fields import survey_kernel_bandwidth
from sapp.fitting import calibrate_background, soft_association_counts
from sapp.types import AnchorMap


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=lambda x: np.asarray(x).tolist()) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def save_torch(path, value):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(value, temporary)
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def unpack(values):
    return [np.sort(row[np.isfinite(row)]) for row in values]


def pack(values, cap=9):
    out = np.full((len(values), cap), np.nan)
    for i, row in enumerate(values):
        out[i, : min(cap, len(row))] = row[:cap]
    return out


def farthest_indices(xy, count):
    selected = [int(np.argmin(np.linalg.norm(xy - xy.mean(0), axis=1)))]
    distance = np.linalg.norm(xy - xy[selected[0]], axis=1)
    while len(selected) < min(count, len(xy)):
        distance[selected] = -np.inf
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(distance, np.linalg.norm(xy - xy[index], axis=1))
    return np.sort(selected)


def residual_model(base, h=None, sigma=None):
    p = replace(base.parameters, source_count=0)
    if h is not None:
        p = replace(p, field_length_m=h * survey_kernel_bandwidth(base))
    if sigma is not None:
        p = replace(p, association_scale_m=sigma)
    return AnchorMap(
        np.empty((0, 3)),
        base.reference_xy,
        np.empty((len(base.reference_xy), 0)),
        np.empty(0),
        base.clutter_rate,
        p,
        np.empty(0),
        np.empty(0),
        base.background_delay,
        base.background_mask,
        base.background_mask.astype(float),
    )


def model_at(base, peaks, method, h, sigma):
    if method == "residual":
        return residual_model(base, h, sigma)
    p = replace(
        base.parameters, field_length_m=h * survey_kernel_bandwidth(base), association_scale_m=sigma
    )
    rows = unpack(peaks)
    counts, glob, clutter = soft_association_counts(base.reference_xy, rows, base.anchors, p)
    model = AnchorMap(
        base.anchors,
        base.reference_xy,
        counts,
        glob,
        clutter,
        p,
        base.discovery_support,
        base.discovery_residual_m,
    )
    calibrate_background(model, rows)
    return model


@dataclass(frozen=True)
class Context:
    root: Path
    output: Path

    @property
    def data(self):
        return self.root / "data"

    @property
    def rooms(self):
        return read_json(self.data / "nist/design.json")["rooms"]

    @property
    def parameters(self):
        return AnchorParameters(**read_json(self.root / "configs/receiver.json")["sapp"])

    def observation_path(self, room, ap, layout, condition="standard", native=False):
        folder = "native" if native else "receiver"
        suffix = "" if native else f"_{condition}_peaks"
        path = self.data / f"nist/{folder}/{room}_{ap}_{layout}{suffix}.npz"
        regenerated = self.output / f"inputs/nist/{folder}/{path.name}"
        if regenerated.exists():
            path = regenerated
        return path

    def observations(self, room, ap, layout, cap=9, condition="standard", native=False):
        with np.load(self.observation_path(room, ap, layout, condition, native)) as z:
            return np.sort(z["ranges"][:, :cap], axis=1)

    def fitted(self, path, xy, peaks, parameters=None):
        if path.exists():
            return load_anchor_map(path)
        model = fit(xy, unpack(peaks), parameters or self.parameters)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_anchor_map(path, model)
        return model

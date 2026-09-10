"""Read predictions into consistently ordered error arrays for figures and tables."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .common import read_json
from .metrics import metrics


def errors(path, key="prediction"):
    with np.load(path) as z:
        prediction, truth = (z[key], z["truth"])
    if prediction.ndim == 3:
        truth = truth[:, None]
    return np.linalg.norm(prediction - truth, axis=-1)


@dataclass
class Results:
    root: Path
    source: Path

    @property
    def rooms(self):
        return read_json(self.root / "data/nist/design.json")["rooms"]

    def cube(self, method, scope="nist"):
        output = np.empty((3, 9, 2, 100))
        for ri, room in enumerate(self.rooms):
            for ai, ap in enumerate(room["access_points_m"]):
                stem = room["room_id"] + "_" + ap
                if method in ("sapp", "mode", "residual", "mpurge", "mca", "chamfer"):
                    value = errors(self.source / scope / "predictions" / f"{stem}.npz", method)
                elif method in ("optimized_residual", "swap"):
                    value = errors(
                        self.source / "nist/optimized" / f"{stem}.npz",
                        "residual" if method == "optimized_residual" else "swap",
                    )
                elif method.startswith("vt_") or method == "direct_mca":
                    value = errors(
                        self.source / "nist/vt" / f"{stem}.npz", method.removeprefix("vt_")
                    )
                else:
                    value = errors(self.source / "nist/components" / f"{stem}.npz", method)
                output[ri, :, ai] = value.reshape(9, 100)
        return output

    def neural(self, method, scope="nist"):
        if scope == "uwb":
            return np.stack(
                [
                    errors(
                        self.source
                        / "networks/final"
                        / method
                        / "uwb"
                        / str(seed)
                        / "predictions.npz"
                    )
                    for seed in (9101, 9102, 9103)
                ]
            )
        result = np.empty((3, 3, 9, 2, 100))
        cnn = method in ("cnn", "cnn100")
        folder = "common100" if method == "cnn100" else "common"
        for si, seed in enumerate((2601, 2602, 2603) if cnn else (9101, 9102, 9103)):
            for ri, room in enumerate(self.rooms):
                for ai, ap in enumerate(room["access_points_m"]):
                    stem = room["room_id"] + "_" + ap
                    if not cnn:
                        value = errors(
                            self.source
                            / "networks/final"
                            / method
                            / stem
                            / str(seed)
                            / "predictions.npz"
                        )
                    else:
                        value = errors(self.source / "cnn" / folder / f"{stem}_{seed}.npz", "cnn")
                    result[si, ri, :, ai] = value.reshape(9, 100)
        return result

    def cnn_budget(self, condition, epoch):
        result = np.empty((3, 3, 9, 2, 100))
        for si, seed in enumerate((7201, 7202, 7203)):
            for ri, room in enumerate(self.rooms):
                stem = f"{room['room_id']}_{seed}"
                folder = self.source / "cnn/budget" / condition / "final"
                path = folder / f"{stem}_e{epoch}.npz"
                with np.load(path) as z:
                    value = np.linalg.norm(z["prediction"] - z["truth"], axis=1)
                    result[si, ri, z["layout"], z["ap"], z["query"]] = value
        return result

    def measured(self, method, scope="joint"):
        if method in ("sapp", "residual"):
            value = errors(
                self.source
                / "uwb/operating"
                / f"n515_{method}_k{(6 if scope == 'joint' else 1)}.npz"
            )
            return value[:, 0] if scope == "joint" else value
        return errors(self.source / "uwb/matching.npz", scope + "_" + method)

    def sensitivity(self, method, condition):
        room = self.rooms[0]
        result = np.empty((9, 2, 100))
        for ai, ap in enumerate(room["access_points_m"]):
            stem = room["room_id"] + "_" + ap
            value = errors(self.source / "nist/sensitivity" / f"{stem}_{condition}.npz", method)
            result[:, ai] = value.reshape(9, 100)
        return result


def independent_metrics(error_arrays):
    rows = [metrics(e) for e in error_arrays]
    return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}

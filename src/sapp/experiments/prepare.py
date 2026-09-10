"""Regenerate NIST delay observations from the bundled native channels."""

from dataclasses import dataclass, replace
import gzip
import json
import numpy as np
from sapp.receiver import PeakFrontend, extract_delay_peaks
from .common import read_json, write_json, save_npz, pack


@dataclass
class ChannelFrame:
    delay_s: np.ndarray
    gain_db: np.ndarray
    phase_rad: np.ndarray
    aoa_el_deg: np.ndarray
    aoa_az_deg: np.ndarray

    @property
    def path_count(self):
        return len(self.delay_s)


def channel_frames(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    record = next(
        r for r in records if (r["TX"], r["RX"], r["PAA_TX"], r["PAA_RX"]) == (0, 1, 0, 0)
    )
    names = ("Delay", "Gain", "Phase", "AODEL", "AODAZ", "AOAEL", "AOAAZ")
    for i in range(len(record["Delay"])):
        values = [np.asarray(record[k][i], dtype=float) for k in names]
        if len({v.size for v in values}) != 1:
            raise ValueError(f"Inconsistent path counts at frame {i}")
        valid = np.logical_and.reduce([np.isfinite(v) for v in values])
        values = [v[valid] for v in values]
        yield ChannelFrame(values[0], values[1], values[2], values[5], values[6])


def prepare(ctx):
    settings = read_json(ctx.root / "configs/receiver.json")
    base = PeakFrontend(**settings["frontend"])
    count = 0
    for room in ctx.rooms:
        rid = room["room_id"]
        for ap in room["access_points_m"]:
            for layout in ["empty_survey"] + [r["layout_id"] for r in room["layouts"]]:
                frames = list(
                    channel_frames(
                        ctx.data / "nist/channels" / rid / f"{ap}_{layout}" / "qdOutput.json.gz"
                    )
                )
                native, gains, native_counts = [], [], []
                for frame in frames:
                    ranges = frame.delay_s * 3e8
                    valid = np.flatnonzero((ranges > 0) & (ranges <= 25))
                    ids = valid[np.argsort(-frame.gain_db[valid], kind="stable")[:9]]
                    ids = ids[np.argsort(ranges[ids], kind="stable")]
                    native.append(ranges[ids])
                    gains.append(frame.gain_db[ids])
                    native_counts.append(len(valid))
                values = pack(native)
                with np.load(ctx.data / "nist/native" / f"{rid}_{ap}_{layout}.npz") as reference:
                    np.testing.assert_array_equal(values, reference["ranges"])
                save_npz(
                    ctx.output / "inputs/nist/native" / f"{rid}_{ap}_{layout}.npz",
                    ranges=values,
                    gain_db=pack(gains),
                    eligible_path_count=native_counts,
                )
                for condition in ("standard", "snr25", "bandwidth1", "bandwidth1_snr25"):
                    token = f"{rid}_{ap}_{layout}_{condition}"
                    if token not in settings["receiver_seeds"]:
                        continue
                    config = replace(
                        base,
                        bandwidth_hz=1e9 if "bandwidth1" in condition else 2e9,
                        snr_db=25 if "snr25" in condition else 40,
                    )
                    seeds = settings["receiver_seeds"][token]
                    rows = [
                        extract_delay_peaks(
                            f, rng=np.random.default_rng(seed), config=config, power_order=True
                        )
                        for f, seed in zip(frames, seeds, strict=True)
                    ]
                    values = pack(rows, 24)
                    filename = f"{token}_peaks.npz"
                    with np.load(ctx.data / "nist/receiver" / filename) as reference:
                        np.testing.assert_array_equal(values, reference["ranges"])
                    save_npz(ctx.output / "inputs/nist/receiver" / filename, ranges=values)
                    count += 1
                print(f"Regenerated observations/{rid}/{ap}/{layout}", flush=True)
    write_json(
        ctx.output / "inputs/preparation.json",
        dict(receiver_files=count, native_files=60, comparison_to_bundled_arrays="exact"),
    )

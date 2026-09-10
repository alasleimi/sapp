"""CNN training with seven layouts and the additional AP survey regime."""

import json
import numpy as np
import torch
from sapp.baselines.cnn import encode
from .common import write_json as write, read_json, save_npz, sha
from .cnn import train


def observation(ctx, room, ap, layout):
    path = ctx.data / "cnn/observations" / room / f"{ap}_{layout}.npz"
    assert path.exists(), f"Missing native observations: {path}"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def prepare_dataset(ctx, condition, room):
    rid = room["room_id"]
    dest = ctx.output / "cnn/budget/inputs" / condition / f"{rid}.npz"
    if dest.exists():
        return dest
    xy = np.asarray(room["reference_xy_m"])
    context_ids = np.asarray(room["context_indices"])
    context = np.column_stack((xy[context_ids], np.full(20, room["receiver_height_m"])))
    origin = np.r_[xy.min(axis=0), 0.0]
    scale = np.r_[np.ptp(xy, axis=0), max(p[2] for p in room["access_points_m"].values())]
    context = (context - origin) / scale
    aps = room["target_aps"] if condition == "layout_rich" else list(room["access_points_m"])
    raw = []
    targets = []
    contexts = []
    ap_ids = []
    layout_ids = []
    reference_ids = []
    for ap in aps:
        ap_context = (np.asarray(room["access_points_m"][ap]) - origin) / scale
        for layout in room["layouts"]:
            if layout["role"] != "train":
                continue
            data = observation(ctx, rid, ap, layout["layout_id"])
            ids = np.arange(len(data["peaks"]))
            if condition == "paper_diversity":
                ids = ids[np.isin(data["reference_indices"], context_ids)]
            if condition == "paper_diversity":
                assert len(ids) == 20
            raw.extend(data["peaks"][ids])
            targets.extend(data["xy"][ids])
            contexts.extend([ap_context] * len(ids))
            ap_ids.extend([ap] * len(ids))
            layout_ids.extend([layout["layout_id"]] * len(ids))
            reference_ids.extend(data["reference_indices"][ids])
    raw = np.asarray(raw)
    targets = np.asarray(targets, dtype=np.float32)
    lo = float(np.nanmin(raw))
    hi = float(np.nanmax(raw))
    x = np.asarray(
        [encode(row, context, ap, lo, hi) for row, ap in zip(raw, contexts, strict=True)]
    )
    vx = []
    vy = []
    for layout in room["layouts"]:
        if layout["role"] != "validation":
            continue
        for ap in room["target_aps"]:
            data = observation(ctx, rid, ap, layout["layout_id"])
            ap_context = (np.asarray(room["access_points_m"][ap]) - origin) / scale
            vx.extend(encode(row, context, ap_context, lo, hi) for row in data["peaks"])
            vy.extend(data["xy"])
    vx = np.asarray(vx)
    vy = np.asarray(vy, dtype=np.float32)
    assert len(vx) == 180
    original = next(
        r
        for r in json.loads((ctx.data / "nist/design.json").read_text())["rooms"]
        if r["room_id"] == rid
    )
    tx = []
    ty = []
    test_ap = []
    test_layout = []
    query_index = []
    for li, layout in enumerate(original["layouts"]):
        for ai, ap in enumerate(original["access_points_m"]):
            path = ctx.observation_path(rid, ap, layout["layout_id"])
            assert path.exists(), path
            with np.load(path) as z:
                peaks = z["ranges"][:, :9]
            ap_context = (np.asarray(room["access_points_m"][ap]) - origin) / scale
            tx.extend(encode(row, context, ap_context, lo, hi) for row in peaks)
            ty.extend(original["query_xy_m"])
            test_ap.extend([ai] * len(peaks))
            test_layout.extend([li] * len(peaks))
            query_index.extend(range(len(peaks)))
    tx = np.asarray(tx)
    ty = np.asarray(ty, dtype=np.float64)
    assert len(tx) == 1800
    assert len(x) == (8400 if condition == "paper_diversity" else len(xy) * 14)
    assert (
        np.min(np.linalg.norm(np.asarray(room["validation_xy_m"])[:, None] - xy[None], axis=2))
        > 0.035
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        x=x,
        y=targets,
        validation_x=vx,
        validation_y=vy,
        test_x=tx,
        test_y=ty,
        training_ap=ap_ids,
        training_layout=layout_ids,
        training_reference=reference_ids,
        test_ap=test_ap,
        test_layout=test_layout,
        test_query=query_index,
    )
    write(
        dest.with_suffix(".json"),
        dict(
            condition=condition,
            room=rid,
            raw_training_examples=len(x),
            validation_examples=len(vx),
            test_examples=len(tx),
            context=context,
            coordinate_origin=origin,
            coordinate_scale=scale,
            delay_min=lo,
            delay_max=hi,
            sha256=sha(dest),
        ),
    )
    return dest


def run(ctx, tune=False, seeds=(7201, 7202, 7203)):
    design = read_json(ctx.data / "cnn/design.json")["rooms"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for condition in ("layout_rich", "paper_diversity"):
        files = {r["room_id"]: prepare_dataset(ctx, condition, r) for r in design}
        epochs = 300 if condition == "layout_rich" else 150
        if tune:
            histories = []
            for room in design:
                rid = room["room_id"]
                with np.load(files[rid]) as z:
                    _, history = train(
                        z["x"],
                        z["y"],
                        7101,
                        300,
                        ctx.output / "cnn/budget" / condition / "validation" / f"{rid}.pt",
                        (z["validation_x"], z["validation_y"]),
                        device=device,
                    )
                histories.append(history)
            scores = {
                e: float(
                    np.mean(
                        [np.mean([r["validation_mean"] for r in h[e - 10 : e]]) for h in histories]
                    )
                )
                for e in (100, 150, 200, 250, 300)
            }
            epochs = min(scores, key=scores.get)
            write(
                ctx.output / "cnn/budget" / condition / "selection.json",
                dict(epochs=epochs, candidates=scores),
            )
        for room in design:
            rid = room["room_id"]
            for seed in seeds:
                path = ctx.output / "cnn/budget" / condition / "final" / f"{rid}_{seed}.pt"
                with np.load(files[rid]) as z:
                    model, _ = train(
                        z["x"], z["y"], seed, epochs, path, device=device, snapshot_epochs=(100,)
                    )
                    tensor = torch.as_tensor(z["test_x"], device=device)
                    for duration in (100, epochs):
                        weights = (
                            path
                            if duration == epochs
                            else path.with_name(f"{path.stem}_epoch100.pt")
                        )
                        model.load_state_dict(
                            torch.load(weights, map_location=device, weights_only=True)
                        )
                        model.eval()
                        with torch.no_grad():
                            pred = (
                                torch.cat([model(batch) for batch in tensor.split(512)])
                                .cpu()
                                .numpy()
                            )
                        save_npz(
                            path.with_name(f"{path.stem}_e{duration}.npz"),
                            prediction=pred,
                            truth=z["test_y"],
                            ap=z["test_ap"],
                            layout=z["test_layout"],
                            query=z["test_query"],
                        )

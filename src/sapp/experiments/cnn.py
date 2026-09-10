"""Common-survey CNN encoding, coordinate validation and independent refits."""

import hashlib
import time
import numpy as np
import torch
from torch import nn
from sapp.baselines.cnn import PaperCNN, encode
from .common import read_json, write_json, save_npz, save_torch


def random_generator(*parts):
    prefix = ("majdi-mpurge-map-native-nist-qd-protocol-v1", "sapp-revision-fresh")
    seed = int.from_bytes(
        hashlib.sha256(":".join(map(str, (*prefix, *parts))).encode()).digest()[:8], "little"
    )
    return np.random.default_rng(seed)


def padded(row):
    out = np.full(9, np.nan)
    out[: min(9, len(row))] = row[:9]
    return out


def dataset(ctx, room):
    xy = np.asarray(room["reference_xy_m"])
    rid = room["room_id"]
    ids = [int(np.argmin(np.linalg.norm(xy - xy.mean(0), axis=1)))]
    distance = np.linalg.norm(xy - xy[ids[0]], axis=1)
    while len(ids) < 20:
        distance[ids] = -np.inf
        ids.append(int(np.argmax(distance)))
        distance = np.minimum(distance, np.linalg.norm(xy - xy[ids[-1]], axis=1))
    origin = np.r_[xy.min(0), 0.0]
    scale = np.r_[np.ptp(xy, axis=0), max(a[2] for a in room["access_points_m"].values())]
    context = (np.column_stack((xy[ids], np.full(20, room["receiver_height_m"]))) - origin) / scale
    surveys = {}
    for ap in room["access_points_m"]:
        with np.load(ctx.observation_path(rid, ap, "empty_survey")) as z:
            surveys[ap] = [row[np.isfinite(row)][:9] for row in z["ranges"]]
    delays = np.concatenate([row for rows in surveys.values() for row in rows])
    lo, hi = float(delays.min()), float(delays.max())
    inputs, targets, coordinates = [], [], []
    for ap, coordinate in room["access_points_m"].items():
        for i, row in enumerate(surveys[ap]):
            for pattern in range(7):
                rng = random_generator("cnn_training", rid, ap, i, pattern)
                observed = row.copy()
                remove = (0, 1, 1, 2, 2, 3, 3)[pattern]
                if remove and len(observed) > 2:
                    observed = np.delete(
                        observed,
                        rng.choice(len(observed), min(remove, len(observed) - 2), replace=False),
                    )
                if pattern in (4, 6):
                    observed += rng.normal(0, 0.015, len(observed))
                inputs.append(
                    encode(
                        padded(observed), context, (np.asarray(coordinate) - origin) / scale, lo, hi
                    )
                )
                targets.append(xy[i])
                coordinates.append(i)
    encoding = dict(
        context=context,
        coordinate_origin=origin,
        coordinate_scale=scale,
        delay_min=lo,
        delay_max=hi,
    )
    return (
        np.asarray(inputs),
        np.asarray(targets, dtype=np.float32),
        np.asarray(coordinates),
        encoding,
    )


def train(x, y, seed, epochs, path, validation=None, device="cpu", snapshot_epochs=()):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    model = PaperCNN().to(device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return model, read_json(path.with_suffix(".json"))["history"]
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        **({"capturable": True, "foreach": False} if device == "cuda" else {}),
    )
    generator = torch.Generator().manual_seed(seed + 17)
    checkpoint = path.with_name(f"{path.stem}_checkpoint.pt")
    history = []
    elapsed = 0.0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if (state["seed"], state["epochs"], state["device"]) != (seed, epochs, device):
            raise ValueError(f"Checkpoint settings differ: {checkpoint}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["shuffle"].cpu())
        history, elapsed = state["history"], state["seconds"]
    x, y = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
    captured = None
    if device == "cuda":
        from sapp.baselines._cnn_cuda import CapturedStep

        captured = CapturedStep(model, optimizer)
    start = time.perf_counter()
    for epoch in range(len(history), epochs):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        total = 0.0
        if captured is not None:
            total = captured.epoch(x, y, order) * len(x)
        else:
            for lo in range(0, len(x), 10):
                ids = order[lo : lo + 10]
                loss = ((model(x[ids]) - y[ids]).square().sum(1).mean() + 1e-12).sqrt()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(ids)
        row = dict(epoch=epoch + 1, training_batch_rmse=total / len(x))
        if validation is not None:
            model.eval()
            vx, vy = (torch.as_tensor(v, device=device) for v in validation)
            with torch.no_grad():
                row["validation_mean"] = float(
                    torch.linalg.vector_norm(model(vx) - vy, dim=1).mean()
                )
        history.append(row)
        if epoch + 1 in snapshot_epochs:
            save_torch(path.with_name(f"{path.stem}_epoch{epoch + 1}.pt"), model.state_dict())
        if (epoch + 1) % 25 == 0 or epoch + 1 == epochs:
            save_torch(
                checkpoint,
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    shuffle=generator.get_state(),
                    history=history,
                    seed=seed,
                    epochs=epochs,
                    device=device,
                    seconds=elapsed + time.perf_counter() - start,
                ),
            )
        if (epoch + 1) % 50 == 0:
            print(f"CNN {path.stem}: epoch {epoch + 1}/{epochs}", flush=True)
    save_torch(path, model.state_dict())
    write_json(
        path.with_suffix(".json"),
        dict(
            seed=seed, epochs=epochs, history=history, seconds=elapsed + time.perf_counter() - start
        ),
    )
    return model, history


def _validation_room(task):
    ctx, room = task
    x, y, coordinate, _ = dataset(ctx, room)
    n = len(room["reference_xy_m"])
    ids = np.sort(np.random.default_rng(2701).choice(n, max(1, int(round(0.2 * n))), replace=False))
    valid = np.isin(coordinate, ids)
    _, history = train(
        x[~valid],
        y[~valid],
        2601,
        300,
        ctx.output / "cnn/validation" / f"{room['room_id']}.pt",
        (x[valid], y[valid]),
    )
    return history


def _fit_room(task):
    ctx, room, epochs, seeds = task
    rid = room["room_id"]
    x, y, _, encoding = dataset(ctx, room)
    for seed in seeds:
        weights = ctx.output / "cnn/common" / f"{rid}_{seed}.pt"
        model, _ = train(x, y, seed, epochs, weights, snapshot_epochs=(100,))
        early = weights.with_name(f"{weights.stem}_epoch100.pt")
        if not early.exists():
            train(x, y, seed, 100, early)
        model.eval()
        for ap, coordinate in room["access_points_m"].items():
            values = []
            for layout in room["layouts"]:
                with np.load(ctx.observation_path(rid, ap, layout["layout_id"])) as z:
                    rows = z["ranges"][:, :9]
                ap_context = (np.asarray(coordinate) - encoding["coordinate_origin"]) / encoding[
                    "coordinate_scale"
                ]
                values.extend(
                    encode(
                        row,
                        encoding["context"],
                        ap_context,
                        encoding["delay_min"],
                        encoding["delay_max"],
                    )
                    for row in rows
                )
            for duration, state in ((100, early), (epochs, weights)):
                model.load_state_dict(torch.load(state, weights_only=True))
                with torch.no_grad():
                    pred = model(torch.as_tensor(np.asarray(values))).numpy()
                folder = "common" if duration == epochs else "common100"
                save_npz(
                    ctx.output / "cnn" / folder / f"{rid}_{ap}_{seed}.npz",
                    cnn=pred,
                    truth=np.tile(room["query_xy_m"], (9, 1)),
                )


def run(ctx, tune=False, seeds=(2601, 2602, 2603), room_filter=None, workers=3):
    from concurrent.futures import ProcessPoolExecutor

    rooms = [r for r in ctx.rooms if room_filter is None or r["room_id"] == room_filter]
    epochs = 250
    with ProcessPoolExecutor(max_workers=min(workers, len(rooms))) as pool:
        if tune:
            histories = list(pool.map(_validation_room, [(ctx, r) for r in rooms]))
            scores = {
                e: float(
                    np.mean(
                        [np.mean([r["validation_mean"] for r in h[e - 10 : e]]) for h in histories]
                    )
                )
                for e in (100, 150, 200, 250, 300)
            }
            epochs = min(scores, key=scores.get)
            write_json(
                ctx.output / "cnn/selection.json",
                dict(epochs=epochs, validation_mean_scores=scores),
            )
        list(pool.map(_fit_room, [(ctx, r, epochs, seeds) for r in rooms]))

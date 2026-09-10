"""Training of P-NN, L-SwiGLU and the PDP Transformer."""

import math
import time
import json
import numpy as np
import torch
from torch.nn import functional as F
from sapp.baselines.networks import MODELS, Encoder
from .common import read_json, write_json as write, sha, save_torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAPS = {"pnn": 1000, "sst": 2000, "cao": 1000}
CONFIGS = {
    "pnn": [
        dict(id=f"f{f}_lr{lr:g}", features=f, lr=lr, augment=False)
        for f in [9, 24]
        for lr in [1e-3, 3e-4]
    ],
    "sst": [
        dict(id=f"{s}_aug{int(a)}", size=s, lr=2e-3, augment=a)
        for s in ["small", "medium"]
        for a in [False, True]
    ],
    "cao": (
        [dict(id=f"lr{lr:g}", lr=lr, augment=False) for lr in [1e-3, 3e-4]]
        + [
            dict(id=f"global_lr{lr:g}", lr=lr, augment=False, power_scaling="global")
            for lr in [1e-3, 3e-4]
        ]
    ),
}


def log(**values):
    print(values, flush=True)


def tensor(x):
    return tuple(torch.as_tensor(v, device=DEVICE) for v in x)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def minibatches(order, batch):
    chunks = list(order.split(batch))
    if len(chunks) > 1 and len(chunks[-1]) < max(1, batch // 4):
        chunks[-2] = torch.cat([chunks[-2], chunks[-1]])
        chunks.pop()
    return chunks


def source_augment(x, y, step_m, rng):
    """SST's dropping, independent timing shift, and smoothed regression mixup."""
    p = x[0].clone()
    b, s, t = p.shape
    maxdrop = int(math.floor(s * 7 / 18))
    if maxdrop:
        counts = np.rint(maxdrop * rng.beta(0.1, 0.1, b)).astype(int)
        rows = []
        columns = []
        for i, c in enumerate(counts):
            if c:
                rows.extend([i] * int(c))
                columns.extend(rng.choice(s, c, replace=False).tolist())
        if rows:
            p[torch.as_tensor(rows, device=p.device), torch.as_tensor(columns, device=p.device)] = 0
    shifts = torch.as_tensor(
        np.rint(rng.normal(0, 25e-9 * 3e8 / step_m, (b, s))).astype(np.int64), device=p.device
    )
    indices = (torch.arange(t, device=p.device)[None, None, :] - shifts[..., None]) % t
    p = p.gather(-1, indices)
    weights = torch.exp(-torch.cdist(y, y).square() / 8.0)
    other = torch.multinomial(weights, 1).squeeze(-1)
    lam = torch.as_tensor(rng.beta(2, 2, b), dtype=p.dtype, device=p.device)
    return (lam[:, None, None] * p + (1 - lam[:, None, None]) * p[other],), other, lam


@torch.no_grad()
def predict(model, x, batch=128):
    model.eval()
    out = []
    for lo in range(0, len(x[0]), batch):
        out.append(model(tuple(v[lo : lo + batch] for v in x)).cpu().numpy())
    return np.concatenate(out)


def fit(method, config, data, training, validation, seed, path, epochs=None, evaluate_test=True):
    done = path / "complete.json"
    if done.exists():
        return json.loads(done.read_text())
    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    power = data["train_power"]
    xy = data["train_xy"]
    is_final = validation is None
    enc = Encoder(method, **config).fit(power[training])
    tx = tensor(enc.transform(power[training]))
    origin = xy[training].mean(0)
    scale = max(float(np.ptp(xy[training], axis=0).max()) / 2, 1e-3)
    ty = torch.as_tensor((xy[training] - origin) / scale, dtype=torch.float32, device=DEVICE)
    vy = xy[validation] if not is_final else None
    vx = tensor(enc.transform(power[validation])) if not is_final else None
    model = MODELS[method](power.shape[1], power.shape[2], **config).to(DEVICE)
    opt = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=0.001 if method == "sst" else 0
    )
    limit = epochs or CAPS[method]
    batch = 64 if method == "pnn" and power.shape[1] > 1 else 256
    history = []
    best = float("inf")
    best_epoch = 0
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, limit + 1):
        model.train()
        if method == "sst":
            warm = min(1.0, epoch / 50.0)
            lr = (
                1e-5 + (config["lr"] - 1e-5) * 0.5 * (1 + math.cos(math.pi * epoch / 2000))
            ) * warm
        elif method == "cao":
            lr = config["lr"] * 0.995**epoch
        else:
            lr = config["lr"]
        for group in opt.param_groups:
            group["lr"] = lr
        order = torch.randperm(len(training), device=DEVICE)
        for ids in minibatches(order, batch):
            x = tuple(z[ids] for z in tx)
            y = ty[ids]
            opt.zero_grad(set_to_none=True)
            if config["augment"]:
                x, other, lam = source_augment(x, y * scale, float(data["step_m"]), rng)
                pred = model(x)
                loss = (
                    lam * (pred - y).abs().mean(-1) + (1 - lam) * (pred - y[other]).abs().mean(-1)
                ).mean()
            else:
                pred = model(x)
                loss = F.l1_loss(pred, y) if method == "sst" else F.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        if epoch % 10 == 0 or epoch == limit:
            trpred = predict(model, tx) * scale + origin
            train_mean = float(np.linalg.norm(trpred - xy[training], axis=1).mean())
            if not is_final:
                valpred = predict(model, vx) * scale + origin
                val = float(np.linalg.norm(valpred - vy, axis=1).mean())
                if not np.isfinite(val + train_mean):
                    raise RuntimeError(f"Nonfinite training: {path}")
                if val < best:
                    best = val
                    best_epoch = epoch
                    best_state = cpu_state(model)
                history.append(dict(epoch=epoch, train_m=train_mean, validation_m=val, lr=lr))
                if epoch >= 200 and epoch - best_epoch >= 200:
                    break
            else:
                history.append(dict(epoch=epoch, train_m=train_mean, lr=lr))
            if epoch % 100 == 0:
                write(
                    path / "progress.json",
                    dict(epoch=epoch, history=history, seconds=time.perf_counter() - started),
                )
        if epoch % 250 == 0:
            log(
                progress=str(path.name),
                epoch=epoch,
                best_m=best if not is_final else None,
                seconds=round(time.perf_counter() - started, 1),
            )
    if not is_final:
        model.load_state_dict(best_state)
        valpred = predict(model, vx) * scale + origin
        np.savez_compressed(
            path / "validation.npz", truth=vy, prediction=valpred, indices=validation
        )
    bundle = dict(
        model=cpu_state(model),
        config=config,
        method=method,
        encoder=enc.__dict__,
        origin=origin,
        scale=scale,
        sensors=power.shape[1],
        bins=power.shape[2],
        seed=seed,
        training=training,
        history=history,
        epochs=epoch if is_final else best_epoch,
    )
    save_torch(path / "model.pt", bundle)
    record = dict(
        method=method,
        config=config,
        seed=seed,
        epochs=epoch if is_final else best_epoch,
        stopped_epoch=epoch,
        parameters=sum(p.numel() for p in model.parameters()),
        best_validation_m=best if not is_final else None,
        training_count=len(training),
        validation_count=0 if is_final else len(validation),
        seconds=time.perf_counter() - started,
        history=history,
        model_sha256=sha(path / "model.pt"),
    )
    if is_final and evaluate_test:
        qx = tensor(enc.transform(data["test_power"]))
        qpred = predict(model, qx) * scale + origin
        np.savez_compressed(path / "predictions.npz", truth=data["test_xy"], prediction=qpred)
        record["test_mean_m"] = float(np.linalg.norm(qpred - data["test_xy"], axis=1).mean())
    write(done, record)
    log(
        done=str(path.name),
        epoch=record["epochs"],
        validation_m=record["best_validation_m"],
        test_m=record.get("test_mean_m"),
        seconds=round(record["seconds"], 1),
    )
    return record


def run(ctx, tune=False, methods=("pnn", "sst", "cao"), seeds=(9101, 9102, 9103)):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    selection = read_json(ctx.root / "configs/networks.json")
    paths = sorted((ctx.data / "power").glob("*.npz"))
    if tune:
        for method in methods:
            for path in paths:
                with np.load(path) as z:
                    data = {k: z[k] for k in ("train_power", "train_xy", "fold", "step_m")}
                for fold in [0, 2] if path.stem == "uwb" else [1]:
                    tr = np.flatnonzero(data["fold"] != fold)
                    va = np.flatnonzero(data["fold"] == fold)
                    for config in CONFIGS[method]:
                        fit(
                            method,
                            config,
                            data,
                            tr,
                            va,
                            9101,
                            ctx.output
                            / "networks/validation"
                            / method
                            / path.stem
                            / f"fold{fold}"
                            / config["id"],
                        )
        for scope in ("nist", "uwb"):
            files = [p for p in paths if (p.stem == "uwb") == (scope == "uwb")]
            for method in methods:
                ranking = []
                for config in CONFIGS[method]:
                    records = []
                    for path in files:
                        for fold in [0, 2] if scope == "uwb" else [1]:
                            result = read_json(
                                ctx.output
                                / "networks/validation"
                                / method
                                / path.stem
                                / f"fold{fold}"
                                / config["id"]
                                / "complete.json"
                            )
                            records.append((path.stem, result))
                    error = sum(
                        r["best_validation_m"] * r["validation_count"] for _, r in records
                    ) / sum(r["validation_count"] for _, r in records)
                    epochs = {
                        p.stem: int(
                            round(np.mean([r["epochs"] for name, r in records if name == p.stem]))
                        )
                        for p in files
                    }
                    ranking.append(dict(config=config, validation_m=error, epochs=epochs))
                ranking.sort(key=lambda r: r["validation_m"])
                selection[f"{scope}_{method}"] = dict(selected=ranking[0], ranking=ranking)
        write(ctx.output / "networks/selection.json", selection)
    for method in methods:
        for path in paths:
            with np.load(path) as z:
                data = {k: z[k] for k in z.files}
            scope = "uwb" if path.stem == "uwb" else "nist"
            selected = selection[f"{scope}_{method}"]["selected"]
            for seed in seeds:
                fit(
                    method,
                    selected["config"],
                    data,
                    np.arange(len(data["train_xy"])),
                    None,
                    seed,
                    ctx.output / "networks/final" / method / path.stem / str(seed),
                    epochs=selected["epochs"][path.stem],
                )

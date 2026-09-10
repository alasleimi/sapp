"""Reproduction stages with explicit inputs and isolated run directories."""

import importlib.metadata
import platform
import time
from .common import read_json, write_json, sha

STAGES = (
    "prepare",
    "primary",
    "native",
    "surfaces",
    "optimized",
    "vt",
    "vt_diagnostics",
    "components",
    "sensitivity",
    "seeds",
    "counts",
    "cnn",
    "cnn_budget",
    "networks",
    "uwb",
    "uwb_matching",
    "figures",
    "tables",
)
PROFILES = {
    "primary": ["primary"],
    "native": ["native"],
    "uwb": ["uwb", "uwb_matching"],
    "paper": [s for s in STAGES if s != "prepare"],
}


def run(ctx, profile="paper", stages=None, tune=False):
    chosen = list(stages or PROFILES[profile])
    if set(chosen) - set(STAGES):
        raise ValueError(f"Unknown stages: {sorted(set(chosen) - set(STAGES))}")
    output = ctx.output.resolve()
    if ctx.root.resolve().is_relative_to(output) or any(
        output.is_relative_to(ctx.root / p)
        for p in ("data", "reference", "src", "configs", "paper")
    ):
        raise ValueError(
            "Choose an output directory separate from source, inputs and reference results"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run.json"
    config_hashes = {
        str(p.relative_to(ctx.root)): sha(p) for p in sorted((ctx.root / "configs").glob("*.json"))
    }
    code_hashes = {
        p.relative_to(ctx.root).as_posix(): sha(p)
        for p in sorted((ctx.root / "src/sapp").rglob("*.py"))
    }
    config = dict(
        profile=profile,
        tune=tune,
        configurations=config_hashes,
        source_files=code_hashes,
        input_manifest=sha(ctx.root / "data/manifest.json"),
        dependency_lock=sha(ctx.root / "uv.lock"),
    )
    if manifest.exists():
        record = read_json(manifest)
        if record["configuration"] != config:
            raise ValueError(
                "Run configuration differs from the existing run. Use another --output directory."
            )
    else:
        from .verify import verify_inputs

        verify_inputs(ctx.root)
        versions = {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "torch", "numba", "matplotlib")
        }
        record = dict(
            configuration=config, python=platform.python_version(), packages=versions, stages={}
        )
    from . import nist, uwb, diagnostics, optimized, cnn, cnn_budget, networks
    from . import tables, vt_diagnostics
    from sapp.plotting.generate import generate

    actions = {
        "primary": lambda: nist.run(ctx),
        "native": lambda: nist.run(ctx, native=True),
        "surfaces": lambda: diagnostics.surfaces(ctx),
        "optimized": lambda: optimized.run(ctx, tune),
        "vt": lambda: nist.virtual_transmitters(ctx),
        "vt_diagnostics": lambda: vt_diagnostics.run(ctx),
        "components": lambda: diagnostics.components(ctx),
        "sensitivity": lambda: diagnostics.sensitivity(ctx),
        "seeds": lambda: diagnostics.seeds(ctx),
        "counts": lambda: diagnostics.counts(ctx),
        "cnn": lambda: cnn.run(ctx, tune=True),
        "cnn_budget": lambda: cnn_budget.run(ctx, tune=True),
        "networks": lambda: networks.run(ctx, tune),
        "uwb": lambda: uwb.run(ctx, operating=profile == "paper"),
        "uwb_matching": lambda: uwb.matching_baselines(ctx),
        "figures": lambda: generate(ctx.root, output, output),
        "tables": lambda: tables.generate(ctx.root, output, output),
    }
    if "prepare" in chosen:
        from .prepare import prepare

        actions["prepare"] = lambda: prepare(ctx)
    if tune and "uwb" in chosen:
        uwb.validate(ctx)
    for stage in chosen:
        started = time.perf_counter()
        record["stages"][stage] = {"status": "running"}
        write_json(manifest, record)
        print(f"Starting {stage}", flush=True)
        try:
            actions[stage]()
        except Exception as error:
            record["stages"][stage] = {"status": "failed", "error": str(error)}
            write_json(manifest, record)
            raise
        record["stages"][stage] = {"status": "complete", "seconds": time.perf_counter() - started}
        write_json(manifest, record)
        print(f"Completed {stage}", flush=True)

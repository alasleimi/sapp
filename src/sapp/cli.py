"""Command-line entry points for SAPP and paper reproduction."""

import argparse
import os
from pathlib import Path


def repository_root(value=None):
    if value:
        root = Path(value).resolve()
    else:
        candidates = [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
        root = next(
            (
                p
                for p in candidates
                if (p / "data/nist/design.json").is_file() and (p / "pyproject.toml").is_file()
            ),
            None,
        )
    if root is None or not (root / "data/nist/design.json").is_file():
        raise ValueError("Pass --root with the path to the SAPP repository and data bundle")
    return root


def main():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    os.environ.setdefault("NUMBA_NUM_THREADS", "4")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(
        prog="sapp", description="SAPP localization and manuscript reproduction"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="Fit a bundled real survey and localize eight queries")
    for name, help_text in [
        ("reproduce", "Refit models, evaluate observations and generate the paper"),
        ("figures", "Regenerate all twelve vector figures"),
        ("tables", "Regenerate numerical result tables"),
        ("verify", "Check data hashes and reproduction metrics"),
        ("paper", "Build anonymous manuscript and supplement PDFs"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--root", type=Path, help="Repository containing data and configs")
        p.add_argument("--output", type=Path, default=Path("runs/reproduction"))
        if name == "reproduce":
            p.add_argument(
                "--profile", choices=["paper", "primary", "native", "uwb"], default="paper"
            )
            p.add_argument(
                "--stages", nargs="+", help="Run selected stages; see sapp reproduce --list-stages"
            )
            p.add_argument("--list-stages", action="store_true")
            p.add_argument(
                "--tune", action="store_true", help="Repeat training-only hyperparameter searches"
            )
        if name in ("figures", "tables"):
            p.add_argument(
                "--from",
                dest="source",
                default="run",
                help="run, reference, or a completed run directory",
            )
        if name == "figures":
            p.add_argument("--only", nargs="+", help="Selected figure basenames")
        if name == "verify":
            p.add_argument("--data-only", action="store_true")
            p.add_argument(
                "--profile", choices=["paper", "primary", "native", "uwb"], default="paper"
            )
    args = parser.parse_args()
    if args.command == "demo":
        from .demo import run

        run()
        return
    from .experiments.common import Context

    root = repository_root(args.root)
    output = args.output.resolve() if args.output.is_absolute() else root / args.output
    ctx = Context(root, output)
    if args.command == "reproduce":
        from .experiments.pipeline import run, STAGES

        if args.list_stages:
            print("\n".join(STAGES))
        else:
            run(ctx, args.profile, args.stages, args.tune)
    elif args.command in ("figures", "tables"):
        if args.command == "figures":
            from .plotting.generate import generate
        else:
            from .experiments.tables import generate
        source = (
            root / "reference"
            if args.source == "reference"
            else output
            if args.source == "run"
            else Path(args.source).resolve()
        )
        options = {"only": args.only} if args.command == "figures" else {}
        generate(root, source, output, **options)
    elif args.command == "verify":
        from .experiments.verify import verify

        verify(ctx, data_only=args.data_only, profile=args.profile)
    else:
        from .experiments.paper import build

        build(ctx)


if __name__ == "__main__":
    main()

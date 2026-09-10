"""Compile the anonymous manuscript and supplement with TeX Live."""

import re
import shutil
import subprocess
from .common import write_json


def build(ctx):
    if not shutil.which("pdflatex") or not shutil.which("bibtex"):
        raise RuntimeError("Install TeX Live with pdflatex and bibtex to build the PDFs")
    target = ctx.output / "paper"
    target.mkdir(parents=True, exist_ok=True)
    for path in (ctx.root / "paper").rglob("*"):
        if path.is_file() and path.suffix in (".tex", ".bib", ".cls", ".sty"):
            dest = target / path.relative_to(ctx.root / "paper")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    for folder in ("figures", "generated"):
        source = ctx.output / folder
        if not source.exists():
            if folder == "figures":
                raise RuntimeError("Generate the figures before building the manuscript")
            continue
        for path in source.iterdir():
            if path.suffix in (".pdf", ".tex"):
                dest = target / folder / path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
    for name in ("main", "supplement"):
        for tool in ("pdflatex", "bibtex", "pdflatex", "pdflatex"):
            if tool == "bibtex" and r"\bibdata" not in (target / f"{name}.aux").read_text():
                continue
            command = (
                [tool, "-interaction=nonstopmode", "-halt-on-error", name + ".tex"]
                if tool == "pdflatex"
                else [tool, name]
            )
            result = subprocess.run(
                command,
                cwd=target,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
            (target / f"{name}_{tool}.txt").write_text(result.stdout, encoding="utf-8")
            if result.returncode:
                raise RuntimeError(f"{tool} failed for {name}: see {target / name}.log")
        log = (target / f"{name}.log").read_text(errors="replace")
        if re.search(
            r"undefined (?:references|citations)|Reference .* undefined|Citation .* undefined", log
        ):
            raise RuntimeError(f"Unresolved references in {name}.log")
        print(f"Built {target / name}.pdf", flush=True)
    write_json(
        target / "build.json", dict(status="complete", documents=["main.pdf", "supplement.pdf"])
    )

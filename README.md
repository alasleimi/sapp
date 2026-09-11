# SAPP

**SAPP estimates an indoor receiver's position from multipath arrival times measured from an access point.** Training uses sets of arrival times recorded at known receiver coordinates. The paths are unlabeled: the survey does not identify which reflection produced each measurement. SAPP fits three components:

1. **Virtual-source geometry.** RANSAC proposes source locations using measurements from three survey positions, retains sources whose predicted delays agree with observations across the survey, and refines their locations by robust least squares. Each source predicts a path delay from its distance to a candidate receiver position.
2. **Spatial visibility.** SAPP assigns each survey peak a probability of belonging to each fitted source. Spatial kernel smoothing of these assignments estimates how many peaks each source is expected to produce at a given position, accounting for paths that appear and disappear across the room.
3. **Residual delay density.** Survey peaks that the fitted sources explain poorly contribute to a kernel density estimate over delay. Weighting these measurements by their proximity to a candidate position captures local delay patterns left unexplained by the source geometry.

To localize a new observation, SAPP combines the visibility-weighted source predictions and the residual density into a single delay intensity at each candidate position. A Poisson point-process likelihood scores the measured arrival times and their count. These scores give a probability distribution over receiver positions; SAPP returns its geometric median, the position that minimizes expected Euclidean error.

This repository includes SAPP, comparison methods, benchmark data, and commands to reproduce the paper's figures and tables.

## Main results

[Table 1 of the manuscript](paper/generated/main_table.tex) reports localization error for 5,400 simulated single-AP queries across three rooms, with 1,800 queries per room. The overall mean, median, P90, and fraction above two metres summarize all three rooms. Errors are in metres; lower values are better.

| Method | Input | L mean | T mean | Oblique mean | Overall mean | Median | P90 | >2 m (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MCA | Delays | 1.528 | 1.498 | 1.845 | 1.624 | 1.044 | 4.145 | 31.7 |
| [VT interpolation (Zayets & Steinbach, 2018)](https://doi.org/10.1109/ICC.2018.8422206) | Delays | 1.720 | 1.588 | 1.926 | 1.745 | 1.278 | 3.939 | 34.9 |
| Chamfer weighted kNN | Delays | 1.372 | 1.297 | 1.499 | 1.389 | 1.023 | 3.143 | 23.6 |
| MPUrge-MAP | Delays | 1.338 | 1.278 | 1.585 | 1.400 | 1.067 | 3.127 | 24.4 |
| CNN regressor | Delays | 1.652 | 1.442 | 1.643 | 1.579 | 1.339 | 3.012 | 28.5 |
| SAPP (mode) | Delays | **0.873** | 1.114 | 1.614 | 1.200 | **0.494** | 3.515 | 22.9 |
| SAPP (geometric median) | Delays | 0.903 | **1.102** | **1.361** | **1.122** | 0.757 | **2.666** | **18.4** |
| P-NN | Power | 1.904 | 1.513 | 1.604 | 1.674 | 1.418 | 3.161 | 33.2 |

Delay methods use up to nine detected peaks; P-NN uses 24 power bins. Neural metrics average three independent fits. Best values in each column are bold. See [the baseline guide](docs/baselines.md) for the cited methods and their experimental settings.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from this directory:

```sh
uv sync --locked
uv run --locked sapp demo
```

The demo fits the bundled L-room survey and localizes eight observations. Python 3.11 and the core numerical dependencies are pinned. The core package requires NumPy and SciPy.

## Reproduce the paper

For a CPU environment:

```sh
uv sync --locked --extra paper
uv run --locked --extra paper sapp verify --data-only
uv run --locked --extra paper sapp reproduce
uv run --locked --extra paper sapp verify
uv run --locked --extra paper sapp paper
```

For an NVIDIA GPU, use `--extra cuda` in place of `--extra paper`. This selects the pinned CUDA 12.1 PyTorch build. The two extras are alternative environments; [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) explains the package indexes. Core SAPP fitting uses the CPU. Query scoring and the larger neural experiments can use the GPU.

`reproduce` refits SAPP and the baselines from the bundled observations, evaluates the held-out queries, and writes figures, tables, and predictions under `runs/reproduction/`. It uses the published training-selected settings. Add `--tune` to repeat the additional spatial validation searches. The CNN learning-curve experiments include their training and duration selection in the default run.

To also regenerate the NIST observations from the bundled native channel records:

```sh
uv run --locked --extra paper sapp reproduce --stages prepare
uv run --locked --extra paper sapp reproduce
```

The preparation stage rebuilds the receiver peaks and exact-delay inputs and checks them against the released inputs. These commands use local data. Dependency installation may use the network. Neural training is the slowest part of the full reproduction; choose stages when checking an individual result.

To redraw the paper figures immediately from the archived predictions:

```sh
uv run --locked --extra paper sapp figures --from reference --output runs/reference
uv run --locked --extra paper sapp tables --from reference --output runs/reference
uv run --locked --extra paper sapp paper --output runs/reference
```

The final PDF command requires TeX Live with `pdflatex` and `bibtex`. Figures are generated as vector PDFs and PNG previews. The manuscript and supplement appear in `runs/reference/paper/` or `runs/reproduction/paper/`.

See [the reproduction guide](docs/reproduction.md) for stage dependencies, result checks, and data provenance, and [the release verification](docs/verification.md) for measured reproduction results.

## Use SAPP in Python

```python
import numpy as np
from sapp import AnchorParameters, fit, localize, save_anchor_map

# survey_xy: shape (N, 2), in metres.
# survey_ranges: N arrays of observed ranges, in metres.
# query_ranges: one array per query observation.
parameters = AnchorParameters(decode="posterior_median")
model = fit(survey_xy, survey_ranges, parameters)
positions = localize(query_ranges, model)
save_anchor_map("map.npz", model)
```

Convert synchronized path delays to ranges using the propagation speed appropriate to the data. Each observation can contain a different number of peaks. The model uses known survey coordinates on a fixed receiver plane. Its inputs do not include a floor plan or AP coordinates. See [the API and data guide](docs/data-and-api.md).

## Structure

```text
src/sapp/               Source discovery, fitting, intensity fields, localization
  baselines/            Matching, VT interpolation, CNN and recent neural models
  experiments/          Data preparation, fitting, evaluation, tables, verification
  plotting/             The twelve manuscript and supplement figures
configs/                Published parameters and training selections
data/                   Local observations, native channels, experiment design
reference/              Published predictions and fitted maps used for comparison
paper/                  Anonymous manuscript and supplement sources
tests/                  Numerical regression and algorithm tests
runs/                   Generated outputs, excluded from Git
```

[Baselines](docs/baselines.md) lists the implementations and their papers. [Data provenance](docs/data-and-api.md#data-provenance) describes the bundled NIST and Fraunhofer inputs.

The baseline modules are named after their methods:

```python
from sapp.baselines import mpurge_map, mca

mpurge_positions = mpurge_map.localize(query_sets, survey_sets, survey_xy)
mca_position = mca.localize(query_sets[0], survey_sets, survey_xy, epsilon_m=0.5)
```

Both use range sets and survey coordinates in metres. MPUrge-MAP accepts a batch of queries; MCA accepts one query.

## Development

```sh
uv sync --locked --extra paper --group dev
uv run --locked --extra paper python -m pytest
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
```
